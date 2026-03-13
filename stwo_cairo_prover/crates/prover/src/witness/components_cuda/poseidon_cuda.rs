//! CUDA version of poseidon context claim generator.
//!
//! Architecture:
//! - poseidon_full_round_chain runs via **CUDA** (trace + interaction trace)
//! - poseidon_3_partial_rounds_chain runs via **CUDA** (trace + interaction trace)
//! - cube_252 runs via **CUDA** (trace + interaction trace)
//! - poseidon_round_keys runs via SIMD (static table, 64 rows)
//! - range_check_252_width_27 runs via **CUDA** (trace + interaction trace)
//!
// CUDA-COVERAGE: poseidon_round_keys — pure SIMD (static lookup table, 64 rows).
// Not worth migrating to CUDA: zero parallelism benefit, runs via SimdToCudaBridge.

use cairo_air::components::{
    cube_252, poseidon_3_partial_rounds_chain, poseidon_full_round_chain, poseidon_round_keys,
    range_check_252_width_27,
};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Column;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use tracing::{span, Level};

use crate::witness::cairo_cuda::SimdToCudaBridge;
use crate::witness::components::poseidon_round_keys as poseidon_round_keys_witness;

/// Local grouped claim type (mirrors legacy `cairo_air::poseidon::air::Claim`).
pub struct PoseidonContextClaim {
    pub claim: Option<Claim>,
}

pub struct Claim {
    pub poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain::Claim,
    pub poseidon_full_round_chain: poseidon_full_round_chain::Claim,
    pub cube_252: cube_252::Claim,
    pub poseidon_round_keys: poseidon_round_keys::Claim,
    pub range_check_252_width_27: range_check_252_width_27::Claim,
}

pub struct PoseidonContextInteractionClaim {
    pub claim: Option<InteractionClaim>,
}

pub struct InteractionClaim {
    pub poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain::InteractionClaim,
    pub poseidon_full_round_chain: poseidon_full_round_chain::InteractionClaim,
    pub cube_252: cube_252::InteractionClaim,
    pub poseidon_round_keys: poseidon_round_keys::InteractionClaim,
    pub range_check_252_width_27: range_check_252_width_27::InteractionClaim,
}
use crate::witness::components_cuda::{
    cube_252_cuda, poseidon_3_partial_rounds_chain_cuda, poseidon_full_round_chain_cuda,
    range_check_252_width_27_cuda,
};
use crate::witness::range_checks_cuda::RangeChecksCudaClaimGenerator;
use crate::witness::utils::TreeBuilder;

pub struct PoseidonContextCudaClaimGenerator {
    /// CUDA generators for trace + interaction trace
    pub poseidon_full_round_chain_cuda: poseidon_full_round_chain_cuda::CudaClaimGenerator,
    pub poseidon_3_partial_rounds_chain_cuda:
        poseidon_3_partial_rounds_chain_cuda::CudaClaimGenerator,
    pub cube_252_cuda: cube_252_cuda::CudaClaimGenerator,
    pub range_check_252_width_27_cuda: range_check_252_width_27_cuda::CudaClaimGenerator,
    /// SIMD round_keys (tiny, 64 rows -- stays SIMD)
    poseidon_round_keys_trace_generator: poseidon_round_keys_witness::ClaimGenerator,
    /// Preprocessed trace (for loading round keys table to GPU)
    preprocessed_trace: std::sync::Arc<PreProcessedTrace>,
}

impl PoseidonContextCudaClaimGenerator {
    pub fn new(preprocessed_trace: std::sync::Arc<PreProcessedTrace>) -> Self {
        Self {
            poseidon_full_round_chain_cuda: poseidon_full_round_chain_cuda::CudaClaimGenerator::new(
            ),
            poseidon_3_partial_rounds_chain_cuda:
                poseidon_3_partial_rounds_chain_cuda::CudaClaimGenerator::new(),
            cube_252_cuda: cube_252_cuda::CudaClaimGenerator::new(),
            range_check_252_width_27_cuda: range_check_252_width_27_cuda::CudaClaimGenerator::new(),
            poseidon_round_keys_trace_generator: poseidon_round_keys_witness::ClaimGenerator::new(
                preprocessed_trace.clone(),
            ),
            preprocessed_trace,
        }
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        range_checks_trace_generator: &RangeChecksCudaClaimGenerator,
    ) -> (
        PoseidonContextClaim,
        PoseidonContextCudaInteractionClaimGenerator,
    ) {
        let span = span!(Level::INFO, "write poseidon context trace (CUDA)").entered();

        if self.poseidon_full_round_chain_cuda.is_empty()
            && self.poseidon_3_partial_rounds_chain_cuda.is_empty()
        {
            return (
                PoseidonContextClaim { claim: None },
                PoseidonContextCudaInteractionClaimGenerator { gen: None },
            );
        }

        // ==== Step 2: Extract round_keys multiplicities from CUDA input_limb_1 ====
        // Download round numbers from CUDA generators (2 .to_cpu() calls total)
        {
            use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};

            let round_nums = self.poseidon_full_round_chain_cuda.input_limb_1.to_cpu();
            let n_total = round_nums.len();
            let n_packed = (n_total + N_LANES - 1) / N_LANES;
            let padded_n_packed = n_packed.next_power_of_two();
            for chunk_start in (0..n_total).step_by(N_LANES) {
                let arr: [M31; N_LANES] = std::array::from_fn(|i| {
                    if chunk_start + i < n_total {
                        round_nums[chunk_start + i]
                    } else {
                        round_nums[0]
                    }
                });
                self.poseidon_round_keys_trace_generator
                    .add_packed_inputs(&[[PackedM31::from_array(arr)]], 0);
            }
            if padded_n_packed > n_packed {
                let first = PackedM31::from_array(std::array::from_fn(|i| {
                    if i < n_total {
                        round_nums[i]
                    } else {
                        round_nums[0]
                    }
                }));
                for _ in n_packed..padded_n_packed {
                    self.poseidon_round_keys_trace_generator
                        .add_packed_inputs(&[[first]], 0);
                }
            }
        }
        {
            use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};

            let round_nums = self
                .poseidon_3_partial_rounds_chain_cuda
                .input_limb_1
                .to_cpu();
            let n_total = round_nums.len();
            let n_packed = (n_total + N_LANES - 1) / N_LANES;
            let padded_n_packed = n_packed.next_power_of_two();
            for chunk_start in (0..n_total).step_by(N_LANES) {
                let arr: [M31; N_LANES] = std::array::from_fn(|i| {
                    if chunk_start + i < n_total {
                        round_nums[chunk_start + i]
                    } else {
                        round_nums[0]
                    }
                });
                self.poseidon_round_keys_trace_generator
                    .add_packed_inputs(&[[PackedM31::from_array(arr)]], 0);
            }
            if padded_n_packed > n_packed {
                let first = PackedM31::from_array(std::array::from_fn(|i| {
                    if i < n_total {
                        round_nums[i]
                    } else {
                        round_nums[0]
                    }
                }));
                for _ in n_packed..padded_n_packed {
                    self.poseidon_round_keys_trace_generator
                        .add_packed_inputs(&[[first]], 0);
                }
            }
        }

        // ==== Step 3: Load round keys table to GPU ====
        let round_keys_table = load_round_keys_to_gpu(&self.preprocessed_trace);

        // ==== Step 4: Run CUDA poseidon_3_partial_rounds_chain ====
        let (
            poseidon_3_partial_rounds_chain_claim,
            poseidon_3_partial_rounds_chain_interaction_gen,
        ) = self.poseidon_3_partial_rounds_chain_cuda.write_trace(
            tree_builder,
            &mut self.cube_252_cuda,
            &round_keys_table,
            &range_checks_trace_generator.rc_4_4_trace_generator,
            &range_checks_trace_generator.rc_4_4_4_4_trace_generator,
            &mut self.range_check_252_width_27_cuda,
        );

        // ==== Step 5: Run CUDA poseidon_full_round_chain ====
        let (poseidon_full_round_chain_claim, poseidon_full_round_chain_interaction_gen) =
            self.poseidon_full_round_chain_cuda.write_trace(
                tree_builder,
                &mut self.cube_252_cuda,
                &round_keys_table,
                &range_checks_trace_generator.rc_3_3_3_3_3_trace_generator,
            );

        // ==== Step 6: Run CUDA cube_252 ====
        let (cube_252_claim, cube_252_interaction_gen) = self.cube_252_cuda.write_trace(
            tree_builder,
            &range_checks_trace_generator.rc_9_9_trace_generator,
            &range_checks_trace_generator.rc_20_trace_generator,
        );

        // ==== Step 7: Run poseidon_round_keys via SIMD (bridge to CUDA) ====
        let (poseidon_round_keys_claim, poseidon_round_keys_interaction_gen) = {
            let (trace, claim, interaction_gen) =
                self.poseidon_round_keys_trace_generator.write_trace();
            let mut bridge = SimdToCudaBridge::new(tree_builder);
            bridge.extend_evals(trace.to_evals());
            (claim, interaction_gen)
        };

        // ==== Step 8: Run CUDA range_check_252_width_27 ====
        let (range_check_252_width_27_claim, range_check_252_width_27_interaction_gen) =
            self.range_check_252_width_27_cuda.write_trace(
                tree_builder,
                &range_checks_trace_generator.rc_18_trace_generator,
                &range_checks_trace_generator.rc_9_9_trace_generator,
            );

        span.exit();

        let claim = Some(Claim {
            poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain_claim,
            poseidon_full_round_chain: poseidon_full_round_chain_claim,
            cube_252: cube_252_claim,
            poseidon_round_keys: poseidon_round_keys_claim,
            range_check_252_width_27: range_check_252_width_27_claim,
        });
        let gen = Some(InteractionClaimGenerator {
            poseidon_3_partial_rounds_chain_interaction_gen,
            poseidon_full_round_chain_interaction_gen,
            cube_252_interaction_gen,
            poseidon_round_keys_interaction_gen,
            range_check_252_width_27_interaction_gen,
        });
        (
            PoseidonContextClaim { claim },
            PoseidonContextCudaInteractionClaimGenerator { gen },
        )
    }
}

/// Interaction claim generator for poseidon context (hybrid SIMD/CUDA).
pub struct PoseidonContextCudaInteractionClaimGenerator {
    gen: Option<InteractionClaimGenerator>,
}

impl PoseidonContextCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> PoseidonContextInteractionClaim {
        PoseidonContextInteractionClaim {
            claim: self
                .gen
                .map(|gen| gen.write_interaction_trace(tree_builder, lookup_elements)),
        }
    }
}

struct InteractionClaimGenerator {
    poseidon_round_keys_interaction_gen: poseidon_round_keys_witness::InteractionClaimGenerator,
    poseidon_full_round_chain_interaction_gen:
        poseidon_full_round_chain_cuda::CudaInteractionClaimGenerator,
    poseidon_3_partial_rounds_chain_interaction_gen:
        poseidon_3_partial_rounds_chain_cuda::CudaInteractionClaimGenerator,
    cube_252_interaction_gen: cube_252_cuda::CudaInteractionClaimGenerator,
    range_check_252_width_27_interaction_gen:
        range_check_252_width_27_cuda::CudaInteractionClaimGenerator,
}

impl InteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        // Order MUST match Claim struct field order
        let poseidon_3_partial_rounds_chain_interaction_claim = self
            .poseidon_3_partial_rounds_chain_interaction_gen
            .write_interaction_trace(tree_builder, lookup_elements);

        let poseidon_full_round_chain_interaction_claim = self
            .poseidon_full_round_chain_interaction_gen
            .write_interaction_trace(tree_builder, lookup_elements);

        let cube_252_interaction_claim = self
            .cube_252_interaction_gen
            .write_interaction_trace(tree_builder, lookup_elements);

        let poseidon_round_keys_interaction_claim = {
            let (interaction_trace, interaction_claim) = self
                .poseidon_round_keys_interaction_gen
                .write_interaction_trace(lookup_elements);
            let mut bridge = SimdToCudaBridge::new(tree_builder);
            bridge.extend_evals(interaction_trace);
            interaction_claim
        };

        let range_check_252_width_27_interaction_claim = self
            .range_check_252_width_27_interaction_gen
            .write_interaction_trace(tree_builder, lookup_elements);

        InteractionClaim {
            poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain_interaction_claim,
            poseidon_full_round_chain: poseidon_full_round_chain_interaction_claim,
            cube_252: cube_252_interaction_claim,
            poseidon_round_keys: poseidon_round_keys_interaction_claim,
            range_check_252_width_27: range_check_252_width_27_interaction_claim,
        }
    }
}

/// Loads the poseidon round keys table from the preprocessed trace to GPU memory.
fn load_round_keys_to_gpu(preprocessed_trace: &PreProcessedTrace) -> [[BaseFieldVec; 10]; 3] {
    use cairo_air::components::poseidon_round_keys::LOG_SIZE;

    let log_n_lanes = stwo::prover::backend::simd::m31::LOG_N_LANES;
    let n_packed_rows = 1usize << (LOG_SIZE - log_n_lanes);

    std::array::from_fn(|group| {
        std::array::from_fn(|col| {
            let col_idx = group * 10 + col;
            let column = preprocessed_trace.get_column(&PreProcessedColumnId {
                id: format!("poseidon_round_keys_{}", col_idx),
            });
            let mut values = Vec::with_capacity(1usize << LOG_SIZE);
            for vec_row in 0..n_packed_rows {
                let packed = column.packed_at(vec_row);
                values.extend(packed.to_array());
            }
            BaseFieldVec::from_vec(values)
        })
    })
}
