//! Encapsulated CUDA wrapper for poseidon_aggregator.
//!
//! Internally uses SIMD `poseidon_aggregator::ClaimGenerator::write_trace()` to compute the
//! 342-column trace, then converts the output to CudaBackend inline. From the orchestration
//! layer's perspective this exposes a CUDA-native API — no `SimdToCudaBridge` is visible.

use std::sync::Arc;

use cairo_air::components::poseidon_aggregator;
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_common::prover_types::simd::PackedFelt252Width27;

use crate::witness::cairo_cuda::convert_simd_to_cuda_evaluation;
use crate::witness::components::{
    cube_252, memory_id_to_big, poseidon_3_partial_rounds_chain,
    poseidon_aggregator as poseidon_aggregator_witness, poseidon_full_round_chain,
    range_check_252_width_27, range_check_3_3_3_3_3, range_check_4_4, range_check_4_4_4_4,
};
use crate::witness::components_cuda::poseidon_cuda::PoseidonContextCudaClaimGenerator;
use crate::witness::components_cuda::{
    memory_id_to_big_cuda, poseidon_3_partial_rounds_chain_cuda, poseidon_full_round_chain_cuda,
};
use crate::witness::range_checks_cuda::RangeChecksCudaClaimGenerator;
use crate::witness::utils::TreeBuilder;

// CUDA-COVERAGE: poseidon_aggregator — SIMD hybrid path.
// Internally holds simd_aggregator (poseidon_aggregator_witness::ClaimGenerator).
// Runs SIMD write_trace() → converts 342-col trace to CUDA inline →
// merges multiplicities into CUDA for downstream chain components.
// Depends on poseidon_builtin also being SIMD (feeds SIMD aggregator inputs).

/// Encapsulated CUDA wrapper around the SIMD poseidon_aggregator.
///
/// Hides all SIMD internals. The orchestration layer only sees a CUDA-native API.
pub struct PoseidonAggregatorCudaClaimGenerator {
    simd_aggregator: poseidon_aggregator_witness::ClaimGenerator,
    memory: Arc<Memory>,
    preprocessed_trace: Arc<PreProcessedTrace>,
}

impl PoseidonAggregatorCudaClaimGenerator {
    pub fn new(
        simd_aggregator: poseidon_aggregator_witness::ClaimGenerator,
        memory: Arc<Memory>,
        preprocessed_trace: Arc<PreProcessedTrace>,
    ) -> Self {
        Self {
            simd_aggregator,
            memory,
            preprocessed_trace,
        }
    }

    /// Write the aggregator base trace (342 cols) and feed downstream CUDA generators.
    ///
    /// Internally runs the SIMD aggregator, converts output to CUDA, and feeds:
    /// - memory_id_to_big_cuda (merged multiplicities)
    /// - poseidon_context_cuda (chain packed_inputs)
    /// - range_checks (merged multiplicities)
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        poseidon_context_cuda: &mut Option<PoseidonContextCudaClaimGenerator>,
        memory_id_to_big_cuda: &mut memory_id_to_big_cuda::CudaClaimGenerator,
        range_checks_trace_generator: &mut RangeChecksCudaClaimGenerator,
    ) -> (
        poseidon_aggregator::Claim,
        PoseidonAggregatorCudaInteractionClaimGenerator,
    ) {
        // Create SIMD chain state objects (aggregator will populate their packed_inputs)
        let mut simd_full_round_chain = poseidon_full_round_chain::ClaimGenerator::new();
        let mut simd_3_partial = poseidon_3_partial_rounds_chain::ClaimGenerator::new();
        let mut simd_cube_252 = cube_252::ClaimGenerator::new();
        let mut simd_rc_252w27 = range_check_252_width_27::ClaimGenerator::new();

        // Range check generators (aggregator-level only — chain-level RCs
        // are handled internally by CUDA chain generators)
        let simd_rc_3_3_3_3_3 =
            range_check_3_3_3_3_3::ClaimGenerator::new(self.preprocessed_trace.clone());
        let simd_rc_4_4_4_4 =
            range_check_4_4_4_4::ClaimGenerator::new(self.preprocessed_trace.clone());
        let simd_rc_4_4 = range_check_4_4::ClaimGenerator::new(self.preprocessed_trace.clone());

        // Memory id_to_big for deduce_output
        let simd_mem_id_to_big = memory_id_to_big::ClaimGenerator::new(self.memory.clone());

        // 1. Run SIMD aggregator write_trace
        let (pa_trace, pa_claim, pa_interaction_gen) = self.simd_aggregator.write_trace(
            &simd_mem_id_to_big,
            &mut simd_full_round_chain,
            &mut simd_rc_252w27,
            &mut simd_cube_252,
            &simd_rc_3_3_3_3_3,
            &simd_rc_4_4_4_4,
            &simd_rc_4_4,
            &mut simd_3_partial,
        );

        // 2. Convert SIMD trace to CUDA inline (no SimdToCudaBridge)
        tree_builder.extend_evals(
            pa_trace
                .to_evals()
                .into_iter()
                .map(convert_simd_to_cuda_evaluation)
                .collect(),
        );

        // 3. Merge SIMD memory_id_to_big multiplicities into CUDA
        {
            let big_mults: Vec<u32> = simd_mem_id_to_big
                .big_mults
                .into_simd_vec()
                .into_iter()
                .flat_map(|p| p.to_array().map(|v| v.0))
                .collect();
            let small_mults: Vec<u32> = simd_mem_id_to_big
                .small_mults
                .into_simd_vec()
                .into_iter()
                .flat_map(|p| p.to_array().map(|v| v.0))
                .collect();
            memory_id_to_big_cuda.merge_simd_multiplicities(&big_mults, &small_mults);
        }

        // 4. Convert SIMD chain packed_inputs → CUDA format and feed CUDA generators
        if let Some(ref mut ctx) = poseidon_context_cuda {
            // Full round chain
            let full_round_cuda_inputs =
                simd_to_cuda_full_round_chain(&simd_full_round_chain.packed_inputs);
            ctx.poseidon_full_round_chain_cuda
                .add_cuda_inputs(&full_round_cuda_inputs);

            // 3 partial rounds chain
            let partial_cuda_inputs = simd_to_cuda_3_partial(&simd_3_partial.packed_inputs);
            ctx.poseidon_3_partial_rounds_chain_cuda
                .add_cuda_inputs(&partial_cuda_inputs);

            // cube_252 (aggregator's direct contribution)
            let cube_cuda_inputs =
                packed_felt252w27_to_basefieldvec_10(&simd_cube_252.packed_inputs);
            ctx.cube_252_cuda.add_cuda_inputs(&cube_cuda_inputs);

            // range_check_252_width_27 (aggregator's direct contribution)
            let rc_cuda_inputs =
                packed_felt252w27_to_basefieldvec_10(&simd_rc_252w27.packed_inputs);
            ctx.range_check_252_width_27_cuda
                .add_cuda_inputs(&rc_cuda_inputs);
        }

        // 5. Merge aggregator-level range check multiplicities only
        // (chain-level RCs: rc_9_9, rc_20, rc_18 are handled by CUDA chain generators)
        macro_rules! merge_single_rc {
            ($simd_gen:expr, $cuda_gen:expr) => {{
                let mults_arr = $simd_gen.mults;
                for rc_mult in mults_arr {
                    let mults: Vec<u32> = rc_mult
                        .into_simd_vec()
                        .into_iter()
                        .flat_map(|p| p.to_array().map(|v| v.0))
                        .collect();
                    $cuda_gen.merge_simd_multiplicities(&mults);
                }
            }};
        }
        merge_single_rc!(
            simd_rc_3_3_3_3_3,
            range_checks_trace_generator.rc_3_3_3_3_3_trace_generator
        );
        merge_single_rc!(
            simd_rc_4_4_4_4,
            range_checks_trace_generator.rc_4_4_4_4_trace_generator
        );
        merge_single_rc!(
            simd_rc_4_4,
            range_checks_trace_generator.rc_4_4_trace_generator
        );

        (
            pa_claim,
            PoseidonAggregatorCudaInteractionClaimGenerator {
                simd_interaction_gen: pa_interaction_gen,
            },
        )
    }
}

/// Encapsulated interaction claim generator for poseidon_aggregator.
///
/// Internally uses the SIMD interaction generator and converts output to CUDA inline.
pub struct PoseidonAggregatorCudaInteractionClaimGenerator {
    simd_interaction_gen: poseidon_aggregator_witness::InteractionClaimGenerator,
}

impl PoseidonAggregatorCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        common_lookup_elements: &CommonLookupElements,
    ) -> poseidon_aggregator::InteractionClaim {
        let (evals, interaction_claim) = self
            .simd_interaction_gen
            .write_interaction_trace(common_lookup_elements);

        // Convert SIMD evals to CUDA inline (no SimdToCudaBridge)
        tree_builder.extend_evals(
            evals
                .into_iter()
                .map(convert_simd_to_cuda_evaluation)
                .collect(),
        );

        interaction_claim
    }
}

// ---------------------------------------------------------------------------
// SIMD → CUDA conversion helpers (moved from cairo_cuda.rs)
// ---------------------------------------------------------------------------

/// Convert a slice of PackedFelt252Width27 values to 10 BaseFieldVec columns.
fn packed_felt252w27_to_basefieldvec_10(packed: &[PackedFelt252Width27]) -> [BaseFieldVec; 10] {
    let n = packed.len() * N_LANES;
    let mut cols: [Vec<M31>; 10] = std::array::from_fn(|_| Vec::with_capacity(n));
    for p in packed {
        let arr = p.to_array(); // [Felt252Width27; N_LANES]
        for limb_idx in 0..10 {
            for lane in 0..N_LANES {
                cols[limb_idx].push(arr[lane].get_m31(limb_idx));
            }
        }
    }
    cols.map(|v| BaseFieldVec::from_vec(v))
}

/// Convert SIMD poseidon_full_round_chain packed_inputs to CUDA format.
fn simd_to_cuda_full_round_chain(
    packed_inputs: &[(PackedM31, PackedM31, [PackedFelt252Width27; 3])],
) -> poseidon_full_round_chain_cuda::CudaPackedInputType {
    let n = packed_inputs.len() * N_LANES;
    let mut limb0 = Vec::with_capacity(n);
    let mut limb1 = Vec::with_capacity(n);
    let mut state: [[Vec<M31>; 10]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(n)));
    for (l0, l1, states) in packed_inputs {
        limb0.extend(l0.to_array());
        limb1.extend(l1.to_array());
        for (s_idx, felt) in states.iter().enumerate() {
            let felt_arr = felt.to_array();
            for limb_idx in 0..10 {
                for lane in 0..N_LANES {
                    state[s_idx][limb_idx].push(felt_arr[lane].get_m31(limb_idx));
                }
            }
        }
    }
    poseidon_full_round_chain_cuda::CudaPackedInputType {
        input_limb_0: BaseFieldVec::from_vec(limb0),
        input_limb_1: BaseFieldVec::from_vec(limb1),
        state_0: state[0]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
        state_1: state[1]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
        state_2: state[2]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
    }
}

/// Convert SIMD poseidon_3_partial_rounds_chain packed_inputs to CUDA format.
fn simd_to_cuda_3_partial(
    packed_inputs: &[(PackedM31, PackedM31, [PackedFelt252Width27; 4])],
) -> poseidon_3_partial_rounds_chain_cuda::CudaPackedInputType {
    let n = packed_inputs.len() * N_LANES;
    let mut limb0 = Vec::with_capacity(n);
    let mut limb1 = Vec::with_capacity(n);
    let mut state: [[Vec<M31>; 10]; 4] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(n)));
    for (l0, l1, states) in packed_inputs {
        limb0.extend(l0.to_array());
        limb1.extend(l1.to_array());
        for (s_idx, felt) in states.iter().enumerate() {
            let felt_arr = felt.to_array();
            for limb_idx in 0..10 {
                for lane in 0..N_LANES {
                    state[s_idx][limb_idx].push(felt_arr[lane].get_m31(limb_idx));
                }
            }
        }
    }
    poseidon_3_partial_rounds_chain_cuda::CudaPackedInputType {
        input_limb_0: BaseFieldVec::from_vec(limb0),
        input_limb_1: BaseFieldVec::from_vec(limb1),
        state_0: state[0]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
        state_1: state[1]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
        state_2: state[2]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
        state_3: state[3]
            .each_ref()
            .map(|v| BaseFieldVec::from_vec(v.clone())),
    }
}
