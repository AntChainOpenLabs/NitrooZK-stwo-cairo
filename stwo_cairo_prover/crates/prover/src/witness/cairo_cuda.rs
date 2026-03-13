//! SIMD/CUDA trace conversion utilities and Cairo CUDA claim generator.
//!
//! This module provides:
//! - Conversion utilities between SIMD and CUDA CircleEvaluations.
//! - `SimdTraceCollector` and `SimdToCudaBridge` for bridging SIMD and CUDA tree builders.
//! - `CairoCudaClaimGenerator` (V0 path): SIMD fallback via `SimdToCudaBridge`.
//! - `NativeCairoCudaClaimGenerator` (native path): Per-component native CUDA trace gen.
//! - Native CUDA wrapper for poseidon_builtin (split) and poseidon_aggregator.

use std::array;
use std::collections::HashSet;
use std::sync::Arc;

use cairo_air::air::{
    MemorySmallValue, PublicData, PublicMemory, PublicSegmentRanges, SegmentRange,
};
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::TreeSubspan;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_adapter::opcodes::CasmStatesByOpcode;
use stwo_cairo_adapter::{ProverInput, PublicSegmentContext};
use stwo_cairo_common::builtins::*;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, MAX_SEQUENCE_LOG_SIZE,
};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use tracing::{span, Level};

use super::cairo::create_cairo_claim_generator;
use super::cairo_claim_generator::{CairoClaimGenerator, CairoInteractionClaimGenerator};
use super::components_cuda::builtins::{
    add_mod_builtin_cuda, bitwise_builtin_cuda, mul_mod_builtin_cuda, pedersen_builtin_cuda,
    pedersen_builtin_narrow_cuda, poseidon_builtin_cuda, poseidon_builtin_split_cuda,
    range_check_bits_128_builtin_cuda, range_check_bits_96_builtin_cuda,
};
use super::components_cuda::{
    blake_g_cuda, blake_round_cuda, blake_round_sigma_cuda, memory_address_to_id_cuda,
    memory_id_to_big_cuda, triple_xor_32_cuda, vbx_12_cuda, vbx_4_cuda, vbx_7_cuda, vbx_8_b_cuda,
    vbx_8_cuda, vbx_9_cuda, verify_instruction_cuda,
};
use super::opcodes_cuda::{OpcodesCudaClaimGenerator, OpcodesCudaInteractionClaimGenerator};
use super::range_checks_cuda::{
    RangeChecksCudaClaimGenerator, RangeChecksCudaInteractionClaimGenerator,
};
use crate::witness::components_cuda::pedersen_cuda::{
    PedersenContextCudaClaimGenerator, PedersenContextCudaInteractionClaimGenerator,
    PedersenInteractionMode,
};
use crate::witness::components_cuda::poseidon_aggregator_native_cuda;
use crate::witness::components_cuda::poseidon_cuda::{
    PoseidonContextCudaClaimGenerator, PoseidonContextCudaInteractionClaimGenerator,
};
use crate::witness::components_cuda::{
    pedersen_aggregator_cuda, pedersen_aggregator_wb9_cuda, pedersen_cuda, pedersen_wb9_cuda,
};
use crate::witness::utils::TreeBuilder;

// ---------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------

/// Convert a SimdBackend CircleEvaluation into a CudaBackend version.
pub fn convert_simd_to_cuda_evaluation(
    eval: CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
) -> CircleEvaluation<CudaBackend, M31, BitReversedOrder> {
    CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(
        eval.domain,
        <CudaBackend as ColumnOps<M31>>::Column::from_iter(eval.values.to_cpu()),
    )
}

/// Convert a CudaBackend CircleEvaluation into a SimdBackend version.
pub fn convert_cuda_to_simd_evaluation(
    eval: CircleEvaluation<CudaBackend, M31, BitReversedOrder>,
) -> CircleEvaluation<SimdBackend, M31, BitReversedOrder> {
    use stwo::prover::backend::simd::column::BaseColumn;
    CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(
        eval.domain,
        BaseColumn::from_iter(eval.values.to_cpu()),
    )
}

// ---------------------------------------------------------------------------
// SimdTraceCollector
// ---------------------------------------------------------------------------

/// A trace collector for SIMD traces, allowing later conversion to CUDA format.
pub struct SimdTraceCollector {
    traces: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    tree_index: usize,
}

impl SimdTraceCollector {
    pub fn new(tree_index: usize) -> Self {
        Self {
            traces: Vec::new(),
            tree_index,
        }
    }

    /// Convert collected traces to CudaBackend format and extend the CUDA tree builder.
    pub fn extend_cuda_tree_builder(self, tree_builder: &mut impl TreeBuilder<CudaBackend>) {
        tree_builder.extend_evals(
            self.traces
                .into_iter()
                .map(convert_simd_to_cuda_evaluation)
                .collect(),
        );
    }
}

impl TreeBuilder<SimdBackend> for SimdTraceCollector {
    fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ) -> TreeSubspan {
        let col_start = self.traces.len();
        self.traces.extend(columns);
        let col_end = self.traces.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }
}

// ---------------------------------------------------------------------------
// SimdToCudaBridge
// ---------------------------------------------------------------------------

/// A bridge adapter that implements `TreeBuilder<SimdBackend>` by converting
/// each batch of SIMD evaluations to CUDA format inline and forwarding them
/// to an inner CUDA tree builder.
pub struct SimdToCudaBridge<'a, TB: TreeBuilder<CudaBackend>> {
    inner: &'a mut TB,
}

impl<'a, TB: TreeBuilder<CudaBackend>> SimdToCudaBridge<'a, TB> {
    pub fn new(inner: &'a mut TB) -> Self {
        Self { inner }
    }
}

impl<TB: TreeBuilder<CudaBackend>> TreeBuilder<SimdBackend> for SimdToCudaBridge<'_, TB> {
    fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ) -> TreeSubspan {
        self.inner.extend_evals(
            columns
                .into_iter()
                .map(convert_simd_to_cuda_evaluation)
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// V0 path: CairoCudaClaimGenerator (all SIMD fallback)
// ---------------------------------------------------------------------------

/// CUDA-aware claim generator wrapping the SIMD `CairoClaimGenerator`.
///
/// All components use SIMD fallback via `SimdToCudaBridge`.
pub struct CairoCudaClaimGenerator(pub CairoClaimGenerator);

impl From<CairoClaimGenerator> for CairoCudaClaimGenerator {
    fn from(inner: CairoClaimGenerator) -> Self {
        Self(inner)
    }
}

impl CairoCudaClaimGenerator {
    pub fn write_trace_cuda<TB: TreeBuilder<CudaBackend>>(
        self,
        tree_builder: &mut TB,
    ) -> (CairoClaim, CairoCudaInteractionClaimGenerator) {
        let mut bridge = SimdToCudaBridge::new(tree_builder);
        let (claim, interaction_gen) = self.0.write_trace(&mut bridge);
        (claim, CairoCudaInteractionClaimGenerator(interaction_gen))
    }
}

/// CUDA-aware interaction claim generator wrapping the SIMD path.
pub struct CairoCudaInteractionClaimGenerator(pub CairoInteractionClaimGenerator);

impl From<CairoInteractionClaimGenerator> for CairoCudaInteractionClaimGenerator {
    fn from(inner: CairoInteractionClaimGenerator) -> Self {
        Self(inner)
    }
}

impl CairoCudaInteractionClaimGenerator {
    pub fn write_interaction_trace_cuda<TB: TreeBuilder<CudaBackend>>(
        self,
        tree_builder: &mut TB,
        common_lookup_elements: &CommonLookupElements,
    ) -> CairoInteractionClaim {
        let mut bridge = SimdToCudaBridge::new(tree_builder);
        self.0
            .write_interaction_trace(&mut bridge, common_lookup_elements)
    }
}

/// Create a V0 CUDA claim generator (SIMD fallback).
pub fn create_cairo_cuda_claim_generator(
    input: ProverInput,
    preprocessed_trace: Arc<PreProcessedTrace>,
) -> CairoCudaClaimGenerator {
    CairoCudaClaimGenerator(create_cairo_claim_generator(input, preprocessed_trace))
}

// ---------------------------------------------------------------------------
// Helper: extract public segments / public memory from Memory
// ---------------------------------------------------------------------------

fn extract_public_segments(
    memory: &Memory,
    initial_ap: u32,
    final_ap: u32,
    public_segment_context: PublicSegmentContext,
) -> PublicSegmentRanges {
    let n_public_segments = public_segment_context.iter().filter(|&b| *b).count() as u32;

    let to_memory_value = |addr: u32| {
        let id = memory.get_raw_id(addr);
        let value = memory.get(addr).as_small() as u32;
        MemorySmallValue { id, value }
    };

    let start_ptrs = (initial_ap..initial_ap + n_public_segments).map(to_memory_value);
    let end_ptrs = (final_ap - n_public_segments..final_ap).map(to_memory_value);
    let mut ranges = start_ptrs
        .zip(end_ptrs)
        .map(|(start_ptr, stop_ptr)| SegmentRange {
            start_ptr,
            stop_ptr,
        });
    let mut present = public_segment_context.into_iter();
    let mut next = || {
        let present = present.next().unwrap();
        if present {
            ranges.next()
        } else {
            None
        }
    };

    PublicSegmentRanges {
        output: next().unwrap(),
        pedersen: next(),
        range_check_128: next(),
        ecdsa: next(),
        bitwise: next(),
        ec_op: next(),
        keccak: next(),
        poseidon: next(),
        range_check_96: next(),
        add_mod: next(),
        mul_mod: next(),
    }
}

fn extract_sections_from_memory(
    memory: &Memory,
    initial_pc: u32,
    initial_ap: u32,
    final_ap: u32,
    public_segment_context: PublicSegmentContext,
) -> PublicMemory {
    use itertools::Itertools;
    let public_segments =
        extract_public_segments(memory, initial_ap, final_ap, public_segment_context);
    let program_memory_addresses = initial_pc..initial_ap - 2;
    let safe_call_addresses = initial_ap - 2..initial_ap;
    let output_memory_addresses =
        public_segments.output.start_ptr.value..public_segments.output.stop_ptr.value;
    let [program, safe_call, output] = [
        program_memory_addresses,
        safe_call_addresses,
        output_memory_addresses,
    ]
    .map(|range| {
        range
            .map(|addr| {
                let id = memory.get_raw_id(addr);
                let value = memory.get(addr).as_u256();
                (id, value)
            })
            .collect_vec()
    });

    assert!(safe_call.len() == 2);
    assert_eq!(safe_call[0].1, [initial_ap, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(safe_call[1].1, [0, 0, 0, 0, 0, 0, 0, 0]);

    PublicMemory {
        program,
        safe_call_ids: array::from_fn(|i| safe_call[i].0),
        public_segments,
        output,
    }
}

// ---------------------------------------------------------------------------
// Helper: build instruction cache from memory
// ---------------------------------------------------------------------------

fn build_instruction_cache(casm_states: &CasmStatesByOpcode, memory: &Memory) -> Vec<(u32, u128)> {
    let mut unique_pcs = HashSet::new();
    macro_rules! collect_pcs {
        ($($field:ident),* $(,)?) => {
            $(for state in &casm_states.$field {
                unique_pcs.insert(state.pc.0);
            })*
        };
    }
    collect_pcs!(
        generic_opcode,
        add_ap_opcode,
        add_opcode,
        add_opcode_small,
        assert_eq_opcode,
        assert_eq_opcode_double_deref,
        assert_eq_opcode_imm,
        call_opcode_abs,
        call_opcode_rel_imm,
        jnz_opcode_non_taken,
        jnz_opcode_taken,
        jump_opcode_rel_imm,
        jump_opcode_rel,
        jump_opcode_double_deref,
        jump_opcode_abs,
        mul_opcode_small,
        mul_opcode,
        ret_opcode,
        blake_compress_opcode,
        qm_31_add_mul_opcode,
    );
    unique_pcs
        .into_iter()
        .map(|pc| (pc, memory.get(pc).as_small()))
        .collect()
}

// ---------------------------------------------------------------------------
// Native CUDA path: NativeCairoCudaClaimGenerator
// ---------------------------------------------------------------------------

/// Determines which pedersen component variant to use.
///
/// - `Wide` (Canonical, window_bits_18): native CUDA kernels for pedersen_builtin + context.
/// - `Narrow` (CanonicalSmall, window_bits_9): native CUDA kernels for context, SIMD bridge for
///   builtin.
enum PedersenVariant {
    Wide {
        pedersen_builtin_cuda: Option<pedersen_builtin_cuda::CudaClaimGenerator>,
        pedersen_context_cuda: Option<PedersenContextCudaClaimGenerator>,
    },
    Narrow {
        pedersen_segment: Option<(u32, u32)>, // (log_size, segment_start)
        pedersen_context_wb9_cuda: Option<pedersen_wb9_cuda::PedersenContextWb9CudaClaimGenerator>,
    },
}

/// Cairo claim generator with native CUDA acceleration for all core components.
///
/// Uses CUDA for: opcodes, verify_instruction, memory_address_to_id,
/// memory_id_to_big, range_checks, blake context, builtins, pedersen context,
/// poseidon context, VBX.
pub struct NativeCairoCudaClaimGenerator {
    public_data: PublicData,

    // CUDA generators for core components
    memory_address_to_id_cuda: memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_cuda: memory_id_to_big_cuda::CudaClaimGenerator,
    verify_instruction_cuda: verify_instruction_cuda::CudaClaimGenerator,

    // CUDA generators for opcodes
    opcodes_cuda: OpcodesCudaClaimGenerator,

    // CUDA range checks
    range_checks_trace_generator: RangeChecksCudaClaimGenerator,

    // CUDA blake sub-components (conditional)
    blake_round_cuda: Option<blake_round_cuda::CudaClaimGenerator>,
    blake_g_cuda: Option<blake_g_cuda::CudaClaimGenerator>,
    blake_round_sigma_cuda: Option<blake_round_sigma_cuda::CudaClaimGenerator>,
    triple_xor_32_cuda: Option<triple_xor_32_cuda::CudaClaimGenerator>,
    // CUDA VBX generators
    vbx_4_cuda: vbx_4_cuda::CudaClaimGenerator,
    vbx_7_cuda: vbx_7_cuda::CudaClaimGenerator,
    vbx_8_cuda: Option<vbx_8_cuda::CudaClaimGenerator>,
    vbx_8_b_cuda: Option<vbx_8_b_cuda::CudaClaimGenerator>,
    vbx_9_cuda: vbx_9_cuda::CudaClaimGenerator,
    vbx_12_cuda: Option<vbx_12_cuda::CudaClaimGenerator>,

    // Builtins (CUDA native)
    add_mod_builtin_cuda: Option<add_mod_builtin_cuda::CudaClaimGenerator>,
    bitwise_builtin_cuda: Option<bitwise_builtin_cuda::CudaClaimGenerator>,
    mul_mod_builtin_cuda: Option<mul_mod_builtin_cuda::CudaClaimGenerator>,
    range_check96_builtin_cuda: Option<range_check_bits_96_builtin_cuda::CudaClaimGenerator>,
    range_check_builtin_cuda: Option<range_check_bits_128_builtin_cuda::CudaClaimGenerator>,
    #[allow(dead_code)] // Monolithic kernel incompatible with split AIR interaction trace
    poseidon_builtin_cuda: Option<poseidon_builtin_cuda::CudaClaimGenerator>,
    /// Poseidon segment info (log_size, segment_start) for encapsulated poseidon_builtin path.
    poseidon_segment: Option<(u32, u32)>,

    // Pedersen variant (Wide = CUDA kernels, Narrow = SIMD fallback)
    pedersen_variant: PedersenVariant,

    // Poseidon context (CUDA wrapper)
    poseidon_context_cuda: Option<PoseidonContextCudaClaimGenerator>,

}

impl NativeCairoCudaClaimGenerator {
    pub fn new(input: ProverInput, preprocessed_trace: Arc<PreProcessedTrace>) -> Self {
        let ProverInput {
            state_transitions,
            memory,
            public_memory_addresses,
            builtin_segments,
            public_segment_context,
            ..
        } = input;

        let initial_state = state_transitions.initial_state;
        let final_state = state_transitions.final_state;

        let has_blake = !state_transitions
            .casm_states_by_opcode
            .blake_compress_opcode
            .is_empty();
        let has_pedersen = builtin_segments.pedersen_builtin.is_some();
        let has_poseidon = builtin_segments.poseidon_builtin.is_some();

        // Build instruction cache
        let inst_cache = build_instruction_cache(&state_transitions.casm_states_by_opcode, &memory);

        // --- CUDA generators for core components ---
        let mut memory_address_to_id_cuda =
            memory_address_to_id_cuda::CudaClaimGenerator::new(&memory);
        let memory_id_to_big_cuda = memory_id_to_big_cuda::CudaClaimGenerator::new(&memory);
        let verify_instruction_cuda = verify_instruction_cuda::CudaClaimGenerator::new(inst_cache);

        // CUDA opcodes
        let opcodes_cuda = OpcodesCudaClaimGenerator::new(state_transitions);

        // Range checks (all CUDA)
        let range_checks_trace_generator = RangeChecksCudaClaimGenerator::new();

        let memory = Arc::new(memory);

        // --- Blake context sub-components (CUDA, conditional) ---
        let blake_round_cuda_gen = if has_blake {
            Some(blake_round_cuda::CudaClaimGenerator::new(memory.clone()))
        } else {
            None
        };
        let blake_g_cuda_gen = if has_blake {
            Some(blake_g_cuda::CudaClaimGenerator::new())
        } else {
            None
        };
        let blake_round_sigma_cuda_gen = if has_blake {
            Some(blake_round_sigma_cuda::CudaClaimGenerator::new())
        } else {
            None
        };
        let triple_xor_32_cuda_gen = if has_blake {
            Some(triple_xor_32_cuda::CudaClaimGenerator::new())
        } else {
            None
        };
        let has_bitwise = builtin_segments.bitwise_builtin.is_some();
        let vbx_8_cuda_gen = if has_blake || has_bitwise {
            Some(vbx_8_cuda::CudaClaimGenerator::new())
        } else {
            None
        };
        let vbx_8_b_cuda_gen = if has_blake {
            Some(vbx_8_b_cuda::CudaClaimGenerator::new())
        } else {
            None
        };
        let vbx_12_cuda_gen = if has_blake {
            Some(vbx_12_cuda::CudaClaimGenerator::new())
        } else {
            None
        };
        let vbx_4_cuda_gen = vbx_4_cuda::CudaClaimGenerator::new();
        let vbx_7_cuda_gen = vbx_7_cuda::CudaClaimGenerator::new();
        let vbx_9_cuda_gen = vbx_9_cuda::CudaClaimGenerator::new();

        // --- Builtins (CUDA native, conditional on segments) ---
        macro_rules! make_cuda_builtin {
            ($segment:expr, $module:ident, $cells:expr) => {
                $segment.map(|seg| {
                    let n = (seg.stop_ptr - seg.begin_addr) / $cells;
                    $module::CudaClaimGenerator::new(n.ilog2(), seg.begin_addr as u32)
                })
            };
        }
        let add_mod_builtin_cuda_gen = make_cuda_builtin!(
            builtin_segments.add_mod_builtin,
            add_mod_builtin_cuda,
            ADD_MOD_BUILTIN_MEMORY_CELLS
        );
        let bitwise_builtin_cuda_gen = make_cuda_builtin!(
            builtin_segments.bitwise_builtin,
            bitwise_builtin_cuda,
            BITWISE_BUILTIN_MEMORY_CELLS
        );
        let mul_mod_builtin_cuda_gen = make_cuda_builtin!(
            builtin_segments.mul_mod_builtin,
            mul_mod_builtin_cuda,
            MUL_MOD_BUILTIN_MEMORY_CELLS
        );
        let range_check96_builtin_cuda_gen = make_cuda_builtin!(
            builtin_segments.range_check96_builtin,
            range_check_bits_96_builtin_cuda,
            RANGE_CHECK_96_BUILTIN_MEMORY_CELLS
        );
        let range_check_builtin_cuda_gen = make_cuda_builtin!(
            builtin_segments.range_check_builtin,
            range_check_bits_128_builtin_cuda,
            RANGE_CHECK_BUILTIN_MEMORY_CELLS
        );
        // poseidon_builtin uses SIMD hybrid path (the monolithic CUDA kernel's interaction
        // trace is incompatible with the split v1.1.0 AIR). Store segment info for SIMD path.
        let poseidon_segment = builtin_segments.poseidon_builtin.map(|seg| {
            let n = (seg.stop_ptr - seg.begin_addr) / POSEIDON_BUILTIN_MEMORY_CELLS;
            (n.ilog2(), seg.begin_addr as u32)
        });

        // --- Pedersen variant detection ---
        // CanonicalSmall uses pedersen_points_small_0 (window_bits_9, 512-row tables).
        // Canonical uses pedersen_points_0 (window_bits_18, 262K-row tables).
        let is_narrow = preprocessed_trace.has_column(&PreProcessedColumnId {
            id: "pedersen_points_small_0".to_owned(),
        });
        let pedersen_variant = if is_narrow {
            let pedersen_segment = builtin_segments.pedersen_builtin.map(|seg| {
                let n = (seg.stop_ptr - seg.begin_addr) / PEDERSEN_BUILTIN_MEMORY_CELLS;
                (n.ilog2(), seg.begin_addr as u32)
            });
            let pedersen_context_wb9_cuda = if has_pedersen {
                Some(
                    pedersen_wb9_cuda::PedersenContextWb9CudaClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ),
                )
            } else {
                None
            };
            PedersenVariant::Narrow {
                pedersen_segment,
                pedersen_context_wb9_cuda,
            }
        } else {
            let pedersen_builtin_cuda_gen = builtin_segments.pedersen_builtin.map(|seg| {
                let n = (seg.stop_ptr - seg.begin_addr) / PEDERSEN_BUILTIN_MEMORY_CELLS;
                pedersen_builtin_cuda::CudaClaimGenerator::new(n.ilog2(), seg.begin_addr as u32)
            });
            let pedersen_context_cuda = if has_pedersen {
                Some(PedersenContextCudaClaimGenerator::new(
                    preprocessed_trace.clone(),
                ))
            } else {
                None
            };
            PedersenVariant::Wide {
                pedersen_builtin_cuda: pedersen_builtin_cuda_gen,
                pedersen_context_cuda,
            }
        };

        // --- Poseidon context CUDA wrapper (conditional) ---
        let poseidon_context_cuda = if has_poseidon {
            Some(PoseidonContextCudaClaimGenerator::new(
                preprocessed_trace.clone(),
            ))
        } else {
            None
        };

        // --- Yield public memory into CUDA generators ---
        for addr in public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda.get_id(addr);
            memory_address_to_id_cuda.add_cuda_input(&addr);
            memory_id_to_big_cuda.add_cuda_input(&id);
        }

        // --- Public data ---
        let initial_pc = initial_state.pc.0;
        let initial_ap = initial_state.ap.0;
        let final_ap = final_state.ap.0;
        let public_memory = extract_sections_from_memory(
            &memory,
            initial_pc,
            initial_ap,
            final_ap,
            public_segment_context,
        );

        let public_data = PublicData {
            public_memory,
            initial_state,
            final_state,
        };

        Self {
            public_data,
            memory_address_to_id_cuda,
            memory_id_to_big_cuda,
            verify_instruction_cuda,
            opcodes_cuda,
            range_checks_trace_generator,
            blake_round_cuda: blake_round_cuda_gen,
            blake_g_cuda: blake_g_cuda_gen,
            blake_round_sigma_cuda: blake_round_sigma_cuda_gen,
            triple_xor_32_cuda: triple_xor_32_cuda_gen,
            vbx_4_cuda: vbx_4_cuda_gen,
            vbx_7_cuda: vbx_7_cuda_gen,
            vbx_8_cuda: vbx_8_cuda_gen,
            vbx_8_b_cuda: vbx_8_b_cuda_gen,
            vbx_9_cuda: vbx_9_cuda_gen,
            vbx_12_cuda: vbx_12_cuda_gen,
            add_mod_builtin_cuda: add_mod_builtin_cuda_gen,
            bitwise_builtin_cuda: bitwise_builtin_cuda_gen,
            mul_mod_builtin_cuda: mul_mod_builtin_cuda_gen,
            range_check96_builtin_cuda: range_check96_builtin_cuda_gen,
            range_check_builtin_cuda: range_check_builtin_cuda_gen,
            poseidon_builtin_cuda: None, /* SIMD hybrid: monolithic kernel incompatible with
                                          * split AIR */
            poseidon_segment,
            pedersen_variant,
            poseidon_context_cuda,
        }
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (CairoClaim, NativeCairoCudaInteractionClaimGenerator) {
        // ==== 1. Generate opcode traces (CUDA) ====
        let span = span!(Level::INFO, "base_trace: opcodes").entered();
        let (opcodes_claim, opcodes_interaction_gen) = self.opcodes_cuda.write_trace(
            tree_builder,
            &mut self.memory_address_to_id_cuda,
            &self.memory_id_to_big_cuda,
            &self.verify_instruction_cuda,
            &self.range_checks_trace_generator,
            &mut self.blake_round_cuda,
            &mut self.triple_xor_32_cuda,
            &self.vbx_8_cuda,
        );
        span.exit();

        // ==== 2. Generate verify_instruction trace (CUDA) ====
        let span = span!(Level::INFO, "base_trace: verify_instruction").entered();
        let (verify_instruction_claim, verify_instruction_interaction_gen) =
            self.verify_instruction_cuda.write_trace(
                tree_builder,
                &mut self.memory_address_to_id_cuda,
                &self.memory_id_to_big_cuda,
                &self.range_checks_trace_generator.rc_4_3_trace_generator,
                &self.range_checks_trace_generator.rc_7_2_5_trace_generator,
            );
        span.exit();

        // ==== 3. Generate blake_context traces (CUDA) ====
        let span = span!(Level::INFO, "base_trace: blake_context").entered();
        let (
            blake_round_claim,
            blake_g_claim,
            blake_round_sigma_claim,
            triple_xor_32_claim,
            vbx_12_claim,
            blake_context_interaction_gens,
        ) = if let (
            Some(blake_round_cuda),
            Some(mut blake_g_cuda),
            Some(blake_round_sigma_cuda),
            Some(triple_xor_32_cuda),
        ) = (
            self.blake_round_cuda,
            self.blake_g_cuda.take(),
            self.blake_round_sigma_cuda,
            self.triple_xor_32_cuda,
        ) {
            let (blake_round_claim, blake_round_interaction_gen) = blake_round_cuda.write_trace(
                tree_builder,
                &mut blake_g_cuda,
                &blake_round_sigma_cuda,
                &mut self.memory_address_to_id_cuda,
                &self.memory_id_to_big_cuda,
                &self.range_checks_trace_generator.rc_7_2_5_trace_generator,
            );

            let vbx_12_cuda_ref = self.vbx_12_cuda.as_ref().unwrap();
            let vbx_8_b_cuda_ref = self.vbx_8_b_cuda.as_ref().unwrap();
            let (blake_g_claim, blake_g_interaction_gen) = blake_g_cuda.write_trace(
                tree_builder,
                vbx_12_cuda_ref,
                &self.vbx_4_cuda,
                &self.vbx_7_cuda,
                self.vbx_8_cuda.as_ref().unwrap(),
                vbx_8_b_cuda_ref,
                &self.vbx_9_cuda,
            );

            let (blake_sigma_claim, blake_sigma_interaction_gen) =
                blake_round_sigma_cuda.write_trace(tree_builder);

            let (triple_xor_32_claim, triple_xor_32_interaction_gen) = triple_xor_32_cuda
                .write_trace(
                    tree_builder,
                    self.vbx_8_cuda.as_ref().unwrap(),
                    vbx_8_b_cuda_ref,
                );

            let vbx_12_cuda = self.vbx_12_cuda.take().unwrap();
            let (vbx_12_claim, vbx_12_interaction_gen) = vbx_12_cuda.write_trace_cuda(tree_builder);

            (
                Some(blake_round_claim),
                Some(blake_g_claim),
                Some(blake_sigma_claim),
                Some(triple_xor_32_claim),
                Some(vbx_12_claim),
                Some(BlakeContextCudaInteractionGens {
                    blake_round: blake_round_interaction_gen,
                    blake_g: blake_g_interaction_gen,
                    blake_sigma: blake_sigma_interaction_gen,
                    triple_xor_32: triple_xor_32_interaction_gen,
                    vbx_12: vbx_12_interaction_gen,
                }),
            )
        } else {
            (None, None, None, None, None, None)
        };
        span.exit();

        // ==== 4. Generate builtins traces ====
        let span = span!(Level::INFO, "base_trace: builtins").entered();

        let (add_mod_builtin_claim, add_mod_builtin_interaction_gen) = self
            .add_mod_builtin_cuda
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                    &self.memory_id_to_big_cuda,
                )
            })
            .unzip();

        let (bitwise_builtin_claim, bitwise_builtin_interaction_gen) = self
            .bitwise_builtin_cuda
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                    &self.memory_id_to_big_cuda,
                    self.vbx_8_cuda.as_ref().unwrap(),
                    &self.vbx_9_cuda,
                )
            })
            .unzip();

        let (mul_mod_builtin_claim, mul_mod_builtin_interaction_gen) = self
            .mul_mod_builtin_cuda
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                    &self.memory_id_to_big_cuda,
                    &self.range_checks_trace_generator.rc_12_trace_generator,
                    &self.range_checks_trace_generator.rc_18_trace_generator,
                    &self.range_checks_trace_generator.rc_3_6_6_3_trace_generator,
                )
            })
            .unzip();

        // Extract pedersen variant for step 4 + 5 dispatch
        let (
            pedersen_builtin_cuda_opt,
            mut pedersen_context_cuda_opt,
            pedersen_narrow_segment,
            mut pedersen_context_wb9_cuda_opt,
        ) = match self.pedersen_variant {
            PedersenVariant::Wide {
                pedersen_builtin_cuda,
                pedersen_context_cuda,
            } => (pedersen_builtin_cuda, pedersen_context_cuda, None, None),
            PedersenVariant::Narrow {
                pedersen_segment,
                pedersen_context_wb9_cuda,
            } => (None, None, pedersen_segment, pedersen_context_wb9_cuda),
        };

        // Wide: CUDA pedersen_builtin (window_bits_18)
        // Returns 3 GPU ID arrays for direct transfer to pedersen_aggregator.
        let (pedersen_builtin_claim, pedersen_builtin_interaction_gen, pedersen_wide_gpu_ids) =
            if let Some(gen) = pedersen_builtin_cuda_opt {
                let (claim, interaction_gen, sub_agg) = gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                );
                let log_size = claim.log_size;
                (Some(claim), Some(interaction_gen), Some((sub_agg, log_size)))
            } else {
                (None, None, None)
            };
        // Overwrite the pedersen_context_cuda aggregator with the direct GPU path
        if let Some((gpu_ids, log_size)) = pedersen_wide_gpu_ids {
            if let Some(ref mut ctx) = pedersen_context_cuda_opt {
                ctx.pedersen_aggregator_cuda =
                    pedersen_aggregator_cuda::CudaClaimGenerator::from_gpu_ids(gpu_ids, log_size);
            }
        }

        // Narrow: Native CUDA pedersen_builtin_narrow_windows (window_bits_9)
        // Returns 3 GPU ID arrays for direct transfer to pedersen_aggregator_wb9.
        let (pedersen_builtin_narrow_claim, pedersen_narrow_builtin_interaction_gen) =
            if let Some((log_size, segment_start)) = pedersen_narrow_segment {
                let narrow_cuda_gen =
                    pedersen_builtin_narrow_cuda::CudaClaimGenerator::new(log_size, segment_start);
                let (claim, interaction_gen, sub_agg) = narrow_cuda_gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                );
                // Overwrite the wb9 aggregator with the direct GPU path
                if let Some(ref mut ctx) = pedersen_context_wb9_cuda_opt {
                    ctx.pedersen_aggregator_wb9_cuda =
                        pedersen_aggregator_wb9_cuda::CudaClaimGenerator::from_gpu_ids(
                            sub_agg,
                            claim.log_size,
                        );
                }
                (Some(claim), Some(interaction_gen))
            } else {
                (None, None)
            };

        // poseidon_builtin: Native CUDA kernel (split, 6 columns).
        // Generates 6 trace columns + 6 GPU ID arrays for poseidon_aggregator.
        // The 6 ID arrays stay on GPU — no CPU roundtrip through DashMap.
        let (
            poseidon_builtin_claim,
            poseidon_builtin_cuda_interaction_gen,
            poseidon_aggregator_cuda_gen,
        ) = if let Some((log_size, segment_start)) = self.poseidon_segment {
            let split_cuda_gen =
                poseidon_builtin_split_cuda::CudaClaimGenerator::new(log_size, segment_start);
            let (claim, interaction_gen, agg_gpu_ids) = split_cuda_gen.write_trace(
                tree_builder,
                &mut self.memory_address_to_id_cuda,
            );

            // Pass GPU ID arrays directly to aggregator — zero CPU roundtrip
            let aggregator_cuda =
                poseidon_aggregator_native_cuda::CudaClaimGenerator::from_gpu_ids(
                    agg_gpu_ids,
                    log_size,
                );
            (Some(claim), Some(interaction_gen), Some(aggregator_cuda))
        } else {
            (None, None, None)
        };

        let (range_check_96_builtin_claim, range_check_96_builtin_interaction_gen) = self
            .range_check96_builtin_cuda
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                    &self.memory_id_to_big_cuda,
                    &self.range_checks_trace_generator.rc_6_trace_generator,
                )
            })
            .unzip();

        let (range_check_128_builtin_claim, range_check_128_builtin_interaction_gen) = self
            .range_check_builtin_cuda
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                    &self.memory_id_to_big_cuda,
                )
            })
            .unzip();

        span.exit();

        // ==== 5. Generate pedersen_context traces ====
        let span = span!(Level::INFO, "base_trace: pedersen_context").entered();

        // Wide path: CUDA pedersen context (window_bits_18)
        let (
            pedersen_aggregator_claim,
            partial_ec_mul_claim,
            pedersen_points_table_claim,
            pedersen_interaction_mode,
        ) = {
            match pedersen_context_cuda_opt {
                Some(ctx) => {
                    let (agg_result, pem_cuda, ppt_cuda) = ctx.write_trace_aggregator(
                        tree_builder,
                        &self.memory_id_to_big_cuda,
                        &self.range_checks_trace_generator.rc_8_trace_generator,
                    );

                    match agg_result {
                        Some(agg) => {
                            let cuda_result = pedersen_cuda::write_cuda_components(
                                pem_cuda,
                                ppt_cuda,
                                tree_builder,
                                &self.range_checks_trace_generator.rc_9_9_trace_generator,
                                &self.range_checks_trace_generator.rc_20_trace_generator,
                            );

                            let gen = PedersenContextCudaInteractionClaimGenerator::new(
                                agg.pedersen_aggregator_interaction_gen,
                                cuda_result.partial_ec_mul_interaction_gen,
                                cuda_result.ppt_interaction_gen,
                            );
                            (
                                Some(agg.pedersen_aggregator_claim),
                                Some(cuda_result.partial_ec_mul_claim),
                                Some(cuda_result.ppt_claim),
                                Some(PedersenInteractionMode::Cuda(gen)),
                            )
                        }
                        None => (None, None, None, Some(PedersenInteractionMode::Empty)),
                    }
                }
                None => (None, None, None, None),
            }
        };

        // Narrow path: Native CUDA pedersen context (window_bits_9)
        let (
            pedersen_aggregator_w9_claim,
            partial_ec_mul_w9_claim,
            ppt_w9_claim,
            pedersen_narrow_context_interaction_gen,
        ) = {
            match pedersen_context_wb9_cuda_opt {
                Some(ctx) => {
                    let (agg_result, pem_cuda, ppt_cuda) = ctx.write_trace_aggregator(
                        tree_builder,
                        &self.memory_id_to_big_cuda,
                        &self.range_checks_trace_generator.rc_8_trace_generator,
                    );

                    match agg_result {
                        Some(agg) => {
                            let cuda_result = pedersen_wb9_cuda::write_cuda_components(
                                pem_cuda,
                                ppt_cuda,
                                tree_builder,
                                &self.range_checks_trace_generator.rc_9_9_trace_generator,
                                &self.range_checks_trace_generator.rc_20_trace_generator,
                            );

                            let gen =
                                pedersen_wb9_cuda::PedersenContextWb9CudaInteractionClaimGenerator::new(
                                    agg.pedersen_aggregator_interaction_gen,
                                    cuda_result.partial_ec_mul_interaction_gen,
                                    cuda_result.ppt_interaction_gen,
                                );
                            (
                                Some(agg.pedersen_aggregator_claim),
                                Some(cuda_result.partial_ec_mul_claim),
                                Some(cuda_result.ppt_claim),
                                Some(pedersen_wb9_cuda::PedersenWb9InteractionMode::Cuda(gen)),
                            )
                        }
                        None => (
                            None,
                            None,
                            None,
                            Some(pedersen_wb9_cuda::PedersenWb9InteractionMode::Empty),
                        ),
                    }
                }
                None => (None, None, None, None),
            }
        };

        span.exit();

        // ==== 5.5 Poseidon aggregator (native CUDA) ====
        // GPU ID arrays from poseidon_builtin_split passed directly — no CPU roundtrip.
        // Runs the full 342-column base trace computation on GPU.
        let span = span!(
            Level::INFO,
            "base_trace: poseidon aggregator (native CUDA)"
        )
        .entered();
        let (poseidon_aggregator_claim, poseidon_aggregator_cuda_interaction_gen) =
            match poseidon_aggregator_cuda_gen {
                Some(agg_cuda) => {
                    let (claim, interaction_gen) = agg_cuda.write_trace(
                        tree_builder,
                        &mut self.poseidon_context_cuda,
                        &mut self.memory_id_to_big_cuda,
                        &mut self.range_checks_trace_generator,
                    );
                    (Some(claim), Some(interaction_gen))
                }
                None => (None, None),
            };
        span.exit();

        // ==== 6. Generate poseidon chain traces (CUDA) ====
        // PoseidonContextCudaClaimGenerator runs the CUDA chain generators
        // (full_round_chain, 3_partial_rounds_chain, cube_252, round_keys, rc_252w27).
        // Their inputs were populated by step 5.5 (GPU-direct from poseidon_aggregator)
        // or are empty (non-poseidon workloads).
        let span = span!(Level::INFO, "base_trace: poseidon_context (CUDA chains)").entered();
        let (poseidon_context_claim, poseidon_context_interaction_gen) =
            match self.poseidon_context_cuda {
                Some(ctx) => {
                    let (claim, gen) =
                        ctx.write_trace(tree_builder, &self.range_checks_trace_generator);
                    (claim.claim, Some(gen))
                }
                None => (None, None),
            };
        span.exit();

        // ==== 7. Generate memory_address_to_id (CUDA) ====
        let span = span!(Level::INFO, "base_trace: memory_address_to_id").entered();
        let (memory_address_to_id_claim, memory_address_to_id_interaction_gen) =
            self.memory_address_to_id_cuda.write_trace(tree_builder);
        span.exit();

        // ==== 8. Generate memory_id_to_big (CUDA) ====
        let span = span!(Level::INFO, "base_trace: memory_id_to_big").entered();
        const LOG_MAX_BIG_SIZE: u32 = MAX_SEQUENCE_LOG_SIZE;
        let (memory_id_to_value_claim, memory_id_to_value_interaction_gen) =
            self.memory_id_to_big_cuda.write_trace_cuda(
                tree_builder,
                &self.range_checks_trace_generator.rc_9_9_trace_generator,
                LOG_MAX_BIG_SIZE,
            );
        span.exit();

        // ==== 9. Generate range_checks (CUDA) ====
        let span = span!(Level::INFO, "base_trace: range_checks").entered();
        let (range_checks_claim, range_checks_interaction_gen) =
            self.range_checks_trace_generator.write_trace(tree_builder);
        span.exit();

        // ==== 10. Generate verify_bitwise_xor 4, 7, 8, 9 (all CUDA) ====
        let span = span!(Level::INFO, "base_trace: verify_bitwise_xor").entered();
        let (verify_bitwise_xor_4_claim, verify_bitwise_xor_4_interaction_gen) =
            self.vbx_4_cuda.write_trace_cuda(tree_builder);
        let (verify_bitwise_xor_7_claim, verify_bitwise_xor_7_interaction_gen) =
            self.vbx_7_cuda.write_trace_cuda(tree_builder);
        let (verify_bitwise_xor_8_claim, verify_bitwise_xor_8_interaction_gen) = {
            let vbx_8 = self
                .vbx_8_cuda
                .take()
                .unwrap_or_else(vbx_8_cuda::CudaClaimGenerator::default);
            let vbx_8_b = self
                .vbx_8_b_cuda
                .take()
                .unwrap_or_else(vbx_8_b_cuda::CudaClaimGenerator::default);
            vbx_8.write_trace_cuda(&vbx_8_b, tree_builder)
        };
        let (verify_bitwise_xor_9_claim, verify_bitwise_xor_9_interaction_gen) =
            self.vbx_9_cuda.write_trace_cuda(tree_builder);
        span.exit();

        // Assemble flat CairoClaim matching v1.1.0 struct field order.
        // Opcodes: unpack from grouped OpcodeClaim → individual Option fields.
        let claim = CairoClaim {
            public_data: self.public_data,
            // Opcodes (each Vec has exactly 0 or 1 elements in practice)
            add_opcode: opcodes_claim.add.into_iter().next(),
            add_opcode_small: opcodes_claim.add_small.into_iter().next(),
            add_ap_opcode: opcodes_claim.add_ap.into_iter().next(),
            assert_eq_opcode: opcodes_claim.assert_eq.into_iter().next(),
            assert_eq_opcode_imm: opcodes_claim.assert_eq_imm.into_iter().next(),
            assert_eq_opcode_double_deref: opcodes_claim.assert_eq_double_deref.into_iter().next(),
            blake_compress_opcode: opcodes_claim.blake.into_iter().next(),
            call_opcode_abs: opcodes_claim.call.into_iter().next(),
            call_opcode_rel_imm: opcodes_claim.call_rel_imm.into_iter().next(),
            generic_opcode: opcodes_claim.generic.into_iter().next(),
            jnz_opcode_non_taken: opcodes_claim.jnz.into_iter().next(),
            jnz_opcode_taken: opcodes_claim.jnz_taken.into_iter().next(),
            jump_opcode_abs: opcodes_claim.jump.into_iter().next(),
            jump_opcode_double_deref: opcodes_claim.jump_double_deref.into_iter().next(),
            jump_opcode_rel: opcodes_claim.jump_rel.into_iter().next(),
            jump_opcode_rel_imm: opcodes_claim.jump_rel_imm.into_iter().next(),
            mul_opcode: opcodes_claim.mul.into_iter().next(),
            mul_opcode_small: opcodes_claim.mul_small.into_iter().next(),
            qm_31_add_mul_opcode: opcodes_claim.qm31.into_iter().next(),
            ret_opcode: opcodes_claim.ret.into_iter().next(),
            // verify_instruction
            verify_instruction: Some(verify_instruction_claim),
            // Blake context (flat)
            blake_round: blake_round_claim,
            blake_g: blake_g_claim,
            blake_round_sigma: blake_round_sigma_claim,
            triple_xor_32: triple_xor_32_claim,
            verify_bitwise_xor_12: vbx_12_claim,
            // Builtins
            add_mod_builtin: add_mod_builtin_claim,
            bitwise_builtin: bitwise_builtin_claim,
            mul_mod_builtin: mul_mod_builtin_claim,
            pedersen_builtin: pedersen_builtin_claim, // Wide only
            pedersen_builtin_narrow_windows: pedersen_builtin_narrow_claim, // Narrow only
            poseidon_builtin: poseidon_builtin_claim,
            range_check96_builtin: range_check_96_builtin_claim,
            range_check_builtin: range_check_128_builtin_claim,
            // Pedersen context (flat) — mutually exclusive: Wide (w18) or Narrow (w9)
            pedersen_aggregator_window_bits_18: pedersen_aggregator_claim,
            partial_ec_mul_window_bits_18: partial_ec_mul_claim,
            pedersen_points_table_window_bits_18: pedersen_points_table_claim,
            pedersen_aggregator_window_bits_9: pedersen_aggregator_w9_claim,
            partial_ec_mul_window_bits_9: partial_ec_mul_w9_claim,
            pedersen_points_table_window_bits_9: ppt_w9_claim,
            // Poseidon context (flat) — chain claims from CUDA poseidon_context
            poseidon_aggregator: poseidon_aggregator_claim,
            poseidon_3_partial_rounds_chain: poseidon_context_claim
                .as_ref()
                .map(|c| c.poseidon_3_partial_rounds_chain),
            poseidon_full_round_chain: poseidon_context_claim
                .as_ref()
                .map(|c| c.poseidon_full_round_chain),
            cube_252: poseidon_context_claim.as_ref().map(|c| c.cube_252),
            poseidon_round_keys: poseidon_context_claim
                .as_ref()
                .map(|c| c.poseidon_round_keys),
            range_check_252_width_27: poseidon_context_claim.map(|c| c.range_check_252_width_27),
            // Memory
            memory_address_to_id: Some(memory_address_to_id_claim),
            memory_id_to_big: Some(memory_id_to_value_claim),
            // Range checks (flat)
            range_check_6: Some(range_checks_claim.rc_6),
            range_check_8: Some(range_checks_claim.rc_8),
            range_check_11: Some(range_checks_claim.rc_11),
            range_check_12: Some(range_checks_claim.rc_12),
            range_check_18: Some(range_checks_claim.rc_18),
            range_check_20: Some(range_checks_claim.rc_20),
            range_check_4_3: Some(range_checks_claim.rc_4_3),
            range_check_4_4: Some(range_checks_claim.rc_4_4),
            range_check_9_9: Some(range_checks_claim.rc_9_9),
            range_check_7_2_5: Some(range_checks_claim.rc_7_2_5),
            range_check_3_6_6_3: Some(range_checks_claim.rc_3_6_6_3),
            range_check_4_4_4_4: Some(range_checks_claim.rc_4_4_4_4),
            range_check_3_3_3_3_3: Some(range_checks_claim.rc_3_3_3_3_3),
            // VBX
            verify_bitwise_xor_4: Some(verify_bitwise_xor_4_claim),
            verify_bitwise_xor_7: Some(verify_bitwise_xor_7_claim),
            verify_bitwise_xor_8: Some(verify_bitwise_xor_8_claim),
            verify_bitwise_xor_9: Some(verify_bitwise_xor_9_claim),
        };

        let interaction_gen = NativeCairoCudaInteractionClaimGenerator {
            opcodes_interaction_gen,
            verify_instruction_interaction_gen,
            blake_context_interaction_gens,
            builtins_interaction_gen: BuiltinsCudaInteractionClaimGenerator {
                add_mod: add_mod_builtin_interaction_gen,
                bitwise: bitwise_builtin_interaction_gen,
                mul_mod: mul_mod_builtin_interaction_gen,
                pedersen: pedersen_builtin_interaction_gen, // Wide only (None for Narrow)
                poseidon: None,                             /* Encapsulated CUDA wrapper —
                                                             * handled via
                                                             * poseidon_builtin_cuda_interaction_gen */
                range_check_96: range_check_96_builtin_interaction_gen,
                range_check_128: range_check_128_builtin_interaction_gen,
            },
            poseidon_builtin_cuda_interaction_gen,
            poseidon_aggregator_cuda_interaction_gen,
            pedersen_context_interaction_gen: pedersen_interaction_mode, // Wide only
            pedersen_narrow_builtin_interaction_gen,                     // Narrow only
            pedersen_narrow_context_interaction_gen,                     // Narrow only
            poseidon_context_interaction_gen,
            memory_address_to_id_interaction_gen,
            memory_id_to_value_interaction_gen,
            range_checks_interaction_gen,
            verify_bitwise_xor_4_interaction_gen,
            verify_bitwise_xor_7_interaction_gen,
            verify_bitwise_xor_8_interaction_gen,
            verify_bitwise_xor_9_interaction_gen,
        };

        (claim, interaction_gen)
    }
}

// ---------------------------------------------------------------------------
// BuiltinsCudaInteractionClaimGenerator
// ---------------------------------------------------------------------------

struct BuiltinsCudaInteractionClaimGenerator {
    add_mod: Option<add_mod_builtin_cuda::CudaInteractionClaimGenerator>,
    bitwise: Option<bitwise_builtin_cuda::CudaInteractionClaimGenerator>,
    mul_mod: Option<mul_mod_builtin_cuda::CudaInteractionClaimGenerator>,
    pedersen: Option<pedersen_builtin_cuda::CudaInteractionClaimGenerator>,
    #[allow(dead_code)] // Always None; poseidon uses SIMD hybrid path
    poseidon: Option<poseidon_builtin_cuda::CudaInteractionClaimGenerator>,
    range_check_96: Option<range_check_bits_96_builtin_cuda::CudaInteractionClaimGenerator>,
    range_check_128: Option<range_check_bits_128_builtin_cuda::CudaInteractionClaimGenerator>,
}

// ---------------------------------------------------------------------------
// BlakeContextCudaInteractionGens
// ---------------------------------------------------------------------------

struct BlakeContextCudaInteractionGens {
    blake_round: blake_round_cuda::CudaInteractionClaimGenerator,
    blake_g: blake_g_cuda::CudaInteractionClaimGenerator,
    blake_sigma: blake_round_sigma_cuda::CudaInteractionClaimGenerator,
    triple_xor_32: triple_xor_32_cuda::CudaInteractionClaimGenerator,
    vbx_12: vbx_12_cuda::CudaInteractionClaimGenerator,
}

// ---------------------------------------------------------------------------
// NativeCairoCudaInteractionClaimGenerator
// ---------------------------------------------------------------------------

/// Interaction claim generator for the native CUDA path.
pub struct NativeCairoCudaInteractionClaimGenerator {
    opcodes_interaction_gen: OpcodesCudaInteractionClaimGenerator,
    verify_instruction_interaction_gen: verify_instruction_cuda::CudaInteractionClaimGenerator,
    blake_context_interaction_gens: Option<BlakeContextCudaInteractionGens>,
    builtins_interaction_gen: BuiltinsCudaInteractionClaimGenerator,
    // Native CUDA poseidon_builtin (split) interaction generator
    poseidon_builtin_cuda_interaction_gen:
        Option<poseidon_builtin_split_cuda::CudaInteractionClaimGenerator>,
    // Native CUDA poseidon_aggregator interaction generator
    poseidon_aggregator_cuda_interaction_gen:
        Option<poseidon_aggregator_native_cuda::CudaInteractionClaimGenerator>,
    pedersen_context_interaction_gen: Option<PedersenInteractionMode>, // Wide only
    // Narrow pedersen interaction generators (CanonicalSmall) — native CUDA
    pedersen_narrow_builtin_interaction_gen:
        Option<pedersen_builtin_narrow_cuda::CudaInteractionClaimGenerator>,
    pedersen_narrow_context_interaction_gen: Option<pedersen_wb9_cuda::PedersenWb9InteractionMode>,
    // CUDA poseidon chain interaction generator (full_round, 3_partial, cube_252, round_keys,
    // rc_252w27)
    poseidon_context_interaction_gen: Option<PoseidonContextCudaInteractionClaimGenerator>,
    memory_address_to_id_interaction_gen: memory_address_to_id_cuda::CudaInteractionClaimGenerator,
    memory_id_to_value_interaction_gen: memory_id_to_big_cuda::CudaInteractionClaimGeneratorCuda,
    range_checks_interaction_gen: RangeChecksCudaInteractionClaimGenerator,
    verify_bitwise_xor_4_interaction_gen: vbx_4_cuda::InteractionClaimGeneratorCuda,
    verify_bitwise_xor_7_interaction_gen: vbx_7_cuda::InteractionClaimGeneratorCuda,
    verify_bitwise_xor_8_interaction_gen: vbx_8_cuda::CudaInteractionClaimGenerator,
    verify_bitwise_xor_9_interaction_gen: vbx_9_cuda::InteractionClaimGeneratorCuda,
}

impl NativeCairoCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        common_lookup_elements: &CommonLookupElements,
    ) -> CairoInteractionClaim {
        // ==== 1. Opcodes interaction traces (CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: opcodes").entered();
        let opcodes_interaction_claim = self
            .opcodes_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        span.exit();

        // ==== 2. verify_instruction interaction trace (CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: verify_instruction").entered();
        let verify_instruction_interaction_claim = self
            .verify_instruction_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        span.exit();

        // ==== 3. blake_context interaction trace (CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: blake_context").entered();
        let (
            blake_round_interaction,
            blake_g_interaction,
            blake_sigma_interaction,
            triple_xor_32_interaction,
            vbx_12_interaction,
        ) = if let Some(gens) = self.blake_context_interaction_gens {
            let blake_round = gens
                .blake_round
                .write_interaction_trace(tree_builder, common_lookup_elements);
            let blake_g = gens
                .blake_g
                .write_interaction_trace(tree_builder, common_lookup_elements);
            let blake_sigma = gens
                .blake_sigma
                .write_interaction_trace(tree_builder, common_lookup_elements);
            let triple_xor_32 = gens
                .triple_xor_32
                .write_interaction_trace(tree_builder, common_lookup_elements);
            let vbx_12 = gens
                .vbx_12
                .write_interaction_trace(tree_builder, common_lookup_elements);
            (
                Some(blake_round),
                Some(blake_g),
                Some(blake_sigma),
                Some(triple_xor_32),
                Some(vbx_12),
            )
        } else {
            (None, None, None, None, None)
        };
        span.exit();

        // ==== 4. builtins interaction trace ====
        let span = span!(Level::INFO, "interaction_trace: builtins").entered();
        let add_mod_bi = self
            .builtins_interaction_gen
            .add_mod
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        let bitwise_bi = self
            .builtins_interaction_gen
            .bitwise
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        let mul_mod_bi = self
            .builtins_interaction_gen
            .mul_mod
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        // Wide: CUDA pedersen_builtin interaction trace
        let pedersen_bi = self
            .builtins_interaction_gen
            .pedersen
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        // Narrow: Native CUDA pedersen_builtin_narrow_windows interaction trace
        let pedersen_narrow_bi = self
            .pedersen_narrow_builtin_interaction_gen
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        // poseidon_builtin: Native CUDA (split) interaction trace
        let poseidon_bi = self
            .poseidon_builtin_cuda_interaction_gen
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        let range_check_96_bi = self
            .builtins_interaction_gen
            .range_check_96
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        let range_check_128_bi = self
            .builtins_interaction_gen
            .range_check_128
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        span.exit();

        // ==== 5. pedersen_context interaction trace ====
        let span = span!(Level::INFO, "interaction_trace: pedersen_context").entered();
        // Wide path: CUDA pedersen context (window_bits_18)
        let pedersen_context_interaction = match self.pedersen_context_interaction_gen {
            Some(gen) if !gen.is_empty() => {
                let result = gen.write_interaction_trace(tree_builder, common_lookup_elements);
                result.claim
            }
            _ => None,
        };
        // Narrow path: Native CUDA pedersen context (window_bits_9)
        let pedersen_narrow_context_interaction = match self.pedersen_narrow_context_interaction_gen
        {
            Some(gen) if !gen.is_empty() => {
                gen.write_interaction_trace(tree_builder, common_lookup_elements)
                    .claim
            }
            _ => None,
        };
        span.exit();

        // ==== 5.5. poseidon_aggregator interaction trace (native CUDA) ====
        let span = span!(
            Level::INFO,
            "interaction_trace: poseidon_aggregator (native CUDA)"
        )
        .entered();
        let poseidon_aggregator_interaction = self
            .poseidon_aggregator_cuda_interaction_gen
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        span.exit();

        // ==== 6. poseidon_context interaction trace (CUDA chains) ====
        let span = span!(Level::INFO, "interaction_trace: poseidon_context").entered();
        let poseidon_context_interaction = match self.poseidon_context_interaction_gen {
            Some(gen) => {
                let result = gen.write_interaction_trace(tree_builder, common_lookup_elements);
                result.claim
            }
            None => None,
        };
        span.exit();

        // ==== 7. memory_address_to_id interaction trace (CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: memory_address_to_id").entered();
        let memory_address_to_id_interaction_claim = self
            .memory_address_to_id_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        span.exit();

        // ==== 8. memory_id_to_big interaction trace (CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: memory_id_to_big").entered();
        let memory_id_to_value_interaction_claim = self
            .memory_id_to_value_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        span.exit();

        // ==== 9. range_checks interaction trace (CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: range_checks").entered();
        let range_checks_interaction_claim = self
            .range_checks_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        span.exit();

        // ==== 10. verify_bitwise_xor interaction traces (all CUDA) ====
        let span = span!(Level::INFO, "interaction_trace: verify_bitwise_xor").entered();
        let verify_bitwise_xor_4_interaction_claim = self
            .verify_bitwise_xor_4_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let verify_bitwise_xor_7_interaction_claim = self
            .verify_bitwise_xor_7_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let verify_bitwise_xor_8_interaction_claim = self
            .verify_bitwise_xor_8_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let verify_bitwise_xor_9_interaction_claim = self
            .verify_bitwise_xor_9_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        span.exit();

        // Assemble flat CairoInteractionClaim matching v1.1.0 struct field order.
        CairoInteractionClaim {
            // Opcodes (unpack from grouped)
            add_opcode: opcodes_interaction_claim.add.into_iter().next(),
            add_opcode_small: opcodes_interaction_claim.add_small.into_iter().next(),
            add_ap_opcode: opcodes_interaction_claim.add_ap.into_iter().next(),
            assert_eq_opcode: opcodes_interaction_claim.assert_eq.into_iter().next(),
            assert_eq_opcode_imm: opcodes_interaction_claim.assert_eq_imm.into_iter().next(),
            assert_eq_opcode_double_deref: opcodes_interaction_claim
                .assert_eq_double_deref
                .into_iter()
                .next(),
            blake_compress_opcode: opcodes_interaction_claim.blake.into_iter().next(),
            call_opcode_abs: opcodes_interaction_claim.call.into_iter().next(),
            call_opcode_rel_imm: opcodes_interaction_claim.call_rel_imm.into_iter().next(),
            generic_opcode: opcodes_interaction_claim.generic.into_iter().next(),
            jnz_opcode_non_taken: opcodes_interaction_claim.jnz.into_iter().next(),
            jnz_opcode_taken: opcodes_interaction_claim.jnz_taken.into_iter().next(),
            jump_opcode_abs: opcodes_interaction_claim.jump.into_iter().next(),
            jump_opcode_double_deref: opcodes_interaction_claim
                .jump_double_deref
                .into_iter()
                .next(),
            jump_opcode_rel: opcodes_interaction_claim.jump_rel.into_iter().next(),
            jump_opcode_rel_imm: opcodes_interaction_claim.jump_rel_imm.into_iter().next(),
            mul_opcode: opcodes_interaction_claim.mul.into_iter().next(),
            mul_opcode_small: opcodes_interaction_claim.mul_small.into_iter().next(),
            qm_31_add_mul_opcode: opcodes_interaction_claim.qm31.into_iter().next(),
            ret_opcode: opcodes_interaction_claim.ret.into_iter().next(),
            // verify_instruction
            verify_instruction: Some(verify_instruction_interaction_claim),
            // Blake context (flat)
            blake_round: blake_round_interaction,
            blake_g: blake_g_interaction,
            blake_round_sigma: blake_sigma_interaction,
            triple_xor_32: triple_xor_32_interaction,
            verify_bitwise_xor_12: vbx_12_interaction,
            // Builtins
            add_mod_builtin: add_mod_bi,
            bitwise_builtin: bitwise_bi,
            mul_mod_builtin: mul_mod_bi,
            pedersen_builtin: pedersen_bi, // Wide only
            pedersen_builtin_narrow_windows: pedersen_narrow_bi, // Narrow only
            poseidon_builtin: poseidon_bi,
            range_check96_builtin: range_check_96_bi,
            range_check_builtin: range_check_128_bi,
            // Pedersen context (flat) — mutually exclusive: Wide (w18) or Narrow (w9)
            pedersen_aggregator_window_bits_18: pedersen_context_interaction
                .as_ref()
                .map(|c| c.pedersen_aggregator.clone()),
            partial_ec_mul_window_bits_18: pedersen_context_interaction
                .as_ref()
                .map(|c| c.partial_ec_mul.clone()),
            pedersen_points_table_window_bits_18: pedersen_context_interaction
                .map(|c| c.pedersen_points_table),
            pedersen_aggregator_window_bits_9: pedersen_narrow_context_interaction
                .as_ref()
                .map(|c| c.pedersen_aggregator.clone()),
            partial_ec_mul_window_bits_9: pedersen_narrow_context_interaction
                .as_ref()
                .map(|c| c.partial_ec_mul.clone()),
            pedersen_points_table_window_bits_9: pedersen_narrow_context_interaction
                .map(|c| c.pedersen_points_table),
            // Poseidon context (flat)
            poseidon_aggregator: poseidon_aggregator_interaction, // native CUDA pipeline
            poseidon_3_partial_rounds_chain: poseidon_context_interaction
                .as_ref()
                .map(|c| c.poseidon_3_partial_rounds_chain.clone()),
            poseidon_full_round_chain: poseidon_context_interaction
                .as_ref()
                .map(|c| c.poseidon_full_round_chain.clone()),
            cube_252: poseidon_context_interaction
                .as_ref()
                .map(|c| c.cube_252.clone()),
            poseidon_round_keys: poseidon_context_interaction
                .as_ref()
                .map(|c| c.poseidon_round_keys.clone()),
            range_check_252_width_27: poseidon_context_interaction
                .map(|c| c.range_check_252_width_27),
            // Memory
            memory_address_to_id: Some(memory_address_to_id_interaction_claim),
            memory_id_to_big: Some(memory_id_to_value_interaction_claim),
            // Range checks (flat)
            range_check_6: Some(range_checks_interaction_claim.rc_6),
            range_check_8: Some(range_checks_interaction_claim.rc_8),
            range_check_11: Some(range_checks_interaction_claim.rc_11),
            range_check_12: Some(range_checks_interaction_claim.rc_12),
            range_check_18: Some(range_checks_interaction_claim.rc_18),
            range_check_20: Some(range_checks_interaction_claim.rc_20),
            range_check_4_3: Some(range_checks_interaction_claim.rc_4_3),
            range_check_4_4: Some(range_checks_interaction_claim.rc_4_4),
            range_check_9_9: Some(range_checks_interaction_claim.rc_9_9),
            range_check_7_2_5: Some(range_checks_interaction_claim.rc_7_2_5),
            range_check_3_6_6_3: Some(range_checks_interaction_claim.rc_3_6_6_3),
            range_check_4_4_4_4: Some(range_checks_interaction_claim.rc_4_4_4_4),
            range_check_3_3_3_3_3: Some(range_checks_interaction_claim.rc_3_3_3_3_3),
            // VBX
            verify_bitwise_xor_4: Some(verify_bitwise_xor_4_interaction_claim),
            verify_bitwise_xor_7: Some(verify_bitwise_xor_7_interaction_claim),
            verify_bitwise_xor_8: Some(verify_bitwise_xor_8_interaction_claim),
            verify_bitwise_xor_9: Some(verify_bitwise_xor_9_interaction_claim),
        }
    }
}

/// Create a native CUDA claim generator with per-component CUDA dispatch.
pub fn create_native_cairo_cuda_claim_generator(
    input: ProverInput,
    preprocessed_trace: Arc<PreProcessedTrace>,
) -> NativeCairoCudaClaimGenerator {
    NativeCairoCudaClaimGenerator::new(input, preprocessed_trace)
}

// CUDA-COVERAGE: poseidon_builtin — NATIVE CUDA path (split, 6 columns).
// Implementation in poseidon_builtin_split_cuda.rs (6 trace cols, 4 logup cols).
// Replaces the old SIMD hybrid PoseidonBuiltinCudaClaimGenerator.

// CUDA-COVERAGE: pedersen_builtin_narrow_windows (window_bits_9) — NATIVE CUDA path.
// Implementation in pedersen_builtin_narrow_cuda.rs (same kernel as Wide variant,
// only aggregator relation_id differs: 194336987 vs 520578465).

// PedersenNarrowContextCuda* structs removed — replaced by native CUDA in pedersen_wb9_cuda
