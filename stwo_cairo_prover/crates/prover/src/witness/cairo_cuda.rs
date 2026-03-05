//! SIMD/CUDA trace conversion utilities and Cairo CUDA claim generator.
//!
//! This module provides:
//! - Conversion utilities between SIMD and CUDA CircleEvaluations.
//! - `SimdTraceCollector` and `SimdToCudaBridge` for bridging SIMD and CUDA tree builders.
//! - `CairoCudaClaimGenerator` (V0 path): SIMD fallback via `SimdToCudaBridge`.
//! - `NativeCairoCudaClaimGenerator` (native path): Per-component native CUDA trace gen.

use std::array;
use std::collections::HashSet;
use std::sync::Arc;

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
use tracing::{span, Level};

use super::cairo::create_cairo_claim_generator;
use super::cairo_claim_generator::{CairoClaimGenerator, CairoInteractionClaimGenerator};
use super::components_cuda::builtins::{
    add_mod_builtin_cuda, bitwise_builtin_cuda, mul_mod_builtin_cuda, pedersen_builtin_cuda,
    poseidon_builtin_cuda, range_check_bits_128_builtin_cuda, range_check_bits_96_builtin_cuda,
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
use crate::witness::components_cuda::pedersen_cuda;
use crate::witness::components_cuda::pedersen_cuda::{
    PedersenContextCudaClaimGenerator, PedersenContextCudaInteractionClaimGenerator,
    PedersenInteractionMode,
};
use crate::witness::components_cuda::poseidon_cuda::{
    PoseidonContextCudaClaimGenerator, PoseidonContextCudaInteractionClaimGenerator,
};
use crate::witness::utils::TreeBuilder;

use cairo_air::air::{MemorySmallValue, PublicData, PublicMemory, PublicSegmentRanges, SegmentRange};

// SIMD components for hybrid poseidon pipeline (builtin + aggregator + chain state objects)
use super::components::{
    cube_252, memory_address_to_id, memory_id_to_big, poseidon_3_partial_rounds_chain,
    poseidon_aggregator, poseidon_builtin, poseidon_full_round_chain,
    range_check_252_width_27, range_check_3_3_3_3_3, range_check_4_4, range_check_4_4_4_4,
};

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
    pedersen_builtin_cuda: Option<pedersen_builtin_cuda::CudaClaimGenerator>,
    #[allow(dead_code)] // Monolithic kernel incompatible with split AIR interaction trace
    poseidon_builtin_cuda: Option<poseidon_builtin_cuda::CudaClaimGenerator>,
    /// Poseidon segment info (log_size, segment_start) for SIMD poseidon_builtin path.
    poseidon_segment: Option<(u32, u32)>,

    // Pedersen context (CUDA wrapper)
    pedersen_context_cuda: Option<PedersenContextCudaClaimGenerator>,

    // Poseidon context (CUDA wrapper)
    poseidon_context_cuda: Option<PoseidonContextCudaClaimGenerator>,

    // Shared state for SIMD hybrid poseidon pipeline
    memory: Arc<Memory>,
    preprocessed_trace: Arc<PreProcessedTrace>,
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
        let pedersen_builtin_cuda_gen = builtin_segments.pedersen_builtin.map(|seg| {
            let n = (seg.stop_ptr - seg.begin_addr) / PEDERSEN_BUILTIN_MEMORY_CELLS;
            pedersen_builtin_cuda::CudaClaimGenerator::new(n.ilog2(), seg.begin_addr as u32)
        });
        // poseidon_builtin uses SIMD hybrid path (the monolithic CUDA kernel's interaction
        // trace is incompatible with the split v1.1.0 AIR). Store segment info for SIMD path.
        let poseidon_segment = builtin_segments.poseidon_builtin.map(|seg| {
            let n = (seg.stop_ptr - seg.begin_addr) / POSEIDON_BUILTIN_MEMORY_CELLS;
            (n.ilog2(), seg.begin_addr as u32)
        });

        // --- Pedersen context CUDA wrapper (conditional) ---
        let pedersen_context_cuda = if has_pedersen {
            Some(PedersenContextCudaClaimGenerator::new(
                preprocessed_trace.clone(),
            ))
        } else {
            None
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
            pedersen_builtin_cuda: pedersen_builtin_cuda_gen,
            poseidon_builtin_cuda: None, // SIMD hybrid: monolithic kernel incompatible with split AIR
            poseidon_segment,
            pedersen_context_cuda,
            poseidon_context_cuda,
            memory: memory.clone(),
            preprocessed_trace: preprocessed_trace.clone(),
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

        let (pedersen_builtin_claim, pedersen_builtin_interaction_gen) = self
            .pedersen_builtin_cuda
            .map(|gen| {
                let pedersen_ctx = self.pedersen_context_cuda.as_ref().unwrap();
                gen.write_trace(
                    tree_builder,
                    &mut self.memory_address_to_id_cuda,
                    pedersen_ctx.pedersen_aggregator_cuda.inner(),
                )
            })
            .unzip();

        // poseidon_builtin: SIMD hybrid path.
        // The monolithic CUDA kernel's interaction trace is incompatible with the split v1.1.0 AIR,
        // so poseidon_builtin runs via SIMD with SimdToCudaBridge.
        // Must be committed HERE (between pedersen and range_check_96) for correct tree ordering.
        let (
            poseidon_builtin_claim,
            poseidon_builtin_simd_interaction_gen,
            poseidon_simd_aggregator,
        ) = if let Some((log_size, segment_start)) = self.poseidon_segment {
            // Create SIMD memory_address_to_id (for poseidon_builtin deduce_output)
            let simd_mem_addr_to_id =
                memory_address_to_id::ClaimGenerator::new(self.memory.clone());

            // Create SIMD poseidon_builtin + aggregator (aggregator populated by write_trace)
            let simd_poseidon_builtin =
                poseidon_builtin::ClaimGenerator::new(log_size, segment_start);
            let simd_aggregator = poseidon_aggregator::ClaimGenerator::new();

            // Run SIMD poseidon_builtin write_trace
            let (pb_trace, pb_claim, pb_interaction_gen) =
                simd_poseidon_builtin.write_trace(
                    &simd_mem_addr_to_id,
                    &simd_aggregator,
                );

            // Bridge poseidon_builtin trace (6 cols) to CUDA tree (correct tree position)
            {
                let mut bridge = SimdToCudaBridge::new(tree_builder);
                bridge.extend_evals(pb_trace.to_evals());
            }

            // Feed CUDA memory_address_to_id with poseidon's 6 addresses per row
            {
                let n_rows = 1usize << log_size;
                for row in 0..n_rows {
                    for offset in 0..6u32 {
                        let addr = M31::from_u32_unchecked(
                            segment_start + (row as u32) * 6 + offset,
                        );
                        self.memory_address_to_id_cuda.add_cuda_input(&addr);
                    }
                }
            }

            (Some(pb_claim), Some(pb_interaction_gen), Some(simd_aggregator))
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
        let (
            pedersen_aggregator_claim,
            partial_ec_mul_claim,
            pedersen_points_table_claim,
            pedersen_interaction_mode,
        ) = {
            match self.pedersen_context_cuda {
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
        span.exit();

        // ==== 5.5 SIMD aggregator + feed CUDA chains ====
        // The SIMD poseidon_aggregator was created and populated by step 4's poseidon_builtin.
        // Now run the aggregator (for dedup), bridge its trace, then convert chain
        // packed_inputs to CUDA format and feed the CUDA chain generators.
        let span = span!(Level::INFO, "base_trace: poseidon SIMD hybrid (5.5 aggregator)").entered();
        let (
            poseidon_aggregator_claim,
            poseidon_aggregator_simd_interaction_gen,
        ): (
            Option<cairo_air::components::poseidon_aggregator::Claim>,
            Option<poseidon_aggregator::InteractionClaimGenerator>,
        ) = if let Some(simd_aggregator) = poseidon_simd_aggregator {
                // Create SIMD chain state objects (aggregator will populate their packed_inputs)
                let mut simd_full_round_chain =
                    poseidon_full_round_chain::ClaimGenerator::new();
                let mut simd_3_partial =
                    poseidon_3_partial_rounds_chain::ClaimGenerator::new();
                let mut simd_cube_252 = cube_252::ClaimGenerator::new();
                let mut simd_rc_252w27 = range_check_252_width_27::ClaimGenerator::new();

                // Range check generators (aggregator-level only — chain-level RCs
                // are handled internally by CUDA chain generators)
                let simd_rc_3_3_3_3_3 =
                    range_check_3_3_3_3_3::ClaimGenerator::new(self.preprocessed_trace.clone());
                let simd_rc_4_4_4_4 =
                    range_check_4_4_4_4::ClaimGenerator::new(self.preprocessed_trace.clone());
                let simd_rc_4_4 =
                    range_check_4_4::ClaimGenerator::new(self.preprocessed_trace.clone());

                // Memory id_to_big for deduce_output
                let simd_mem_id_to_big =
                    memory_id_to_big::ClaimGenerator::new(self.memory.clone());

                // 1. Run SIMD aggregator write_trace
                let (pa_trace, pa_claim, pa_interaction_gen) =
                    simd_aggregator.write_trace(
                        &simd_mem_id_to_big,
                        &mut simd_full_round_chain,
                        &mut simd_rc_252w27,
                        &mut simd_cube_252,
                        &simd_rc_3_3_3_3_3,
                        &simd_rc_4_4_4_4,
                        &simd_rc_4_4,
                        &mut simd_3_partial,
                    );

                // Bridge aggregator trace (342 cols) to CUDA tree
                {
                    let mut bridge = SimdToCudaBridge::new(tree_builder);
                    bridge.extend_evals(pa_trace.to_evals());
                }

                // Merge SIMD memory_id_to_big multiplicities into CUDA.
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
                    self.memory_id_to_big_cuda
                        .merge_simd_multiplicities(&big_mults, &small_mults);
                }

                // 2. Convert SIMD chain packed_inputs → CUDA format and feed CUDA generators
                if let Some(ref mut ctx) = self.poseidon_context_cuda {
                    // Full round chain
                    let full_round_cuda_inputs =
                        simd_to_cuda_full_round_chain(&simd_full_round_chain.packed_inputs);
                    ctx.poseidon_full_round_chain_cuda
                        .add_cuda_inputs(&full_round_cuda_inputs);

                    // 3 partial rounds chain
                    let partial_cuda_inputs =
                        simd_to_cuda_3_partial(&simd_3_partial.packed_inputs);
                    ctx.poseidon_3_partial_rounds_chain_cuda
                        .add_cuda_inputs(&partial_cuda_inputs);

                    // cube_252 (aggregator's direct contribution)
                    let cube_cuda_inputs =
                        packed_felt252w27_to_basefieldvec_10(&simd_cube_252.packed_inputs);
                    ctx.cube_252_cuda.add_cuda_inputs(&cube_cuda_inputs);

                    // range_check_252_width_27 (aggregator's direct contribution)
                    let rc_cuda_inputs =
                        packed_felt252w27_to_basefieldvec_10(&simd_rc_252w27.packed_inputs);
                    ctx.range_check_252_width_27_cuda.add_cuda_inputs(&rc_cuda_inputs);
                }

                // 3. Merge aggregator-level range check multiplicities only
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
                merge_single_rc!(simd_rc_3_3_3_3_3, self.range_checks_trace_generator.rc_3_3_3_3_3_trace_generator);
                merge_single_rc!(simd_rc_4_4_4_4, self.range_checks_trace_generator.rc_4_4_4_4_trace_generator);
                merge_single_rc!(simd_rc_4_4, self.range_checks_trace_generator.rc_4_4_trace_generator);

                (
                    Some(pa_claim),
                    Some(pa_interaction_gen),
                )
            } else {
                (None, None)
            };
        span.exit();

        // ==== 6. Generate poseidon chain traces (CUDA) ====
        // PoseidonContextCudaClaimGenerator runs the CUDA chain generators
        // (full_round_chain, 3_partial_rounds_chain, cube_252, round_keys, rc_252w27).
        // Their inputs were populated either by step 5.5 (SIMD aggregator conversion)
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
            assert_eq_opcode_double_deref: opcodes_claim
                .assert_eq_double_deref
                .into_iter()
                .next(),
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
            pedersen_builtin: pedersen_builtin_claim,
            pedersen_builtin_narrow_windows: None, // v1.1.0 only — SIMD fallback
            poseidon_builtin: poseidon_builtin_claim,
            range_check96_builtin: range_check_96_builtin_claim,
            range_check_builtin: range_check_128_builtin_claim,
            // Pedersen context (flat)
            pedersen_aggregator_window_bits_18: pedersen_aggregator_claim,
            partial_ec_mul_window_bits_18: partial_ec_mul_claim,
            pedersen_points_table_window_bits_18: pedersen_points_table_claim,
            pedersen_aggregator_window_bits_9: None, // v1.1.0 only — no CUDA port
            partial_ec_mul_window_bits_9: None,
            pedersen_points_table_window_bits_9: None,
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
            range_check_252_width_27: poseidon_context_claim
                .map(|c| c.range_check_252_width_27),
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
                pedersen: pedersen_builtin_interaction_gen,
                poseidon: None, // SIMD hybrid path — handled via poseidon_builtin_simd_interaction_gen
                range_check_96: range_check_96_builtin_interaction_gen,
                range_check_128: range_check_128_builtin_interaction_gen,
            },
            poseidon_builtin_simd_interaction_gen,
            poseidon_aggregator_simd_interaction_gen,
            pedersen_context_interaction_gen: pedersen_interaction_mode,
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
    // SIMD hybrid poseidon_builtin interaction generator
    poseidon_builtin_simd_interaction_gen:
        Option<poseidon_builtin::InteractionClaimGenerator>,
    // SIMD hybrid poseidon aggregator interaction generator
    poseidon_aggregator_simd_interaction_gen:
        Option<poseidon_aggregator::InteractionClaimGenerator>,
    pedersen_context_interaction_gen: Option<PedersenInteractionMode>,
    // CUDA poseidon chain interaction generator (full_round, 3_partial, cube_252, round_keys, rc_252w27)
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
        let pedersen_bi = self
            .builtins_interaction_gen
            .pedersen
            .map(|gen| gen.write_interaction_trace(tree_builder, common_lookup_elements));
        // poseidon_builtin: SIMD hybrid path (bridge to CUDA tree)
        let poseidon_bi = self
            .poseidon_builtin_simd_interaction_gen
            .map(|gen| {
                let (evals, interaction_claim) =
                    gen.write_interaction_trace(common_lookup_elements);
                let mut bridge = SimdToCudaBridge::new(tree_builder);
                bridge.extend_evals(evals);
                interaction_claim
            });
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
        let pedersen_context_interaction = match self.pedersen_context_interaction_gen {
            Some(gen) if !gen.is_empty() => {
                let result = gen.write_interaction_trace(tree_builder, common_lookup_elements);
                result.claim
            }
            _ => None,
        };
        span.exit();

        // ==== 5.5. poseidon_aggregator interaction trace (SIMD hybrid) ====
        let span =
            span!(Level::INFO, "interaction_trace: poseidon_aggregator (SIMD hybrid)").entered();
        let poseidon_aggregator_interaction =
            self.poseidon_aggregator_simd_interaction_gen
                .map(|gen| {
                    let (evals, interaction_claim) =
                        gen.write_interaction_trace(common_lookup_elements);
                    let mut bridge = SimdToCudaBridge::new(tree_builder);
                    bridge.extend_evals(evals);
                    interaction_claim
                });
        span.exit();

        // ==== 6. poseidon_context interaction trace (CUDA chains) ====
        let span = span!(Level::INFO, "interaction_trace: poseidon_context").entered();
        let poseidon_context_interaction = match self.poseidon_context_interaction_gen {
            Some(gen) => {
                let result =
                    gen.write_interaction_trace(tree_builder, common_lookup_elements);
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
            pedersen_builtin: pedersen_bi,
            pedersen_builtin_narrow_windows: None,
            poseidon_builtin: poseidon_bi,
            range_check96_builtin: range_check_96_bi,
            range_check_builtin: range_check_128_bi,
            // Pedersen context (flat)
            pedersen_aggregator_window_bits_18: pedersen_context_interaction
                .as_ref()
                .map(|c| c.pedersen_aggregator.clone()),
            partial_ec_mul_window_bits_18: pedersen_context_interaction
                .as_ref()
                .map(|c| c.partial_ec_mul.clone()),
            pedersen_points_table_window_bits_18: pedersen_context_interaction
                .map(|c| c.pedersen_points_table),
            pedersen_aggregator_window_bits_9: None,
            partial_ec_mul_window_bits_9: None,
            pedersen_points_table_window_bits_9: None,
            // Poseidon context (flat)
            poseidon_aggregator: poseidon_aggregator_interaction, // SIMD hybrid pipeline
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

// ---------------------------------------------------------------------------
// SIMD → CUDA conversion helpers for poseidon chain generators
// ---------------------------------------------------------------------------

use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo_cairo_common::prover_types::simd::PackedFelt252Width27;

use super::components_cuda::{
    poseidon_3_partial_rounds_chain_cuda, poseidon_full_round_chain_cuda,
};

/// Convert a slice of PackedFelt252Width27 values to 10 BaseFieldVec columns.
fn packed_felt252w27_to_basefieldvec_10(
    packed: &[PackedFelt252Width27],
) -> [BaseFieldVec; 10] {
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
        state_0: state[0].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
        state_1: state[1].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
        state_2: state[2].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
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
        state_0: state[0].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
        state_1: state[1].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
        state_2: state[2].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
        state_3: state[3].each_ref().map(|v| BaseFieldVec::from_vec(v.clone())),
    }
}
