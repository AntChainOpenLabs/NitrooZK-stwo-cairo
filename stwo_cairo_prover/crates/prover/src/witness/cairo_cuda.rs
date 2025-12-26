//! Mixed SIMD/CUDA Cairo claim generator.
//!
//! This module provides `CairoCudaClaimGenerator` which uses CUDA for opcode trace generation
//! while other components use SIMD mode. The traces are combined in the correct order for
//! the AIR constraint system.

use std::array;

use cairo_air::air::{
    CairoClaim, CairoInteractionClaim, CairoInteractionElements, MemorySmallValue, PublicData,
    PublicMemory, PublicSegmentRanges, SegmentRange,
};
use itertools::Itertools;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::TreeSubspan;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_adapter::{ProverInput, PublicSegmentContext};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::MAX_SEQUENCE_LOG_SIZE;
use tracing::{span, Level};

use super::blake_context_cuda::{BlakeContextCudaClaimGenerator, BlakeContextCudaInteractionClaimGenerator};
use super::builtins_cuda::{BuiltinsCudaClaimGenerator, BuiltinsCudaInteractionClaimGenerator};
use super::components_cuda::{
    memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_11_cuda, range_check_18_cuda,
    range_check_4_4_4_4_cuda, range_check_7_2_5_cuda,
    verify_bitwise_xor_4_cuda, verify_bitwise_xor_7_cuda, verify_bitwise_xor_8_cuda,
    verify_bitwise_xor_8_b_cuda, verify_bitwise_xor_9_cuda, verify_instruction_cuda,
    blake_round_cuda, triple_xor_32_cuda,
};
use super::opcodes_cuda::{OpcodesCudaClaimGenerator, OpcodesCudaInteractionClaimGenerator};
use super::range_checks_cuda::{RangeChecksCudaClaimGenerator, RangeChecksCudaInteractionClaimGenerator};
use super::components_cuda::pedersen_cuda::{
    PedersenContextCudaClaimGenerator, PedersenContextCudaInteractionClaimGenerator,
};
use super::components_cuda::poseidon_cuda::{
    PoseidonContextCudaClaimGenerator, PoseidonContextCudaInteractionClaimGenerator,
};
use crate::witness::components::{
    memory_address_to_id, memory_id_to_big, verify_bitwise_xor_4, verify_bitwise_xor_7,
    verify_bitwise_xor_8, verify_bitwise_xor_8_b, verify_bitwise_xor_9, verify_instruction,
};
use crate::witness::utils::TreeBuilder;

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

    /// Add CUDA evaluations to this collector by converting them to SIMD format.
    /// This allows CUDA-generated traces to be added in the correct order.
    pub fn extend_cuda_evals(
        &mut self,
        evals: impl IntoIterator<Item = CircleEvaluation<CudaBackend, M31, BitReversedOrder>>,
    ) {
        self.traces
            .extend(evals.into_iter().map(convert_cuda_to_simd_evaluation));
    }

    /// Convert collected traces to CudaBackend format and extend the CUDA tree builder.
    pub fn extend_cuda_tree_builder(self, tree_builder: &mut impl TreeBuilder<CudaBackend>) {
        tree_builder.extend_evals(
            self.traces
                .into_iter()
                .map(convert_simd_to_cuda_evaluation),
        );
    }
}

impl TreeBuilder<SimdBackend> for SimdTraceCollector {
    fn extend_evals(
        &mut self,
        columns: impl IntoIterator<Item = CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
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

/// Cairo claim generator with CUDA acceleration for opcodes.
///
/// Uses CUDA for opcode trace generation while other components use SIMD mode.
/// The traces are combined in the correct order for the AIR constraint system.

pub struct CairoCudaClaimGenerator {
    public_data: PublicData,

    // CUDA generators for opcode dependencies
    memory_address_to_id_cuda: memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_cuda: memory_id_to_big_cuda::CudaClaimGenerator,
    verify_instruction_cuda: verify_instruction_cuda::CudaClaimGenerator,

    // CUDA generators for range checks
    range_check_11_cuda: range_check_11_cuda::CudaClaimGenerator,
    range_check_18_cuda: range_check_18_cuda::CudaClaimGenerator,
    range_check_4_4_4_4_cuda: range_check_4_4_4_4_cuda::CudaClaimGenerator,
    range_check_7_2_5_cuda: range_check_7_2_5_cuda::CudaClaimGenerator,

    // CUDA generators for blake sub-components
    blake_round_cuda: blake_round_cuda::CudaClaimGenerator,
    triple_xor_32_cuda: triple_xor_32_cuda::CudaClaimGenerator,

    // CUDA generators for opcodes (replaces OpcodesClaimGenerator)
    opcodes_cuda: OpcodesCudaClaimGenerator,

    // Internal components (SIMD)
    verify_instruction_trace_generator: verify_instruction::ClaimGenerator,
    blake_context_trace_generator: BlakeContextCudaClaimGenerator,
    builtins: BuiltinsCudaClaimGenerator,
    pedersen_context_trace_generator: PedersenContextCudaClaimGenerator,
    poseidon_context_trace_generator: PoseidonContextCudaClaimGenerator,
    memory_address_to_id_trace_generator: memory_address_to_id::ClaimGenerator,
    memory_id_to_value_trace_generator: memory_id_to_big::ClaimGenerator,
    range_checks_trace_generator: RangeChecksCudaClaimGenerator,
    verify_bitwise_xor_4_trace_generator: verify_bitwise_xor_4::ClaimGenerator,
    verify_bitwise_xor_7_trace_generator: verify_bitwise_xor_7::ClaimGenerator,
    verify_bitwise_xor_8_trace_generator: verify_bitwise_xor_8::ClaimGenerator,
    verify_bitwise_xor_8_b_trace_generator: verify_bitwise_xor_8_b::ClaimGenerator,
    verify_bitwise_xor_9_trace_generator: verify_bitwise_xor_9::ClaimGenerator,

    // CUDA generators for verify_bitwise_xor (multiplicity tracking)
    verify_bitwise_xor_4_cuda: verify_bitwise_xor_4_cuda::CudaClaimGenerator,
    verify_bitwise_xor_7_cuda: verify_bitwise_xor_7_cuda::CudaClaimGenerator,
    verify_bitwise_xor_8_cuda: verify_bitwise_xor_8_cuda::CudaClaimGenerator,
    verify_bitwise_xor_8_b_cuda: verify_bitwise_xor_8_b_cuda::CudaClaimGenerator,
    verify_bitwise_xor_9_cuda: verify_bitwise_xor_9_cuda::CudaClaimGenerator,
}

impl CairoCudaClaimGenerator {
    pub fn new(
        ProverInput {
            state_transitions,
            memory,
            inst_cache,
            public_memory_addresses,
            builtin_segments,
            public_segment_context,
            ..
        }: ProverInput,
    ) -> Self {
        let initial_state = state_transitions.initial_state;
        let final_state = state_transitions.final_state;

        // 1. Create CUDA dependency state generators
        let memory_address_to_id_cuda =
            memory_address_to_id_cuda::CudaClaimGenerator::new(&memory);
        let memory_id_to_big_cuda = memory_id_to_big_cuda::CudaClaimGenerator::new(&memory);
        let verify_instruction_cuda =
            verify_instruction_cuda::CudaClaimGenerator::new(inst_cache.clone());

        // 2. Create CUDA generators for range checks
        let range_check_11_cuda = range_check_11_cuda::CudaClaimGenerator::new_rc_11();
        let range_check_18_cuda = range_check_18_cuda::CudaClaimGenerator::new_rc_18();
        let range_check_4_4_4_4_cuda =
            range_check_4_4_4_4_cuda::CudaClaimGenerator::new_rc_4_4_4_4();
        let range_check_7_2_5_cuda = range_check_7_2_5_cuda::CudaClaimGenerator::new_rc_7_2_5();

        // 3. Create CUDA generators for blake sub-components
        let blake_round_cuda = blake_round_cuda::CudaClaimGenerator::new(memory.clone());
        let triple_xor_32_cuda = triple_xor_32_cuda::CudaClaimGenerator::new();

        // 4. Create CUDA generators for opcodes (takes all opcodes from state_transitions)
        let opcodes_cuda = OpcodesCudaClaimGenerator::new(state_transitions);

        // 4. Initialize other generators same as CairoClaimGenerator
        let verify_instruction_trace_generator =
            verify_instruction::ClaimGenerator::new(inst_cache);

        let builtins = BuiltinsCudaClaimGenerator::new(builtin_segments);
        let pedersen_context_trace_generator = PedersenContextCudaClaimGenerator::new();
        let poseidon_context_trace_generator = PoseidonContextCudaClaimGenerator::new();
        let memory_address_to_id_trace_generator =
            memory_address_to_id::ClaimGenerator::new(&memory);
        let memory_id_to_value_trace_generator = memory_id_to_big::ClaimGenerator::new(&memory);
        let range_checks_trace_generator = RangeChecksCudaClaimGenerator::new();
        let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4::ClaimGenerator::new();
        let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7::ClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b::ClaimGenerator::new();
        let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9::ClaimGenerator::new();

        // CUDA generators for verify_bitwise_xor (multiplicity tracking)
        let verify_bitwise_xor_4_cuda = verify_bitwise_xor_4_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_7_cuda = verify_bitwise_xor_7_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_8_cuda = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_8_b_cuda = verify_bitwise_xor_8_b_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_9_cuda = verify_bitwise_xor_9_cuda::CudaClaimGenerator::new();

        // Yield public memory - add to SIMD generators (matching original cairo.rs behavior)
        for addr in public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_trace_generator.get_id(addr);
            memory_address_to_id_trace_generator.add_input(&addr);
            memory_id_to_value_trace_generator.add_input(&id);
        }

        // Public data
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

        let blake_context_trace_generator = BlakeContextCudaClaimGenerator::new(memory);

        Self {
            public_data,
            memory_address_to_id_cuda,
            memory_id_to_big_cuda,
            verify_instruction_cuda,
            range_check_11_cuda,
            range_check_18_cuda,
            range_check_4_4_4_4_cuda,
            range_check_7_2_5_cuda,
            blake_round_cuda,
            triple_xor_32_cuda,
            opcodes_cuda,
            verify_instruction_trace_generator,
            blake_context_trace_generator,
            builtins,
            pedersen_context_trace_generator,
            poseidon_context_trace_generator,
            memory_address_to_id_trace_generator,
            memory_id_to_value_trace_generator,
            range_checks_trace_generator,
            verify_bitwise_xor_4_trace_generator,
            verify_bitwise_xor_7_trace_generator,
            verify_bitwise_xor_8_trace_generator,
            verify_bitwise_xor_8_b_trace_generator,
            verify_bitwise_xor_9_trace_generator,
            verify_bitwise_xor_4_cuda,
            verify_bitwise_xor_7_cuda,
            verify_bitwise_xor_8_cuda,
            verify_bitwise_xor_8_b_cuda,
            verify_bitwise_xor_9_cuda,
        }
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (CairoClaim, CairoCudaInteractionClaimGenerator) {
        let span = span!(Level::INFO, "write opcode trace (mixed CUDA/SIMD)").entered();

        // ==== 1. Generate opcode traces (CUDA for all opcodes) ====
        let (opcodes_claim, opcodes_interaction_gen) = self.opcodes_cuda.write_trace(
            tree_builder,
            &mut self.memory_address_to_id_cuda,
            &self.memory_id_to_big_cuda,
            &self.range_check_11_cuda,
            &self.range_check_18_cuda,
            &self.range_check_4_4_4_4_cuda,
            &self.range_check_7_2_5_cuda,
            &self.verify_instruction_cuda,
            &self.verify_bitwise_xor_8_cuda,
            &mut self.blake_round_cuda,
            &mut self.triple_xor_32_cuda,
            &mut self.blake_context_trace_generator,
            &self.memory_address_to_id_trace_generator,
            &self.memory_id_to_value_trace_generator,
            &self.verify_instruction_trace_generator,
            &self.range_checks_trace_generator,
            &self.verify_bitwise_xor_8_trace_generator,
        );

        span.exit();

        // ==== 2. Generate internal component traces via SIMD, convert to CUDA ====
        let span = span!(Level::INFO, "internal component trace (SIMD)").entered();

        let mut simd_collector = SimdTraceCollector::new(1);

        let (verify_instruction_claim, verify_instruction_interaction_gen) =
            self.verify_instruction_trace_generator.write_trace(
                &mut simd_collector,
                &self.memory_address_to_id_trace_generator,
                &self.memory_id_to_value_trace_generator,
                &self.range_checks_trace_generator.rc_4_3_trace_generator,
                &self.range_checks_trace_generator.rc_7_2_5_simd_trace_generator,
            );

        let (blake_context_claim, blake_context_interaction_gen) =
            self.blake_context_trace_generator.write_trace(
                &mut simd_collector,
                &self.memory_address_to_id_trace_generator,
                &self.memory_id_to_value_trace_generator,
                &self.range_checks_trace_generator,
                &self.verify_bitwise_xor_4_trace_generator,
                &self.verify_bitwise_xor_7_trace_generator,
                &self.verify_bitwise_xor_8_trace_generator,
                &self.verify_bitwise_xor_8_b_trace_generator,
                &self.verify_bitwise_xor_9_trace_generator,
            );

        // Convert verify_instruction and blake_context SIMD traces to CUDA first
        simd_collector.extend_cuda_tree_builder(tree_builder);

        // ==== 3. Generate builtins traces directly to CUDA ====
        let (builtins_claim, builtins_interaction_gen) = self.builtins.write_trace(
            tree_builder,
            &mut self.memory_address_to_id_cuda,
            &self.memory_id_to_big_cuda,
            &self.memory_address_to_id_trace_generator,
            &self.memory_id_to_value_trace_generator,
            &mut self.pedersen_context_trace_generator,
            &mut self.poseidon_context_trace_generator,
            &self.range_checks_trace_generator.rc_6_trace_generator,
        );

        // Start a new SIMD collector for pedersen_context, poseidon_context, memory
        let mut simd_collector = SimdTraceCollector::new(1);

        let (pedersen_context_claim, pedersen_context_interaction_gen) = self
            .pedersen_context_trace_generator
            .write_trace(&mut simd_collector, &self.range_checks_trace_generator);

        let (poseidon_context_claim, poseidon_context_interaction_gen) = self
            .poseidon_context_trace_generator
            .write_trace(&mut simd_collector, &self.range_checks_trace_generator);

        // NOTE: memory_address_to_id uses SIMD trace generation for now.
        // CUDA trace generation causes poly.rs:257 errors when traces are added in multiple batches.
        // SIMD generator already has all multiplicities (CUDA opcodes add to both CUDA and SIMD generators)
        let (memory_address_to_id_claim, memory_address_to_id_interaction_gen) =
            self.memory_address_to_id_trace_generator.write_trace(&mut simd_collector);

        // Convert first batch of SIMD traces to CUDA and add to tree builder
        simd_collector.extend_cuda_tree_builder(tree_builder);

        // Start a new SIMD collector for remaining components
        let mut simd_collector = SimdTraceCollector::new(1);

        // Memory uses "Sequence", split it according to `MAX_SEQUENCE_LOG_SIZE`.
        const LOG_MAX_BIG_SIZE: u32 = MAX_SEQUENCE_LOG_SIZE;
        let (memory_id_to_value_claim, memory_id_to_value_interaction_gen) =
            self.memory_id_to_value_trace_generator.write_trace(
                &mut simd_collector,
                &self.range_checks_trace_generator.rc_9_9_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_b_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_c_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_d_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_e_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_f_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_g_trace_generator,
                &self.range_checks_trace_generator.rc_9_9_h_trace_generator,
                LOG_MAX_BIG_SIZE,
            );

        let (range_checks_claim, range_checks_interaction_gen) = self
            .range_checks_trace_generator
            .write_trace(&mut simd_collector);

        let (verify_bitwise_xor_4_claim, verify_bitwise_xor_4_interaction_gen) = self
            .verify_bitwise_xor_4_trace_generator
            .write_trace(&mut simd_collector);

        let (verify_bitwise_xor_7_claim, verify_bitwise_xor_7_interaction_gen) = self
            .verify_bitwise_xor_7_trace_generator
            .write_trace(&mut simd_collector);

        let (verify_bitwise_xor_8_claim, verify_bitwise_xor_8_interaction_gen) = self
            .verify_bitwise_xor_8_trace_generator
            .write_trace(&mut simd_collector);

        let (verify_bitwise_xor_8_b_claim, verify_bitwise_xor_8_b_interaction_gen) = self
            .verify_bitwise_xor_8_b_trace_generator
            .write_trace(&mut simd_collector);

        let (verify_bitwise_xor_9_claim, verify_bitwise_xor_9_interaction_gen) = self
            .verify_bitwise_xor_9_trace_generator
            .write_trace(&mut simd_collector);

        // Convert all SIMD traces to CUDA and extend tree builder
        simd_collector.extend_cuda_tree_builder(tree_builder);

        span.exit();

        (
            CairoClaim {
                public_data: self.public_data,
                opcodes: opcodes_claim,
                verify_instruction: verify_instruction_claim,
                blake_context: blake_context_claim,
                builtins: builtins_claim,
                pedersen_context: pedersen_context_claim,
                poseidon_context: poseidon_context_claim,
                memory_address_to_id: memory_address_to_id_claim,
                memory_id_to_value: memory_id_to_value_claim,
                range_checks: range_checks_claim,
                verify_bitwise_xor_4: verify_bitwise_xor_4_claim,
                verify_bitwise_xor_7: verify_bitwise_xor_7_claim,
                verify_bitwise_xor_8: verify_bitwise_xor_8_claim,
                verify_bitwise_xor_8_b: verify_bitwise_xor_8_b_claim,
                verify_bitwise_xor_9: verify_bitwise_xor_9_claim,
            },
            CairoCudaInteractionClaimGenerator {
                opcodes_interaction_gen,
                verify_instruction_interaction_gen,
                blake_context_interaction_gen,
                builtins_interaction_gen,
                pedersen_context_interaction_gen,
                poseidon_context_interaction_gen,
                memory_address_to_id_interaction_gen,
                memory_id_to_value_interaction_gen,
                range_checks_interaction_gen,
                verify_bitwise_xor_4_interaction_gen,
                verify_bitwise_xor_7_interaction_gen,
                verify_bitwise_xor_8_interaction_gen,
                verify_bitwise_xor_8_b_interaction_gen,
                verify_bitwise_xor_9_interaction_gen,
            },
        )
    }
}

pub struct CairoCudaInteractionClaimGenerator {
    opcodes_interaction_gen: OpcodesCudaInteractionClaimGenerator,
    verify_instruction_interaction_gen: verify_instruction::InteractionClaimGenerator,
    blake_context_interaction_gen: BlakeContextCudaInteractionClaimGenerator,
    builtins_interaction_gen: BuiltinsCudaInteractionClaimGenerator,
    pedersen_context_interaction_gen: PedersenContextCudaInteractionClaimGenerator,
    poseidon_context_interaction_gen: PoseidonContextCudaInteractionClaimGenerator,
    memory_address_to_id_interaction_gen: memory_address_to_id::InteractionClaimGenerator,
    memory_id_to_value_interaction_gen: memory_id_to_big::InteractionClaimGenerator,
    range_checks_interaction_gen: RangeChecksCudaInteractionClaimGenerator,
    verify_bitwise_xor_4_interaction_gen: verify_bitwise_xor_4::InteractionClaimGenerator,
    verify_bitwise_xor_7_interaction_gen: verify_bitwise_xor_7::InteractionClaimGenerator,
    verify_bitwise_xor_8_interaction_gen: verify_bitwise_xor_8::InteractionClaimGenerator,
    verify_bitwise_xor_8_b_interaction_gen: verify_bitwise_xor_8_b::InteractionClaimGenerator,
    verify_bitwise_xor_9_interaction_gen: verify_bitwise_xor_9::InteractionClaimGenerator,
}

impl CairoCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        interaction_elements: &CairoInteractionElements,
    ) -> CairoInteractionClaim {
        // ==== 1. Generate opcode interaction traces ====
        let opcodes_interaction_claim = self
            .opcodes_interaction_gen
            .write_interaction_trace(tree_builder, interaction_elements);

        // ==== 2. Generate other component interaction traces via SIMD, convert to CUDA ====
        let mut simd_collector = SimdTraceCollector::new(2);

        let verify_instruction_interaction_claim = self
            .verify_instruction_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.memory_address_to_id,
                &interaction_elements.memory_id_to_value,
                &interaction_elements.range_checks.rc_4_3,
                &interaction_elements.range_checks.rc_7_2_5,
                &interaction_elements.verify_instruction,
            );

        let blake_context_interaction_claim = self
            .blake_context_interaction_gen
            .write_interaction_trace(&mut simd_collector, interaction_elements);

        // Convert verify_instruction and blake_context SIMD interaction traces to CUDA
        simd_collector.extend_cuda_tree_builder(tree_builder);

        // ==== 3. Generate builtins interaction traces directly to CUDA ====
        let builtins_interaction_claims = self
            .builtins_interaction_gen
            .write_interaction_trace(tree_builder, interaction_elements);

        // ==== 4. Start new SIMD collector for remaining components ====
        let mut simd_collector = SimdTraceCollector::new(2);

        let pedersen_context_interaction_claim = self
            .pedersen_context_interaction_gen
            .write_interaction_trace(&mut simd_collector, interaction_elements);

        let poseidon_context_interaction_claim = self
            .poseidon_context_interaction_gen
            .write_interaction_trace(&mut simd_collector, interaction_elements);

        // NOTE: memory_address_to_id uses SIMD interaction trace generation for now.
        // CUDA trace generation causes poly.rs:257 errors when traces are added in multiple batches.
        let memory_address_to_id_interaction_claim = self
            .memory_address_to_id_interaction_gen
            .write_interaction_trace(&mut simd_collector, &interaction_elements.memory_address_to_id);

        // Convert pedersen_context, poseidon_context, memory_address_to_id SIMD traces to CUDA
        simd_collector.extend_cuda_tree_builder(tree_builder);

        // ==== 5. Start new SIMD collector for remaining components ====
        let mut simd_collector = SimdTraceCollector::new(2);

        let memory_id_to_value_interaction_claim = self
            .memory_id_to_value_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.memory_id_to_value,
                &interaction_elements.range_checks.rc_9_9,
                &interaction_elements.range_checks.rc_9_9_b,
                &interaction_elements.range_checks.rc_9_9_c,
                &interaction_elements.range_checks.rc_9_9_d,
                &interaction_elements.range_checks.rc_9_9_e,
                &interaction_elements.range_checks.rc_9_9_f,
                &interaction_elements.range_checks.rc_9_9_g,
                &interaction_elements.range_checks.rc_9_9_h,
            );

        let range_checks_interaction_claim = self
            .range_checks_interaction_gen
            .write_interaction_trace(&mut simd_collector, &interaction_elements.range_checks);

        let verify_bitwise_xor_4_interaction_claim = self
            .verify_bitwise_xor_4_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.verify_bitwise_xor_4,
            );

        let verify_bitwise_xor_7_interaction_claim = self
            .verify_bitwise_xor_7_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.verify_bitwise_xor_7,
            );

        let verify_bitwise_xor_8_interaction_claim = self
            .verify_bitwise_xor_8_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.verify_bitwise_xor_8,
            );

        let verify_bitwise_xor_8_b_interaction_claim = self
            .verify_bitwise_xor_8_b_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.verify_bitwise_xor_8_b,
            );

        let verify_bitwise_xor_9_interaction_claim = self
            .verify_bitwise_xor_9_interaction_gen
            .write_interaction_trace(
                &mut simd_collector,
                &interaction_elements.verify_bitwise_xor_9,
            );

        // ==== 6. Convert remaining SIMD traces to CUDA and extend tree builder ====
        simd_collector.extend_cuda_tree_builder(tree_builder);

        CairoInteractionClaim {
            opcodes: opcodes_interaction_claim,
            verify_instruction: verify_instruction_interaction_claim,
            blake_context: blake_context_interaction_claim,
            builtins: builtins_interaction_claims,
            pedersen_context: pedersen_context_interaction_claim,
            poseidon_context: poseidon_context_interaction_claim,
            memory_address_to_id: memory_address_to_id_interaction_claim,
            memory_id_to_value: memory_id_to_value_interaction_claim,
            range_checks: range_checks_interaction_claim,
            verify_bitwise_xor_4: verify_bitwise_xor_4_interaction_claim,
            verify_bitwise_xor_7: verify_bitwise_xor_7_interaction_claim,
            verify_bitwise_xor_8: verify_bitwise_xor_8_interaction_claim,
            verify_bitwise_xor_8_b: verify_bitwise_xor_8_b_interaction_claim,
            verify_bitwise_xor_9: verify_bitwise_xor_9_interaction_claim,
        }
    }
}
