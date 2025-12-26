#![allow(unused_parens)]
use cairo_air::components::qm_31_add_mul_opcode::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda, range_check_4_4_4_4_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big, verify_instruction, range_check_4_4_4_4};
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 73;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 6;

pub type CudaPackedInputs = [BaseFieldVec; 3];
use itertools::Itertools;
use stwo::prover::backend::{Col, Column};
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}
macro_rules! init_subcomponent_basefield_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size)))
    };
}

macro_rules! collect_lookup_ptrs {
    ($lookup_data:expr, $field:ident) => {
        $lookup_data.$field.iter().map(|x| x.device_ptr).collect_vec()
    };
}

macro_rules! collect_sub_input_ptrs {
    ($sub_inputs:expr, $field:ident) => {
        $sub_inputs.$field
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec()
    };
}

macro_rules! collect_input_ptrs {
    ($input:expr) => {
        $input.iter().map(|x| x.device_ptr).collect_vec()
    };
}

pub struct CudaClaimGenerator {
    pub n_rows: usize,
    pub inputs: CudaPackedInputs,
}
impl CudaClaimGenerator {
    pub fn new(inputs: Vec<InputType>) -> Self {
        let n_rows = inputs.len();

        let mut pcs = Vec::with_capacity(n_rows);
        let mut aps = Vec::with_capacity(n_rows);
        let mut fps = Vec::with_capacity(n_rows);

        for input in inputs.clone() {
            pcs.push(BaseField::from(input.pc));
            aps.push(BaseField::from(input.ap));
            fps.push(BaseField::from(input.fp));
        }
        let size = std::cmp::max(inputs.len().next_power_of_two(), N_LANES);
        pcs.resize(size, pcs[0]);
        aps.resize(size, aps[0]);
        fps.resize(size, fps[0]);

        let pc_vec = BaseFieldVec::from_vec(pcs);
        let ap_vec = BaseFieldVec::from_vec(aps);
        let fp_vec = BaseFieldVec::from_vec(fps);

        Self {
            n_rows,
            inputs: [pc_vec, ap_vec, fp_vec]
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_4_4_4_4_cuda_state: &range_check_4_4_4_4_cuda::CudaClaimGenerator,
        verify_instruction_cuda_state: &verify_instruction_cuda::CudaClaimGenerator,
        // Also pass SIMD generators for multiplicity tracking (needed for final memory traces)
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        verify_instruction_simd_state: &verify_instruction::ClaimGenerator,
        range_check_4_4_4_4_simd_state: &range_check_4_4_4_4::ClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let size = self.inputs[0].size;
        let log_size = size.ilog2();
        let packed_inputs = self.inputs;

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            self.n_rows,
            packed_inputs,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
            verify_instruction_cuda_state,
        );

        // Add to CUDA generators (for CUDA-specific needs if any)
        verify_instruction_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        // Copy GPU data to CPU and add to SIMD generators
        let padded_size = 1usize << log_size;
        for input_arr in &sub_component_inputs.memory_address_to_id {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for addr in cpu_data.iter().take(padded_size) {
                memory_address_to_id_simd_state.add_input(addr);
            }
        }
        for input_arr in &sub_component_inputs.memory_id_to_big {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for id in cpu_data.iter().take(padded_size) {
                memory_id_to_big_simd_state.add_input(id);
            }
        }
        // verify_instruction has 7 fields per row: (pc, [3 offsets], [2 flags], imm_val)
        for input_arr in &sub_component_inputs.verify_instruction {
            let field0: Vec<M31> = input_arr[0].to_vec();
            let field1: Vec<M31> = input_arr[1].to_vec();
            let field2: Vec<M31> = input_arr[2].to_vec();
            let field3: Vec<M31> = input_arr[3].to_vec();
            let field4: Vec<M31> = input_arr[4].to_vec();
            let field5: Vec<M31> = input_arr[5].to_vec();
            let field6: Vec<M31> = input_arr[6].to_vec();

            for i in 0..padded_size {
                let input: verify_instruction::InputType = (
                    field0[i],
                    [field1[i], field2[i], field3[i]],
                    [field4[i], field5[i]],
                    field6[i],
                );
                verify_instruction_simd_state.add_input(&input);
            }
        }
        // range_check_4_4_4_4 has 3 lookups, each with 4 fields per row
        for input_arr in sub_component_inputs.range_check_4_4_4_4.iter() {
            let f0: Vec<M31> = input_arr[0].to_vec();
            let f1: Vec<M31> = input_arr[1].to_vec();
            let f2: Vec<M31> = input_arr[2].to_vec();
            let f3: Vec<M31> = input_arr[3].to_vec();
            for i in 0..padded_size {
                range_check_4_4_4_4_simd_state.add_input(&[f0[i], f1[i], f2[i], f3[i]]);
            }
        }

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                n_rows: self.n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

pub struct CudaSubComponentInputs {
    pub verify_instruction: [verify_instruction_cuda::CudaPackedInputType; 1],
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 3],
    pub memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 3],
    pub range_check_4_4_4_4: [range_check_4_4_4_4_cuda::CudaPackedInputType; 3],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    n_rows: usize,
    inputs: CudaPackedInputs,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    verify_instruction_state: &verify_instruction_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let log_size = inputs[0].size.ilog2();
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                opcodes_0:  init_lookup_array!(log_size),
                opcodes_1:  init_lookup_array!(log_size),
                range_check_4_4_4_4_0: init_lookup_array!(log_size),
                range_check_4_4_4_4_1: init_lookup_array!(log_size),
                range_check_4_4_4_4_2: init_lookup_array!(log_size),
                verify_instruction_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_instruction: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                range_check_4_4_4_4: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_opcodes_0 = collect_lookup_ptrs!(lookup_data, opcodes_0);
    let lookup_opcodes_1 = collect_lookup_ptrs!(lookup_data, opcodes_1);
    let lookup_range_check_4_4_4_4_0 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_0);
    let lookup_range_check_4_4_4_4_1 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_1);
    let lookup_range_check_4_4_4_4_2 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_2);
    let lookup_verify_instruction_0 = collect_lookup_ptrs!(lookup_data, verify_instruction_0);

    let sub_component_inputs_verify_instruction_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_instruction);
    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_range_check_4_4_4_4_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_4_4_4_4);

    let opcodes_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();
    unsafe {
        bindings_airs::generate_qm_31_add_mul_opcode_traces(
            traces_vec.as_ptr(),

            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            lookup_opcodes_0.as_ptr(),
            lookup_opcodes_1.as_ptr(),
            lookup_range_check_4_4_4_4_0.as_ptr(),
            lookup_range_check_4_4_4_4_1.as_ptr(),
            lookup_range_check_4_4_4_4_2.as_ptr(),
            lookup_verify_instruction_0.as_ptr(),

            sub_component_inputs_verify_instruction_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            sub_component_inputs_range_check_4_4_4_4_vec.as_ptr(),

            opcodes_input_vec.as_ptr(),

            memory_address_to_id_state.address_to_raw_id.device_ptr,

            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,

            n_rows as u32,
            log_size,
        );
    }
    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    opcodes_0:  [BaseFieldVec; 3],
    opcodes_1:  [BaseFieldVec; 3],
    range_check_4_4_4_4_0: [BaseFieldVec; 4],
    range_check_4_4_4_4_1: [BaseFieldVec; 4],
    range_check_4_4_4_4_2: [BaseFieldVec; 4],
    verify_instruction_0: [BaseFieldVec; 7],
}

pub struct CudaInteractionClaimGenerator {
    n_rows: usize,
    log_size: u32,
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        verify_instruction: &relations::VerifyInstruction,
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        range_check_4_4_4_4: &relations::RangeCheck_4_4_4_4,
        opcodes: &relations::Opcodes,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);

        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);

        let lookup_opcodes_0_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_0);
        let lookup_opcodes_1_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_1);
        let lookup_range_check_4_4_4_4_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_0);
        let lookup_range_check_4_4_4_4_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_1);
        let lookup_range_check_4_4_4_4_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_2);
        let lookup_verify_instruction_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_instruction_0);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;
            let opcodes_ptr = opcodes as *const _ as *mut std::os::raw::c_void;
            let verify_instruction_ptr = verify_instruction as *const _ as *mut std::os::raw::c_void;
            let range_check_4_4_4_4_ptr = range_check_4_4_4_4 as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_qm_31_add_mul_opcode_interaction_traces(
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                opcodes_ptr,
                verify_instruction_ptr,
                range_check_4_4_4_4_ptr,

                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_opcodes_0_vec.as_ptr(),
                lookup_opcodes_1_vec.as_ptr(),
                lookup_range_check_4_4_4_4_0_vec.as_ptr(),
                lookup_range_check_4_4_4_4_1_vec.as_ptr(),
                lookup_range_check_4_4_4_4_2_vec.as_ptr(),
                lookup_verify_instruction_0_vec.as_ptr(),

                self.n_rows as u32,
                trace_log_size as u32,
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum =  SecureField::from_m31_array([claimed_sum_vec[0], claimed_sum_vec[1], claimed_sum_vec[2], claimed_sum_vec[3]]);

        let domain = CanonicCoset::new(trace_log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}

#[cfg(test)]
pub mod tests {
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use test_log::test;

    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use crate::debug_tools::assert_constraints::assert_component;
    use crate::test_utils::prover_input_from_compiled_cairo_program;
    use crate::witness::components::{
        memory_address_to_id, memory_id_to_big, range_check_4_4_4_4, verify_instruction,
    };
    use crate::witness::components_cuda::{
        memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_4_4_4_4_cuda,
        verify_instruction_cuda, qm_31_add_mul_opcode_cuda,
    };
    use cairo_air::components::qm_31_add_mul_opcode::{Component, Eval};
    use cairo_air::relations;
    use itertools::Itertools;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::m31::BaseField;
    use stwo::prover::backend::Column;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::stwo_cuda::bindings::CudaSecureField;
    use stwo::stwo_cuda::bindings;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
    use stwo_constraint_framework::TraceLocationAllocator;

    use crate::witness::components::qm_31_add_mul_opcode;

    /// CPU-only reference test: generate trace and verify constraints on CPU.
    #[test]
    fn test_qm_31_add_mul_opcode_cpu_ref() {
        // Use precompiled test program that exercises all opcode components (including qm_31).
        let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
        let input_state = input.state_transitions;

        if input_state.casm_states_by_opcode.qm_31_add_mul_opcode.is_empty() {
            println!("Warning: No qm_31_add_mul_opcode states found in test input, skipping.");
            return;
        }

        let qm31_gen = qm_31_add_mul_opcode::ClaimGenerator::new(
            input_state.casm_states_by_opcode.qm_31_add_mul_opcode,
        );

        let memory_address_to_id_trace_generator =
            memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator =
            memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_4_4_4_4_trace_generator =
            range_check_4_4_4_4::ClaimGenerator::new();
        let verify_instruction_trace_generator =
            verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (qm31_claim, qm31_interaction_gen) = qm31_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_4_4_4_4_trace_generator,
            &verify_instruction_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let qm31_interaction_claim = qm31_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_4_4_4_4_relation,
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();

        let trace = mock_commitment_scheme.trace_domain_evaluations();

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("qm_31_add_mul_opcode"),
                claim: qm31_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_4_4_4_4_lookup_elements: relations::RangeCheck_4_4_4_4::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            qm31_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace);
    }

    /// Full pipeline test: CPU trace generation + CUDA constraint evaluation.
    #[test]
    fn test_qm_31_add_mul_opcode_trace_gen_by_cpu_and_verify_by_cuda() {
        // Use a small CASM snippet that triggers qm_31_add_mul_opcode via encoded instruction.
        // We rely on the compiled test input to contain such an opcode.
        let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
        let input_state = input.state_transitions;

        if input_state.casm_states_by_opcode.qm_31_add_mul_opcode.is_empty() {
            println!("Warning: No qm_31_add_mul_opcode states found in test input, skipping CUDA test.");
            return;
        }

        let qm31_gen = qm_31_add_mul_opcode::ClaimGenerator::new(
            input_state.casm_states_by_opcode.qm_31_add_mul_opcode,
        );

        let memory_address_to_id_trace_generator =
            memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator =
            memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_4_4_4_4_state = range_check_4_4_4_4::ClaimGenerator::new();
        let verify_instruction_trace_generator =
            verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        // Yield public memory.
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_trace_generator.get_id(addr);
            memory_address_to_id_trace_generator.add_input(&addr);
            memory_id_to_big_trace_generator.add_input(&id);
        }

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (qm31_claim, qm31_interaction_gen) = qm31_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_4_4_4_4_state,
            &verify_instruction_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let qm31_interaction_claim = qm31_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_4_4_4_4_relation,
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();

        let trace = mock_commitment_scheme.trace_domain_evaluations();

        let trace0_vec: Vec<_> = trace[0]
            .clone()
            .into_iter()
            .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
            .collect();
        let trace1_vec: Vec<_> = trace[1]
            .clone()
            .into_iter()
            .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
            .collect();
        let trace2_vec: Vec<_> = trace[2]
            .clone()
            .into_iter()
            .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
            .collect();

        let trace0_evaluations_vec = trace0_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();
        let trace1_evaluations_vec = trace1_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let mut trace2_evaluations_vec = vec![];
        if trace.len() != 2 {
            trace2_evaluations_vec = trace2_vec
                .iter()
                .map(|column_evaluations| column_evaluations.device_ptr)
                .collect_vec();
        }

        let mock_random_coeff_powers =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_gpu_denom_inv =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());

        let mock_accum_col_columns_0 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_accum_col_columns_1 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_accum_col_columns_2 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_accum_col_columns_3 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("qm_31_add_mul_opcode"),
                claim: qm31_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_4_4_4_4_lookup_elements: relations::RangeCheck_4_4_4_4::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            qm31_interaction_claim.claimed_sum,
        );

        let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;
        unsafe {
            bindings::evaluate_constraint_quotients_on_domain(
                mock_accum_col_columns_0.device_ptr,
                mock_accum_col_columns_1.device_ptr,
                mock_accum_col_columns_2.device_ptr,
                mock_accum_col_columns_3.device_ptr,
                trace0_evaluations_vec.as_ptr(),
                trace0_evaluations_vec.len() as u32,
                trace1_evaluations_vec.as_ptr(),
                trace1_evaluations_vec.len() as u32,
                trace2_evaluations_vec.as_ptr(),
                trace2_evaluations_vec.len() as u32,
                mock_random_coeff_powers.device_ptr,
                mock_gpu_denom_inv.device_ptr,
                qm31_claim.log_size as u32,
                qm31_claim.log_size as u32,
                component.info.n_constraints as u32,
                component
                    .info
                    .logup_counts
                    .iter()
                    .map(|(_, &count)| count)
                    .sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    qm31_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << qm31_claim.log_size),
                ),
                false,
                true, // use_assert_evaluator for better debugging
            );
        }
    }

    /// CUDA trace generation + CPU constraint verification.
    #[test]
    fn test_qm_31_add_mul_opcode_trace_gen_by_cuda_and_verify_by_cpu() {
        let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
        let input_state = input.state_transitions;

        if input_state.casm_states_by_opcode.qm_31_add_mul_opcode.is_empty() {
            println!("Warning: No qm_31_add_mul_opcode states found in test input, skipping.");
            return;
        }

        let qm31_gen = qm_31_add_mul_opcode_cuda::CudaClaimGenerator::new(
            input_state.casm_states_by_opcode.qm_31_add_mul_opcode,
        );

        let mut memory_address_to_id_cuda_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking (required by new API)
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());
        let range_check_4_4_4_4_simd_generator = range_check_4_4_4_4::ClaimGenerator::new();

        // Yield public memory.
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_generator.get_id(addr);
            memory_address_to_id_cuda_generator.add_cuda_input(&addr);
            memory_id_to_big_cuda_generator.add_cuda_input(&id);
        }

        let verify_instruction_cuda_generator =
            verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());
        let range_check_4_4_4_4_cuda_generator = range_check_4_4_4_4_cuda::CudaClaimGenerator::new_rc_4_4_4_4();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (qm31_claim, qm31_interaction_gen) = qm31_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_generator,
            &memory_id_to_big_cuda_generator,
            &range_check_4_4_4_4_cuda_generator,
            &verify_instruction_cuda_generator,
            &memory_address_to_id_simd_generator,
            &memory_id_to_big_simd_generator,
            &verify_instruction_simd_generator,
            &range_check_4_4_4_4_simd_generator,
        );
        mock_tree_builder.finalize_interaction();

        println!("qm_31_add_mul_opcode claim log_size: {:?}", qm31_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let qm31_interaction_claim = qm31_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_4_4_4_4_relation,
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("qm_31_add_mul_opcode_interaction_claim.claimed_sum: {:?}", qm31_interaction_claim.claimed_sum);

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("qm_31_add_mul_opcode"),
                claim: qm31_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_4_4_4_4_lookup_elements: relations::RangeCheck_4_4_4_4::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            qm31_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace);
    }

    /// CUDA trace generation + CUDA constraint verification.
    #[test]
    fn test_qm_31_add_mul_opcode_trace_gen_by_cuda_and_verify_by_cuda() {
        let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
        let input_state = input.state_transitions;

        if input_state.casm_states_by_opcode.qm_31_add_mul_opcode.is_empty() {
            println!("Warning: No qm_31_add_mul_opcode states found in test input, skipping.");
            return;
        }

        let qm31_gen = qm_31_add_mul_opcode_cuda::CudaClaimGenerator::new(
            input_state.casm_states_by_opcode.qm_31_add_mul_opcode,
        );

        let mut memory_address_to_id_cuda_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // Yield public memory.
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_generator.get_id(addr);
            memory_address_to_id_cuda_generator.add_cuda_input(&addr);
            memory_id_to_big_cuda_generator.add_cuda_input(&id);
        }

        let verify_instruction_cuda_generator =
            verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());
        let range_check_4_4_4_4_cuda_generator = range_check_4_4_4_4_cuda::CudaClaimGenerator::new_rc_4_4_4_4();

        // SIMD generators for multiplicity tracking (required by new API)
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache);
        let range_check_4_4_4_4_simd_generator = range_check_4_4_4_4::ClaimGenerator::new();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (qm31_claim, qm31_interaction_gen) = qm31_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_generator,
            &memory_id_to_big_cuda_generator,
            &range_check_4_4_4_4_cuda_generator,
            &verify_instruction_cuda_generator,
            &memory_address_to_id_simd_generator,
            &memory_id_to_big_simd_generator,
            &verify_instruction_simd_generator,
            &range_check_4_4_4_4_simd_generator,
        );
        mock_tree_builder.finalize_interaction();

        println!("qm_31_add_mul_opcode claim log_size: {:?}", qm31_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let qm31_interaction_claim = qm31_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_4_4_4_4_relation,
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("qm_31_add_mul_opcode_interaction_claim.claimed_sum: {:?}", qm31_interaction_claim.claimed_sum);

        let trace0_vec: Vec<_> = trace[0]
            .clone()
            .into_iter()
            .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
            .collect();
        let trace1_vec: Vec<_> = trace[1]
            .clone()
            .into_iter()
            .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
            .collect();
        let trace2_vec: Vec<_> = trace[2]
            .clone()
            .into_iter()
            .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
            .collect();

        let trace0_evaluations_vec = trace0_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();
        let trace1_evaluations_vec = trace1_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let mut trace2_evaluations_vec = vec![];
        if trace.len() != 2 {
            trace2_evaluations_vec = trace2_vec
                .iter()
                .map(|column_evaluations| column_evaluations.device_ptr)
                .collect_vec();
        }

        let mock_random_coeff_powers =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_gpu_denom_inv =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());

        let mock_accum_col_columns_0 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_accum_col_columns_1 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_accum_col_columns_2 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());
        let mock_accum_col_columns_3 =
            BaseFieldVec::from_vec([M31::from_u32_unchecked(0); 100].to_vec());

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("qm_31_add_mul_opcode"),
                claim: qm31_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_4_4_4_4_lookup_elements: relations::RangeCheck_4_4_4_4::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            qm31_interaction_claim.claimed_sum,
        );

        let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;
        unsafe {
            bindings::evaluate_constraint_quotients_on_domain(
                mock_accum_col_columns_0.device_ptr,
                mock_accum_col_columns_1.device_ptr,
                mock_accum_col_columns_2.device_ptr,
                mock_accum_col_columns_3.device_ptr,
                trace0_evaluations_vec.as_ptr(),
                trace0_evaluations_vec.len() as u32,
                trace1_evaluations_vec.as_ptr(),
                trace1_evaluations_vec.len() as u32,
                trace2_evaluations_vec.as_ptr(),
                trace2_evaluations_vec.len() as u32,
                mock_random_coeff_powers.device_ptr,
                mock_gpu_denom_inv.device_ptr,
                qm31_claim.log_size as u32,
                qm31_claim.log_size as u32,
                component.info.n_constraints as u32,
                component
                    .info
                    .logup_counts
                    .iter()
                    .map(|(_, &count)| count)
                    .sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    qm31_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << qm31_claim.log_size),
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }
    }
}
