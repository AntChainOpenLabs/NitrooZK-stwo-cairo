#![allow(unused_parens)]
use cairo_air::components::mul_opcode_small::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda, range_check_11_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big, verify_instruction, range_check_11};
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 37;
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
        range_check_11_cuda_state: &range_check_11_cuda::CudaClaimGenerator,
        verify_instruction_cuda_state: &verify_instruction_cuda::CudaClaimGenerator,
        // Also pass SIMD generators for multiplicity tracking (needed for final memory traces)
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        verify_instruction_simd_state: &verify_instruction::ClaimGenerator,
        range_check_11_simd_state: &range_check_11::ClaimGenerator,
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
        // range_check_11 has 3 lookups, each with 1 field per row
        for (i, input_arr) in sub_component_inputs.range_check_11.iter().enumerate() {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for val in cpu_data.iter().take(padded_size) {
                range_check_11_simd_state.add_input(&[*val]);
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
    pub range_check_11: [range_check_11_cuda::CudaPackedInputType; 3],
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
                range_check_11_0: init_lookup_array!(log_size),
                range_check_11_1: init_lookup_array!(log_size),
                range_check_11_2: init_lookup_array!(log_size),
                verify_instruction_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_instruction: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                range_check_11: init_subcomponent_basefield_array!(log_size),
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
    let lookup_range_check_11_0 = collect_lookup_ptrs!(lookup_data, range_check_11_0);
    let lookup_range_check_11_1 = collect_lookup_ptrs!(lookup_data, range_check_11_1);
    let lookup_range_check_11_2 = collect_lookup_ptrs!(lookup_data, range_check_11_2);
    let lookup_verify_instruction_0 = collect_lookup_ptrs!(lookup_data, verify_instruction_0);

    let sub_component_inputs_verify_instruction_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_instruction);
    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_range_check_11_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_11);

    let opcodes_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();
    unsafe {
        bindings_airs::generate_mul_opcode_small_traces(
            traces_vec.as_ptr(),

            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            lookup_opcodes_0.as_ptr(),
            lookup_opcodes_1.as_ptr(),
            lookup_range_check_11_0.as_ptr(),
            lookup_range_check_11_1.as_ptr(),
            lookup_range_check_11_2.as_ptr(),
            lookup_verify_instruction_0.as_ptr(),

            sub_component_inputs_verify_instruction_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            sub_component_inputs_range_check_11_vec.as_ptr(),

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
    range_check_11_0: [BaseFieldVec; 1],
    range_check_11_1: [BaseFieldVec; 1],
    range_check_11_2: [BaseFieldVec; 1],
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
        range_check_11: &relations::RangeCheck_11,
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
        let lookup_range_check_11_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_11_0);
        let lookup_range_check_11_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_11_1);
        let lookup_range_check_11_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_11_2);
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
            let range_check_11_ptr = range_check_11 as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_mul_opcode_small_interaction_traces(
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                opcodes_ptr,
                verify_instruction_ptr,
                range_check_11_ptr,

                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_opcodes_0_vec.as_ptr(),
                lookup_opcodes_1_vec.as_ptr(),
                lookup_range_check_11_0_vec.as_ptr(),
                lookup_range_check_11_1_vec.as_ptr(),
                lookup_range_check_11_2_vec.as_ptr(),
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
    use crate::witness::components_cuda::mul_opcode_small_cuda;
    use crate::witness::components_cuda::memory_address_to_id_cuda;
    use crate::witness::components_cuda::memory_id_to_big_cuda;
    use crate::witness::components_cuda::verify_instruction_cuda;
    use crate::witness::components_cuda::range_check_11_cuda;
    use cairo_air::relations;

    use stwo_constraint_framework::TraceLocationAllocator;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::mul_opcode_small::Eval;
    use crate::witness::components::{memory_id_to_big, memory_address_to_id, range_check_11};
    use stwo::core::fields::m31::M31;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
    use crate::witness::components::mul_opcode_small;
    use crate::witness::components::verify_instruction;
    use cairo_lang_casm::casm;
    use crate::test_utils::input_from_plain_casm;
    use cairo_air::components::mul_opcode_small::Component;
    use itertools::Itertools;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::prover::backend::Column;
    use stwo::stwo_cuda::bindings::CudaSecureField;
    use stwo::core::fields::m31::BaseField;

    #[test]
    fn test_mul_opcode_small_compare_cpu_cuda_traces() {
        let instructions = casm! {
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] =  262145, ap++;
            [ap] =  [ap-1]*262143, ap++;
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] = [ap-1], ap++;
            [ap] = [ap-1] * [ap-2], ap++;
            [ap] = [ap-2]*2147483647, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        if input_state.casm_states_by_opcode.mul_opcode_small.is_empty() {
            println!("Warning: No mul_opcode_small states generated. Test will be skipped.");
            return;
        }

        // Clone inputs for later use
        let mul_small_inputs = input_state.casm_states_by_opcode.mul_opcode_small.clone();

        // Create CPU and CUDA generators with same inputs
        let mul_gen_cpu = mul_opcode_small::ClaimGenerator::new(mul_small_inputs.clone());
        let mul_gen_cuda = mul_opcode_small_cuda::CudaClaimGenerator::new(mul_small_inputs.clone());

        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_trace_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());
        let range_check_11_gen = range_check_11::ClaimGenerator::new();

        let mut memory_address_to_id_cuda_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);
        let verify_instruction_cuda_generator = verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());
        let range_check_11_cuda_generator = range_check_11_cuda::CudaClaimGenerator::new_rc_11();

        // Yield public memory for both
        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_trace_generator.get_id(addr);
            memory_address_to_id_trace_generator.add_input(&addr);
            memory_id_to_big_trace_generator.add_input(&id);
            let id_cuda = memory_address_to_id_cuda_generator.get_id(addr);
            memory_address_to_id_cuda_generator.add_cuda_input(&addr);
            memory_id_to_big_cuda_generator.add_cuda_input(&id_cuda);
        }

        // Generate CPU trace
        let mut mock_commitment_scheme_cpu = MockCommitmentScheme::default();
        let preprocessed_trace_cpu = testing_preprocessed_tree(4);
        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        mock_tree_builder_cpu.extend_evals(preprocessed_trace_cpu.gen_trace());
        mock_tree_builder_cpu.finalize_interaction();

        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        let (cpu_claim, _cpu_interaction_gen) = mul_gen_cpu.write_trace(
            &mut mock_tree_builder_cpu,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_11_gen,
            &verify_instruction_trace_generator,
        );
        mock_tree_builder_cpu.finalize_interaction();
        let cpu_trace = mock_commitment_scheme_cpu.trace_domain_evaluations();

        // Generate CUDA trace
        let mut mock_commitment_scheme_cuda = MockCommitmentScheme::default();
        let preprocessed_trace_cuda = testing_preprocessed_tree(4);
        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        mock_tree_builder_cuda.extend_evals(preprocessed_trace_cuda.gen_trace());
        mock_tree_builder_cuda.finalize_interaction();

        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        let (cuda_claim, _cuda_interaction_gen) = mul_gen_cuda.write_trace(
            &mut mock_tree_builder_cuda,
            &mut memory_address_to_id_cuda_generator,
            &memory_id_to_big_cuda_generator,
            &range_check_11_cuda_generator,
            &verify_instruction_cuda_generator,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &verify_instruction_trace_generator,
            &range_check_11_gen,
        );
        mock_tree_builder_cuda.finalize_interaction();
        let cuda_trace = mock_commitment_scheme_cuda.trace_domain_evaluations();

        println!("CPU log_size: {:?}, CUDA log_size: {:?}", cpu_claim.log_size, cuda_claim.log_size);
        assert_eq!(cpu_claim.log_size, cuda_claim.log_size);

        let trace_size = 1usize << cpu_claim.log_size;

        // Compare base trace columns
        let cpu_base_trace = &cpu_trace[1];
        let cuda_base_trace = &cuda_trace[1];

        println!("CPU base trace columns: {:?}", cpu_base_trace.len());
        println!("CUDA base trace columns: {:?}", cuda_base_trace.len());

        for col in 0..std::cmp::min(cpu_base_trace.len(), cuda_base_trace.len()) {
            let cpu_col = &cpu_base_trace[col];
            let cuda_col = &cuda_base_trace[col];
            let cpu_vals: Vec<M31> = cpu_col.to_cpu().to_vec();
            let cuda_vals: Vec<M31> = cuda_col.to_cpu().to_vec();

            let mut diff = false;
            for row in 0..std::cmp::min(trace_size, 8) {
                if cpu_vals[row] != cuda_vals[row] {
                    diff = true;
                    println!("Column {:?} row {:?}: CPU={:?}, CUDA={:?}", col, row, cpu_vals[row], cuda_vals[row]);
                }
            }
            if !diff {
                println!("Column {:?}: OK (first 8 rows match)", col);
            }
        }

        // Now compare interaction traces
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_11_relation = relations::RangeCheck_11::dummy();

        // Regenerate CPU trace with interaction
        let mul_gen_cpu2 = mul_opcode_small::ClaimGenerator::new(mul_small_inputs.clone());
        let mut memory_address_to_id_cuda_generator2 = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator2 = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_address_to_id_trace_generator2 = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator2 = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_trace_generator2 = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());
        let range_check_11_gen2 = range_check_11::ClaimGenerator::new();
        let verify_instruction_cuda_generator2 = verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());
        let range_check_11_cuda_generator2 = range_check_11_cuda::CudaClaimGenerator::new_rc_11();

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_trace_generator2.get_id(addr);
            memory_address_to_id_trace_generator2.add_input(&addr);
            memory_id_to_big_trace_generator2.add_input(&id);
            let id_cuda = memory_address_to_id_cuda_generator2.get_id(addr);
            memory_address_to_id_cuda_generator2.add_cuda_input(&addr);
            memory_id_to_big_cuda_generator2.add_cuda_input(&id_cuda);
        }

        let mul_gen_cuda2 = mul_opcode_small_cuda::CudaClaimGenerator::new(mul_small_inputs.clone());

        // CPU interaction trace
        let mut mock_commitment_scheme_cpu2 = MockCommitmentScheme::default();
        let preprocessed_trace_cpu2 = testing_preprocessed_tree(4);
        let mut mock_tree_builder_cpu2 = mock_commitment_scheme_cpu2.tree_builder();
        mock_tree_builder_cpu2.extend_evals(preprocessed_trace_cpu2.gen_trace());
        mock_tree_builder_cpu2.finalize_interaction();
        let mut mock_tree_builder_cpu2 = mock_commitment_scheme_cpu2.tree_builder();
        let (cpu_claim2, cpu_interaction_gen2) = mul_gen_cpu2.write_trace(
            &mut mock_tree_builder_cpu2,
            &memory_address_to_id_trace_generator2,
            &memory_id_to_big_trace_generator2,
            &range_check_11_gen2,
            &verify_instruction_trace_generator2,
        );
        mock_tree_builder_cpu2.finalize_interaction();
        let mut mock_tree_builder_cpu2 = mock_commitment_scheme_cpu2.tree_builder();
        let cpu_interaction_claim2 = cpu_interaction_gen2.write_interaction_trace(
            &mut mock_tree_builder_cpu2,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_11_relation,
            &opcodes_relation,
        );
        mock_tree_builder_cpu2.finalize_interaction();
        let cpu_trace2 = mock_commitment_scheme_cpu2.trace_domain_evaluations();

        // CUDA interaction trace
        let mut mock_commitment_scheme_cuda2 = MockCommitmentScheme::default();
        let preprocessed_trace_cuda2 = testing_preprocessed_tree(4);
        let mut mock_tree_builder_cuda2 = mock_commitment_scheme_cuda2.tree_builder();
        mock_tree_builder_cuda2.extend_evals(preprocessed_trace_cuda2.gen_trace());
        mock_tree_builder_cuda2.finalize_interaction();
        let mut mock_tree_builder_cuda2 = mock_commitment_scheme_cuda2.tree_builder();
        let (cuda_claim2, cuda_interaction_gen2) = mul_gen_cuda2.write_trace(
            &mut mock_tree_builder_cuda2,
            &mut memory_address_to_id_cuda_generator2,
            &memory_id_to_big_cuda_generator2,
            &range_check_11_cuda_generator2,
            &verify_instruction_cuda_generator2,
            &memory_address_to_id_trace_generator2,
            &memory_id_to_big_trace_generator2,
            &verify_instruction_trace_generator2,
            &range_check_11_gen2,
        );
        mock_tree_builder_cuda2.finalize_interaction();
        let mut mock_tree_builder_cuda2 = mock_commitment_scheme_cuda2.tree_builder();
        let cuda_interaction_claim2 = cuda_interaction_gen2.write_interaction_trace(
            &mut mock_tree_builder_cuda2,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_11_relation,
            &opcodes_relation,
        );
        mock_tree_builder_cuda2.finalize_interaction();
        let cuda_trace2 = mock_commitment_scheme_cuda2.trace_domain_evaluations();

        println!("\nCPU claimed_sum: {:?}", cpu_interaction_claim2.claimed_sum);
        println!("CUDA claimed_sum: {:?}", cuda_interaction_claim2.claimed_sum);

        // Compare interaction trace columns
        if cpu_trace2.len() > 2 && cuda_trace2.len() > 2 {
            let cpu_interaction_trace = &cpu_trace2[2];
            let cuda_interaction_trace = &cuda_trace2[2];

            println!("\nCPU interaction trace columns: {:?}", cpu_interaction_trace.len());
            println!("CUDA interaction trace columns: {:?}", cuda_interaction_trace.len());

            for col in 0..std::cmp::min(cpu_interaction_trace.len(), cuda_interaction_trace.len()) {
                let cpu_col = &cpu_interaction_trace[col];
                let cuda_col = &cuda_interaction_trace[col];
                let cpu_vals: Vec<M31> = cpu_col.to_cpu().to_vec();
                let cuda_vals: Vec<M31> = cuda_col.to_cpu().to_vec();

                let mut diff = false;
                for row in 0..std::cmp::min(trace_size, 8) {
                    if cpu_vals[row] != cuda_vals[row] {
                        diff = true;
                        println!("Interaction Column {:?} row {:?}: CPU={:?}, CUDA={:?}", col, row, cpu_vals[row], cuda_vals[row]);
                    }
                }
                if !diff {
                    println!("Interaction Column {:?}: OK (first 8 rows match)", col);
                }
            }
        }
    }

    #[test]
    fn test_mul_opcode_small_cpu_ref() {
        let instructions = casm! {
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] =  262145, ap++;
            [ap] =  [ap-1]*262143, ap++;
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] = [ap-1], ap++;
            [ap] = [ap-1] * [ap-2], ap++;
            [ap] = [ap-2]*2147483647, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode_small states
        if input_state.casm_states_by_opcode.mul_opcode_small.is_empty() {
            println!("Warning: No mul_opcode_small states generated. Test will be skipped.");
            return;
        }

        let mul_gen = mul_opcode_small::ClaimGenerator::new(input_state.casm_states_by_opcode.mul_opcode_small);

        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);

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

        let verify_instruction_trace_generator =
            verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_11_relation = relations::RangeCheck_11::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check_11_gen = range_check_11::ClaimGenerator::new();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
                    &mut mock_tree_builder,
                    &memory_address_to_id_trace_generator,
                    &memory_id_to_big_trace_generator,
                    &range_check_11_gen,
                    &verify_instruction_trace_generator,
                );

        mock_tree_builder.finalize_interaction();

        println!("mul_opcode_small_claim log_size: {:?}", mul_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_11_relation,
                    &opcodes_relation,
                 );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("mul_opcode_small_interaction_claim.claimed_sum: {:?}", mul_interaction_claim.claimed_sum);

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_opcode_small"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            mul_interaction_claim.claimed_sum,
        );

        assert_component(&mul_component, &trace)
    }

    #[test]
    fn test_mul_opcode_small_trace_gen_by_cpu_and_verify_by_cuda() {
        let instructions = casm! {
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] =  262145, ap++;
            [ap] =  [ap-1]*262143, ap++;
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] = [ap-1], ap++;
            [ap] = [ap-1] * [ap-2], ap++;
            [ap] = [ap-2]*2147483647, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode_small states
        if input_state.casm_states_by_opcode.mul_opcode_small.is_empty() {
            println!("Warning: No mul_opcode_small states generated. Test will be skipped.");
            return;
        }

        let mul_gen = mul_opcode_small::ClaimGenerator::new(
                input_state.casm_states_by_opcode.mul_opcode_small,
            );

        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);

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

        let verify_instruction_trace_generator =
            verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_11_relation = relations::RangeCheck_11::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check_11_gen = range_check_11::ClaimGenerator::new();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
                    &mut mock_tree_builder,
                    &memory_address_to_id_trace_generator,
                    &memory_id_to_big_trace_generator,
                    &range_check_11_gen,
                    &verify_instruction_trace_generator,
                );

        mock_tree_builder.finalize_interaction();

        println!("mul_opcode_small_claims log_size: {:?}", mul_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_11_relation,
                    &opcodes_relation,
                );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        let trace0_vec: Vec<_> = trace[0].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace1_vec: Vec<_> = trace[1].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace2_vec: Vec<_> = trace[2].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();

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

        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_opcode_small"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            mul_interaction_claim.claimed_sum,
        );

        let eval_ptr = &mul_component.eval as *const _ as *mut std::os::raw::c_void;
        unsafe {
            stwo::stwo_cuda::bindings::evaluate_constraint_quotients_on_domain(
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
                mul_claim.log_size as u32,
                mul_claim.log_size as u32,
                mul_component.info.n_constraints as u32,
                mul_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    mul_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << mul_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }
    }

    #[test]
    fn test_mul_opcode_small_trace_gen_by_cuda_and_verify_by_cpu() {
        let instructions = casm! {
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] =  262145, ap++;
            [ap] =  [ap-1]*262143, ap++;
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] = [ap-1], ap++;
            [ap] = [ap-1] * [ap-2], ap++;
            [ap] = [ap-2]*2147483647, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode_small states
        if input_state.casm_states_by_opcode.mul_opcode_small.is_empty() {
            println!("Warning: No mul_opcode_small states generated. Test will be skipped.");
            return;
        }

        let mul_gen = mul_opcode_small_cuda::CudaClaimGenerator::new(input_state.casm_states_by_opcode.mul_opcode_small);

        let mut memory_address_to_id_cuda_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking (required by new API)
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());
        let range_check_11_simd_generator = range_check_11::ClaimGenerator::new();

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
        let range_check_11_cuda_generator = range_check_11_cuda::CudaClaimGenerator::new_rc_11();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_11_relation = relations::RangeCheck_11::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
                &mut mock_tree_builder,
                &mut memory_address_to_id_cuda_generator,
                &memory_id_to_big_cuda_generator,
                &range_check_11_cuda_generator,
                &verify_instruction_cuda_generator,
                &memory_address_to_id_simd_generator,
                &memory_id_to_big_simd_generator,
                &verify_instruction_simd_generator,
                &range_check_11_simd_generator,
            );

        mock_tree_builder.finalize_interaction();

        println!("mul_opcode_small_claim log_size: {:?}", mul_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_11_relation,
                    &opcodes_relation,
                );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("mul_opcode_small_interaction_claim.claimed_sum: {:?}", mul_interaction_claim.claimed_sum);

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_opcode_small"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            mul_interaction_claim.claimed_sum,
        );

        assert_component(&mul_component, &trace)
    }

    #[test]
    fn test_mul_opcode_small_trace_gen_by_cuda_and_verify_by_cuda() {
        let instructions = casm! {
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] =  262145, ap++;
            [ap] =  [ap-1]*262143, ap++;
            // 2^36-1 is the maximal factor value for a small mul.
            [ap] = [ap-1], ap++;
            [ap] = [ap-1] * [ap-2], ap++;
            [ap] = [ap-2]*2147483647, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode_small states
        if input_state.casm_states_by_opcode.mul_opcode_small.is_empty() {
            println!("Warning: No mul_opcode_small states generated. Test will be skipped.");
            return;
        }

        let mul_gen = mul_opcode_small_cuda::CudaClaimGenerator::new(input_state.casm_states_by_opcode.mul_opcode_small);

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
        let range_check_11_cuda_generator = range_check_11_cuda::CudaClaimGenerator::new_rc_11();

        // SIMD generators for multiplicity tracking (required by new API)
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache);
        let range_check_11_simd_generator = range_check_11::ClaimGenerator::new();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_11_relation = relations::RangeCheck_11::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
                &mut mock_tree_builder,
                &mut memory_address_to_id_cuda_generator,
                &memory_id_to_big_cuda_generator,
                &range_check_11_cuda_generator,
                &verify_instruction_cuda_generator,
                &memory_address_to_id_simd_generator,
                &memory_id_to_big_simd_generator,
                &verify_instruction_simd_generator,
                &range_check_11_simd_generator,
            );

        mock_tree_builder.finalize_interaction();

        println!("mul_opcode_small_claim log_size: {:?}", mul_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_11_relation,
                    &opcodes_relation,
                );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("mul_opcode_small_interaction_claim.claimed_sum: {:?}", mul_interaction_claim.claimed_sum);

        let trace0_vec: Vec<_> = trace[0].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace1_vec: Vec<_> = trace[1].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace2_vec: Vec<_> = trace[2].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();

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

        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_opcode_small"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            mul_interaction_claim.claimed_sum,
        );

        let eval_ptr = &mul_component.eval as *const _ as *mut std::os::raw::c_void;
        unsafe {
            stwo::stwo_cuda::bindings::evaluate_constraint_quotients_on_domain(
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
                mul_claim.log_size as u32,
                mul_claim.log_size as u32,
                mul_component.info.n_constraints as u32,
                mul_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    mul_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << mul_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }
    }
}
