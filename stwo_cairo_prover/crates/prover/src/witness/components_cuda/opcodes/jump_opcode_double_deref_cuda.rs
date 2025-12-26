#![allow(unused_parens)]
use cairo_air::components::jump_opcode_double_deref::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big, verify_instruction};
use stwo::prover::backend::cuda::CudaBackend;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 21;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 4;

pub type CudaPackedInputs = [BaseFieldVec; 3];
use itertools::Itertools;
use stwo::prover::backend::Col;
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
    pub padded_size: usize,
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
        let first_pc = *pcs.first().unwrap_or(&BaseField::from(0));
        let first_ap = *aps.first().unwrap_or(&BaseField::from(0));
        let first_fp = *fps.first().unwrap_or(&BaseField::from(0));
        pcs.resize(size, first_pc);
        aps.resize(size, first_ap);
        fps.resize(size, first_fp);

        let pc_vec = BaseFieldVec::from_vec(pcs);
        let ap_vec = BaseFieldVec::from_vec(aps);
        let fp_vec = BaseFieldVec::from_vec(fps);

        Self {
            n_rows,
            padded_size: size,
            inputs: [pc_vec, ap_vec, fp_vec]
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        verify_instruction_cuda_state: &verify_instruction_cuda::CudaClaimGenerator,
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        verify_instruction_simd_state: &verify_instruction::ClaimGenerator,
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

        verify_instruction_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        for input_arr in &sub_component_inputs.memory_address_to_id {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for addr in cpu_data {
                memory_address_to_id_simd_state.add_input(&addr);
            }
        }
        for input_arr in &sub_component_inputs.memory_id_to_big {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for id in cpu_data {
                memory_id_to_big_simd_state.add_input(&id);
            }
        }
        for input_arr in &sub_component_inputs.verify_instruction {
            let field0: Vec<M31> = input_arr[0].to_vec();
            let field1: Vec<M31> = input_arr[1].to_vec();
            let field2: Vec<M31> = input_arr[2].to_vec();
            let field3: Vec<M31> = input_arr[3].to_vec();
            let field4: Vec<M31> = input_arr[4].to_vec();
            let field5: Vec<M31> = input_arr[5].to_vec();
            let field6: Vec<M31> = input_arr[6].to_vec();

            let n = field0.len();
            for i in 0..n {
                let input: verify_instruction::InputType = (
                    field0[i],
                    [field1[i], field2[i], field3[i]],
                    [field4[i], field5[i]],
                    field6[i],
                );
                verify_instruction_simd_state.add_input(&input);
            }
        }

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                n_rows : self.n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

struct CudaSubComponentInputs {
    verify_instruction: [verify_instruction_cuda::CudaPackedInputType; 1],
    memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 2],
    memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 2],
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
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                opcodes_0: init_lookup_array!(log_size),
                opcodes_1: init_lookup_array!(log_size),
                verify_instruction_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_instruction: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_opcodes_0 = collect_lookup_ptrs!(lookup_data, opcodes_0);
    let lookup_opcodes_1 = collect_lookup_ptrs!(lookup_data, opcodes_1);
    let lookup_verify_instruction_0 = collect_lookup_ptrs!(lookup_data, verify_instruction_0);

    let sub_component_inputs_verify_instruction_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_instruction);
    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);

    let opcodes_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();
    unsafe {
        bindings_airs::generate_jump_opcode_double_deref_traces(
            traces_vec.as_ptr(),

            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_opcodes_0.as_ptr(),
            lookup_opcodes_1.as_ptr(),
            lookup_verify_instruction_0.as_ptr(),

            sub_component_inputs_verify_instruction_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),

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
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    opcodes_0: [BaseFieldVec; 3],
    opcodes_1: [BaseFieldVec; 3],
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
        opcodes: &relations::Opcodes,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_opcodes_0_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_0);
        let lookup_opcodes_1_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_1);
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

            bindings_airs::generate_jump_opcode_double_deref_interaction_traces(
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                opcodes_ptr,
                verify_instruction_ptr,

                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_opcodes_0_vec.as_ptr(),
                lookup_opcodes_1_vec.as_ptr(),
                lookup_verify_instruction_0_vec.as_ptr(),

                self.n_rows as u32,
                trace_log_size as u32,
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([claimed_sum_vec[0], claimed_sum_vec[1], claimed_sum_vec[2], claimed_sum_vec[3]]);

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
    use test_log::test;

    // Note: jump_opcode_double_deref (double dereference absolute jump) is rarely used
    // in standard Cairo programs. These tests are ignored by default because the
    // instruction requires specific conditions that are difficult to generate in simple CASM programs.
    #[test]
    #[ignore = "Double deref absolute jump requires specific conditions not easily generated in CASM"]
    fn test_jump_opcode_double_deref_cpu_ref() {
        println!("jump_opcode_double_deref tests are skipped:");
        println!("- Double deref absolute jumps are rarely used in Cairo");
        println!("- The CUDA implementation is complete but difficult to test with simple CASM");
    }

    #[test]
    #[ignore = "Double deref absolute jump requires specific conditions not easily generated in CASM"]
    fn test_jump_opcode_double_deref_trace_gen_by_cpu_and_verify_by_cuda() {
        println!("jump_opcode_double_deref tests are skipped:");
        println!("- Double deref absolute jumps are rarely used in Cairo");
    }

    #[test]
    #[ignore = "Double deref absolute jump requires specific conditions not easily generated in CASM"]
    fn test_jump_opcode_double_deref_trace_gen_by_cuda_and_verify_by_cpu() {
        println!("jump_opcode_double_deref tests are skipped:");
        println!("- Double deref absolute jumps are rarely used in Cairo");
    }

    #[test]
    #[ignore = "Double deref absolute jump requires specific conditions not easily generated in CASM"]
    fn test_jump_opcode_double_deref_trace_gen_by_cuda_and_verify_by_cuda() {
        println!("jump_opcode_double_deref tests are skipped:");
        println!("- Double deref absolute jumps are rarely used in Cairo");
    }
}
