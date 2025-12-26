#![allow(unused_parens)]
use cairo_air::components::mul_opcode::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda};
use crate::witness::components::{
    memory_address_to_id, memory_id_to_big, verify_instruction,
    range_check_19, range_check_19_b, range_check_19_c, range_check_19_d,
    range_check_19_e, range_check_19_f, range_check_19_g, range_check_19_h,
};
use stwo::prover::backend::cuda::CudaBackend;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 130;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 19;

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

    #[allow(clippy::too_many_arguments)]
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        verify_instruction_cuda_state: &verify_instruction_cuda::CudaClaimGenerator,
        // SIMD generators for multiplicity tracking
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        verify_instruction_simd_state: &verify_instruction::ClaimGenerator,
        // Range check generators
        range_check_19_state: &range_check_19::ClaimGenerator,
        range_check_19_b_state: &range_check_19_b::ClaimGenerator,
        range_check_19_c_state: &range_check_19_c::ClaimGenerator,
        range_check_19_d_state: &range_check_19_d::ClaimGenerator,
        range_check_19_e_state: &range_check_19_e::ClaimGenerator,
        range_check_19_f_state: &range_check_19_f::ClaimGenerator,
        range_check_19_g_state: &range_check_19_g::ClaimGenerator,
        range_check_19_h_state: &range_check_19_h::ClaimGenerator,
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

        // Add to CUDA generators for multiplicity accumulation
        verify_instruction_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        // Use padded_size to match SIMD behavior - inputs are padded with first row duplicates
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
        // verify_instruction has 7 fields per row
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

        // Add range check inputs from lookup data
        // range_check_19: 4 lookups × 1 field
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_0,
                1 => &lookup_data.range_check_19_1,
                2 => &lookup_data.range_check_19_2,
                3 => &lookup_data.range_check_19_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_b: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_b_0,
                1 => &lookup_data.range_check_19_b_1,
                2 => &lookup_data.range_check_19_b_2,
                3 => &lookup_data.range_check_19_b_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_b_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_c: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_c_0,
                1 => &lookup_data.range_check_19_c_1,
                2 => &lookup_data.range_check_19_c_2,
                3 => &lookup_data.range_check_19_c_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_c_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_d: 3 lookups
        for i in 0..3 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_d_0,
                1 => &lookup_data.range_check_19_d_1,
                2 => &lookup_data.range_check_19_d_2,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_d_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_e: 3 lookups
        for i in 0..3 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_e_0,
                1 => &lookup_data.range_check_19_e_1,
                2 => &lookup_data.range_check_19_e_2,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_e_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_f: 3 lookups
        for i in 0..3 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_f_0,
                1 => &lookup_data.range_check_19_f_1,
                2 => &lookup_data.range_check_19_f_2,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_f_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_g: 3 lookups
        for i in 0..3 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_g_0,
                1 => &lookup_data.range_check_19_g_1,
                2 => &lookup_data.range_check_19_g_2,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_g_state.add_input(&[field0[j]]);
            }
        }

        // range_check_19_h: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_h_0,
                1 => &lookup_data.range_check_19_h_1,
                2 => &lookup_data.range_check_19_h_2,
                3 => &lookup_data.range_check_19_h_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..size {
                range_check_19_h_state.add_input(&[field0[j]]);
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

struct CudaSubComponentInputs {
    verify_instruction: [verify_instruction_cuda::CudaPackedInputType; 1],
    memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 3],
    memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 3],
    range_check_19: [[BaseFieldVec; 1]; 4],
    range_check_19_b: [[BaseFieldVec; 1]; 4],
    range_check_19_c: [[BaseFieldVec; 1]; 4],
    range_check_19_d: [[BaseFieldVec; 1]; 3],
    range_check_19_e: [[BaseFieldVec; 1]; 3],
    range_check_19_f: [[BaseFieldVec; 1]; 3],
    range_check_19_g: [[BaseFieldVec; 1]; 3],
    range_check_19_h: [[BaseFieldVec; 1]; 4],
}

#[allow(unused_variables)]
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
                // memory_address_to_id: 3 lookups × 2 fields
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                // memory_id_to_big: 3 lookups × 29 fields
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                // opcodes: 2 lookups × 3 fields
                opcodes_0: init_lookup_array!(log_size),
                opcodes_1: init_lookup_array!(log_size),
                // range_check_19: 4 lookups × 1 field
                range_check_19_0: init_lookup_array!(log_size),
                range_check_19_1: init_lookup_array!(log_size),
                range_check_19_2: init_lookup_array!(log_size),
                range_check_19_3: init_lookup_array!(log_size),
                // range_check_19_b: 4 lookups × 1 field
                range_check_19_b_0: init_lookup_array!(log_size),
                range_check_19_b_1: init_lookup_array!(log_size),
                range_check_19_b_2: init_lookup_array!(log_size),
                range_check_19_b_3: init_lookup_array!(log_size),
                // range_check_19_c: 4 lookups × 1 field
                range_check_19_c_0: init_lookup_array!(log_size),
                range_check_19_c_1: init_lookup_array!(log_size),
                range_check_19_c_2: init_lookup_array!(log_size),
                range_check_19_c_3: init_lookup_array!(log_size),
                // range_check_19_d: 3 lookups × 1 field
                range_check_19_d_0: init_lookup_array!(log_size),
                range_check_19_d_1: init_lookup_array!(log_size),
                range_check_19_d_2: init_lookup_array!(log_size),
                // range_check_19_e: 3 lookups × 1 field
                range_check_19_e_0: init_lookup_array!(log_size),
                range_check_19_e_1: init_lookup_array!(log_size),
                range_check_19_e_2: init_lookup_array!(log_size),
                // range_check_19_f: 3 lookups × 1 field
                range_check_19_f_0: init_lookup_array!(log_size),
                range_check_19_f_1: init_lookup_array!(log_size),
                range_check_19_f_2: init_lookup_array!(log_size),
                // range_check_19_g: 3 lookups × 1 field
                range_check_19_g_0: init_lookup_array!(log_size),
                range_check_19_g_1: init_lookup_array!(log_size),
                range_check_19_g_2: init_lookup_array!(log_size),
                // range_check_19_h: 4 lookups × 1 field
                range_check_19_h_0: init_lookup_array!(log_size),
                range_check_19_h_1: init_lookup_array!(log_size),
                range_check_19_h_2: init_lookup_array!(log_size),
                range_check_19_h_3: init_lookup_array!(log_size),
                // verify_instruction: 1 lookup × 7 fields
                verify_instruction_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_instruction: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                range_check_19: init_subcomponent_basefield_array!(log_size),
                range_check_19_b: init_subcomponent_basefield_array!(log_size),
                range_check_19_c: init_subcomponent_basefield_array!(log_size),
                range_check_19_d: init_subcomponent_basefield_array!(log_size),
                range_check_19_e: init_subcomponent_basefield_array!(log_size),
                range_check_19_f: init_subcomponent_basefield_array!(log_size),
                range_check_19_g: init_subcomponent_basefield_array!(log_size),
                range_check_19_h: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect all lookup pointers
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_opcodes_0 = collect_lookup_ptrs!(lookup_data, opcodes_0);
    let lookup_opcodes_1 = collect_lookup_ptrs!(lookup_data, opcodes_1);

    let lookup_range_check_19_0 = collect_lookup_ptrs!(lookup_data, range_check_19_0);
    let lookup_range_check_19_1 = collect_lookup_ptrs!(lookup_data, range_check_19_1);
    let lookup_range_check_19_2 = collect_lookup_ptrs!(lookup_data, range_check_19_2);
    let lookup_range_check_19_3 = collect_lookup_ptrs!(lookup_data, range_check_19_3);

    let lookup_range_check_19_b_0 = collect_lookup_ptrs!(lookup_data, range_check_19_b_0);
    let lookup_range_check_19_b_1 = collect_lookup_ptrs!(lookup_data, range_check_19_b_1);
    let lookup_range_check_19_b_2 = collect_lookup_ptrs!(lookup_data, range_check_19_b_2);
    let lookup_range_check_19_b_3 = collect_lookup_ptrs!(lookup_data, range_check_19_b_3);

    let lookup_range_check_19_c_0 = collect_lookup_ptrs!(lookup_data, range_check_19_c_0);
    let lookup_range_check_19_c_1 = collect_lookup_ptrs!(lookup_data, range_check_19_c_1);
    let lookup_range_check_19_c_2 = collect_lookup_ptrs!(lookup_data, range_check_19_c_2);
    let lookup_range_check_19_c_3 = collect_lookup_ptrs!(lookup_data, range_check_19_c_3);

    let lookup_range_check_19_d_0 = collect_lookup_ptrs!(lookup_data, range_check_19_d_0);
    let lookup_range_check_19_d_1 = collect_lookup_ptrs!(lookup_data, range_check_19_d_1);
    let lookup_range_check_19_d_2 = collect_lookup_ptrs!(lookup_data, range_check_19_d_2);

    let lookup_range_check_19_e_0 = collect_lookup_ptrs!(lookup_data, range_check_19_e_0);
    let lookup_range_check_19_e_1 = collect_lookup_ptrs!(lookup_data, range_check_19_e_1);
    let lookup_range_check_19_e_2 = collect_lookup_ptrs!(lookup_data, range_check_19_e_2);

    let lookup_range_check_19_f_0 = collect_lookup_ptrs!(lookup_data, range_check_19_f_0);
    let lookup_range_check_19_f_1 = collect_lookup_ptrs!(lookup_data, range_check_19_f_1);
    let lookup_range_check_19_f_2 = collect_lookup_ptrs!(lookup_data, range_check_19_f_2);

    let lookup_range_check_19_g_0 = collect_lookup_ptrs!(lookup_data, range_check_19_g_0);
    let lookup_range_check_19_g_1 = collect_lookup_ptrs!(lookup_data, range_check_19_g_1);
    let lookup_range_check_19_g_2 = collect_lookup_ptrs!(lookup_data, range_check_19_g_2);

    let lookup_range_check_19_h_0 = collect_lookup_ptrs!(lookup_data, range_check_19_h_0);
    let lookup_range_check_19_h_1 = collect_lookup_ptrs!(lookup_data, range_check_19_h_1);
    let lookup_range_check_19_h_2 = collect_lookup_ptrs!(lookup_data, range_check_19_h_2);
    let lookup_range_check_19_h_3 = collect_lookup_ptrs!(lookup_data, range_check_19_h_3);

    let lookup_verify_instruction_0 = collect_lookup_ptrs!(lookup_data, verify_instruction_0);

    let sub_component_inputs_verify_instruction_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_instruction);
    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_range_check_19_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19);
    let sub_component_inputs_range_check_19_b_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_b);
    let sub_component_inputs_range_check_19_c_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_c);
    let sub_component_inputs_range_check_19_d_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_d);
    let sub_component_inputs_range_check_19_e_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_e);
    let sub_component_inputs_range_check_19_f_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_f);
    let sub_component_inputs_range_check_19_g_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_g);
    let sub_component_inputs_range_check_19_h_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_h);

    let mul_opcode_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_mul_opcode_traces(
            traces_vec.as_ptr(),

            // memory_address_to_id (3 lookups)
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),

            // memory_id_to_big (3 lookups)
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),

            // opcodes (2 lookups)
            lookup_opcodes_0.as_ptr(),
            lookup_opcodes_1.as_ptr(),

            // range_check_19 (4 lookups)
            lookup_range_check_19_0.as_ptr(),
            lookup_range_check_19_1.as_ptr(),
            lookup_range_check_19_2.as_ptr(),
            lookup_range_check_19_3.as_ptr(),

            // range_check_19_b (4 lookups)
            lookup_range_check_19_b_0.as_ptr(),
            lookup_range_check_19_b_1.as_ptr(),
            lookup_range_check_19_b_2.as_ptr(),
            lookup_range_check_19_b_3.as_ptr(),

            // range_check_19_c (4 lookups)
            lookup_range_check_19_c_0.as_ptr(),
            lookup_range_check_19_c_1.as_ptr(),
            lookup_range_check_19_c_2.as_ptr(),
            lookup_range_check_19_c_3.as_ptr(),

            // range_check_19_d (3 lookups)
            lookup_range_check_19_d_0.as_ptr(),
            lookup_range_check_19_d_1.as_ptr(),
            lookup_range_check_19_d_2.as_ptr(),

            // range_check_19_e (3 lookups)
            lookup_range_check_19_e_0.as_ptr(),
            lookup_range_check_19_e_1.as_ptr(),
            lookup_range_check_19_e_2.as_ptr(),

            // range_check_19_f (3 lookups)
            lookup_range_check_19_f_0.as_ptr(),
            lookup_range_check_19_f_1.as_ptr(),
            lookup_range_check_19_f_2.as_ptr(),

            // range_check_19_g (3 lookups)
            lookup_range_check_19_g_0.as_ptr(),
            lookup_range_check_19_g_1.as_ptr(),
            lookup_range_check_19_g_2.as_ptr(),

            // range_check_19_h (4 lookups)
            lookup_range_check_19_h_0.as_ptr(),
            lookup_range_check_19_h_1.as_ptr(),
            lookup_range_check_19_h_2.as_ptr(),
            lookup_range_check_19_h_3.as_ptr(),

            // verify_instruction (1 lookup)
            lookup_verify_instruction_0.as_ptr(),

            // Sub-component inputs
            sub_component_inputs_verify_instruction_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            sub_component_inputs_range_check_19_vec.as_ptr(),
            sub_component_inputs_range_check_19_b_vec.as_ptr(),
            sub_component_inputs_range_check_19_c_vec.as_ptr(),
            sub_component_inputs_range_check_19_d_vec.as_ptr(),
            sub_component_inputs_range_check_19_e_vec.as_ptr(),
            sub_component_inputs_range_check_19_f_vec.as_ptr(),
            sub_component_inputs_range_check_19_g_vec.as_ptr(),
            sub_component_inputs_range_check_19_h_vec.as_ptr(),

            // Opcode inputs
            mul_opcode_input_vec.as_ptr(),

            // Memory lookup tables
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
    // memory_address_to_id: 3 lookups × 2 fields
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    // memory_id_to_big: 3 lookups × 29 fields
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    // opcodes: 2 lookups × 3 fields
    opcodes_0: [BaseFieldVec; 3],
    opcodes_1: [BaseFieldVec; 3],
    // range_check_19: 4 lookups × 1 field
    range_check_19_0: [BaseFieldVec; 1],
    range_check_19_1: [BaseFieldVec; 1],
    range_check_19_2: [BaseFieldVec; 1],
    range_check_19_3: [BaseFieldVec; 1],
    // range_check_19_b: 4 lookups × 1 field
    range_check_19_b_0: [BaseFieldVec; 1],
    range_check_19_b_1: [BaseFieldVec; 1],
    range_check_19_b_2: [BaseFieldVec; 1],
    range_check_19_b_3: [BaseFieldVec; 1],
    // range_check_19_c: 4 lookups × 1 field
    range_check_19_c_0: [BaseFieldVec; 1],
    range_check_19_c_1: [BaseFieldVec; 1],
    range_check_19_c_2: [BaseFieldVec; 1],
    range_check_19_c_3: [BaseFieldVec; 1],
    // range_check_19_d: 3 lookups × 1 field
    range_check_19_d_0: [BaseFieldVec; 1],
    range_check_19_d_1: [BaseFieldVec; 1],
    range_check_19_d_2: [BaseFieldVec; 1],
    // range_check_19_e: 3 lookups × 1 field
    range_check_19_e_0: [BaseFieldVec; 1],
    range_check_19_e_1: [BaseFieldVec; 1],
    range_check_19_e_2: [BaseFieldVec; 1],
    // range_check_19_f: 3 lookups × 1 field
    range_check_19_f_0: [BaseFieldVec; 1],
    range_check_19_f_1: [BaseFieldVec; 1],
    range_check_19_f_2: [BaseFieldVec; 1],
    // range_check_19_g: 3 lookups × 1 field
    range_check_19_g_0: [BaseFieldVec; 1],
    range_check_19_g_1: [BaseFieldVec; 1],
    range_check_19_g_2: [BaseFieldVec; 1],
    // range_check_19_h: 4 lookups × 1 field
    range_check_19_h_0: [BaseFieldVec; 1],
    range_check_19_h_1: [BaseFieldVec; 1],
    range_check_19_h_2: [BaseFieldVec; 1],
    range_check_19_h_3: [BaseFieldVec; 1],
    // verify_instruction: 1 lookup × 7 fields
    verify_instruction_0: [BaseFieldVec; 7],
}

pub struct CudaInteractionClaimGenerator {
    n_rows: usize,
    log_size: u32,
    lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    #[allow(clippy::too_many_arguments)]
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        verify_instruction: &relations::VerifyInstruction,
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        range_check_19_h: &relations::RangeCheck_19_H,
        range_check_19: &relations::RangeCheck_19,
        range_check_19_b: &relations::RangeCheck_19_B,
        range_check_19_c: &relations::RangeCheck_19_C,
        range_check_19_d: &relations::RangeCheck_19_D,
        range_check_19_e: &relations::RangeCheck_19_E,
        range_check_19_f: &relations::RangeCheck_19_F,
        range_check_19_g: &relations::RangeCheck_19_G,
        opcodes: &relations::Opcodes,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect all lookup pointers for interaction trace generation
        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_opcodes_0_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_0);
        let lookup_opcodes_1_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_1);

        let lookup_range_check_19_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_0);
        let lookup_range_check_19_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_1);
        let lookup_range_check_19_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_2);
        let lookup_range_check_19_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_3);

        let lookup_range_check_19_b_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_b_0);
        let lookup_range_check_19_b_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_b_1);
        let lookup_range_check_19_b_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_b_2);
        let lookup_range_check_19_b_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_b_3);

        let lookup_range_check_19_c_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_c_0);
        let lookup_range_check_19_c_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_c_1);
        let lookup_range_check_19_c_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_c_2);
        let lookup_range_check_19_c_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_c_3);

        let lookup_range_check_19_d_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_d_0);
        let lookup_range_check_19_d_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_d_1);
        let lookup_range_check_19_d_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_d_2);

        let lookup_range_check_19_e_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_e_0);
        let lookup_range_check_19_e_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_e_1);
        let lookup_range_check_19_e_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_e_2);

        let lookup_range_check_19_f_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_f_0);
        let lookup_range_check_19_f_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_f_1);
        let lookup_range_check_19_f_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_f_2);

        let lookup_range_check_19_g_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_g_0);
        let lookup_range_check_19_g_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_g_1);
        let lookup_range_check_19_g_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_g_2);

        let lookup_range_check_19_h_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_h_0);
        let lookup_range_check_19_h_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_h_1);
        let lookup_range_check_19_h_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_h_2);
        let lookup_range_check_19_h_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_h_3);

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
            let range_check_19_ptr = range_check_19 as *const _ as *mut std::os::raw::c_void;
            let range_check_19_b_ptr = range_check_19_b as *const _ as *mut std::os::raw::c_void;
            let range_check_19_c_ptr = range_check_19_c as *const _ as *mut std::os::raw::c_void;
            let range_check_19_d_ptr = range_check_19_d as *const _ as *mut std::os::raw::c_void;
            let range_check_19_e_ptr = range_check_19_e as *const _ as *mut std::os::raw::c_void;
            let range_check_19_f_ptr = range_check_19_f as *const _ as *mut std::os::raw::c_void;
            let range_check_19_g_ptr = range_check_19_g as *const _ as *mut std::os::raw::c_void;
            let range_check_19_h_ptr = range_check_19_h as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_mul_opcode_interaction_traces(
                // Relation pointers (12 relations)
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                opcodes_ptr,
                verify_instruction_ptr,
                range_check_19_ptr,
                range_check_19_b_ptr,
                range_check_19_c_ptr,
                range_check_19_d_ptr,
                range_check_19_e_ptr,
                range_check_19_f_ptr,
                range_check_19_g_ptr,
                range_check_19_h_ptr,

                // All lookup data
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_opcodes_0_vec.as_ptr(),
                lookup_opcodes_1_vec.as_ptr(),
                lookup_range_check_19_0_vec.as_ptr(),
                lookup_range_check_19_1_vec.as_ptr(),
                lookup_range_check_19_2_vec.as_ptr(),
                lookup_range_check_19_3_vec.as_ptr(),
                lookup_range_check_19_b_0_vec.as_ptr(),
                lookup_range_check_19_b_1_vec.as_ptr(),
                lookup_range_check_19_b_2_vec.as_ptr(),
                lookup_range_check_19_b_3_vec.as_ptr(),
                lookup_range_check_19_c_0_vec.as_ptr(),
                lookup_range_check_19_c_1_vec.as_ptr(),
                lookup_range_check_19_c_2_vec.as_ptr(),
                lookup_range_check_19_c_3_vec.as_ptr(),
                lookup_range_check_19_d_0_vec.as_ptr(),
                lookup_range_check_19_d_1_vec.as_ptr(),
                lookup_range_check_19_d_2_vec.as_ptr(),
                lookup_range_check_19_e_0_vec.as_ptr(),
                lookup_range_check_19_e_1_vec.as_ptr(),
                lookup_range_check_19_e_2_vec.as_ptr(),
                lookup_range_check_19_f_0_vec.as_ptr(),
                lookup_range_check_19_f_1_vec.as_ptr(),
                lookup_range_check_19_f_2_vec.as_ptr(),
                lookup_range_check_19_g_0_vec.as_ptr(),
                lookup_range_check_19_g_1_vec.as_ptr(),
                lookup_range_check_19_g_2_vec.as_ptr(),
                lookup_range_check_19_h_0_vec.as_ptr(),
                lookup_range_check_19_h_1_vec.as_ptr(),
                lookup_range_check_19_h_2_vec.as_ptr(),
                lookup_range_check_19_h_3_vec.as_ptr(),
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
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use test_log::test;

    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use crate::witness::components::{memory_address_to_id, memory_id_to_big, verify_instruction};
    use cairo_air::relations;

    use stwo_constraint_framework::TraceLocationAllocator;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::mul_opcode::Eval;
    use stwo::core::fields::m31::M31;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
    use crate::witness::components::mul_opcode;
    use cairo_lang_casm::casm;
    use crate::test_utils::input_from_plain_casm;
    use cairo_air::components::mul_opcode::Component;
    use itertools::Itertools;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::prover::backend::Column;
    use stwo::stwo_cuda::bindings::CudaSecureField;
    use stwo::core::fields::m31::BaseField;

    use crate::witness::components_cuda::{
        memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda,
    };
    use super::CudaClaimGenerator as MulOpcodeCudaClaimGenerator;

    #[test]
    fn test_mul_opcode_trace_gen_by_cuda_and_verify_by_cpu() {
        // Create mul opcode test cases
        let instructions = casm! {
            [ap] =  8, ap++;
            // 2^36 is the minimal factor value for a big mul.
            [ap] = 262144, ap++;
            [ap] = [ap-1] * 262144, ap++;
            [ap] = [ap-1] * [ap-3], ap++;
            [ap] = [ap-2]* 2, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode states
        assert!(!input_state.casm_states_by_opcode.mul_opcode.is_empty());

        // Create CUDA claim generator for mul_opcode
        let mul_gen = MulOpcodeCudaClaimGenerator::new(
            input_state.casm_states_by_opcode.mul_opcode,
        );

        // Create CUDA claim generators for memory components
        let mut memory_address_to_id_cuda_gen = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_gen = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_gen.get_id(addr);
            memory_address_to_id_cuda_gen.add_cuda_input(&addr);
            memory_id_to_big_cuda_gen.add_cuda_input(&id);
        }

        let verify_instruction_cuda_gen =
            verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_gen = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_gen = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_gen = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        // Range check SIMD generators
        let range_check_19_gen = crate::witness::components::range_check_19::ClaimGenerator::new();
        let range_check_19_b_gen = crate::witness::components::range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_gen = crate::witness::components::range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_gen = crate::witness::components::range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_gen = crate::witness::components::range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_gen = crate::witness::components::range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_gen = crate::witness::components::range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_gen = crate::witness::components::range_check_19_h::ClaimGenerator::new();

        // Relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let range_check_19_h_relation = relations::RangeCheck_19_H::dummy();
        let range_check_19_relation = relations::RangeCheck_19::dummy();
        let range_check_19_b_relation = relations::RangeCheck_19_B::dummy();
        let range_check_19_c_relation = relations::RangeCheck_19_C::dummy();
        let range_check_19_d_relation = relations::RangeCheck_19_D::dummy();
        let range_check_19_e_relation = relations::RangeCheck_19_E::dummy();
        let range_check_19_f_relation = relations::RangeCheck_19_F::dummy();
        let range_check_19_g_relation = relations::RangeCheck_19_G::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace - using CUDA generator
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_gen,
            &memory_id_to_big_cuda_gen,
            &verify_instruction_cuda_gen,
            &memory_address_to_id_simd_gen,
            &memory_id_to_big_simd_gen,
            &verify_instruction_simd_gen,
            &range_check_19_gen,
            &range_check_19_b_gen,
            &range_check_19_c_gen,
            &range_check_19_d_gen,
            &range_check_19_e_gen,
            &range_check_19_f_gen,
            &range_check_19_g_gen,
            &range_check_19_h_gen,
        );

        mock_tree_builder.finalize_interaction();

        // Interaction trace - using CUDA generator
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_19_h_relation,
            &range_check_19_relation,
            &range_check_19_b_relation,
            &range_check_19_c_relation,
            &range_check_19_d_relation,
            &range_check_19_e_relation,
            &range_check_19_f_relation,
            &range_check_19_g_relation,
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // Verify with CPU assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_opcode"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            mul_interaction_claim.claimed_sum,
        );

        assert_component(&mul_component, &trace)
    }

    #[test]
    fn test_mul_opcode_cpu_ref() {
        // Create mul opcode test cases
        let instructions = casm! {
            [ap] =  8, ap++;
            // 2^36 is the minimal factor value for a big mul.
            [ap] = 262144, ap++;
            [ap] = [ap-1] * 262144, ap++;
            [ap] = [ap-1] * [ap-3], ap++;
            [ap] = [ap-2]* 2, ap++;
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode states
        assert!(!input_state.casm_states_by_opcode.mul_opcode.is_empty());

        let mul_gen = mul_opcode::ClaimGenerator::new(input_state.casm_states_by_opcode.mul_opcode);

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
        let range_check_19_h_relation = relations::RangeCheck_19_H::dummy();
        let range_check_19_relation = relations::RangeCheck_19::dummy();
        let range_check_19_b_relation = relations::RangeCheck_19_B::dummy();
        let range_check_19_c_relation = relations::RangeCheck_19_C::dummy();
        let range_check_19_d_relation = relations::RangeCheck_19_D::dummy();
        let range_check_19_e_relation = relations::RangeCheck_19_E::dummy();
        let range_check_19_f_relation = relations::RangeCheck_19_F::dummy();
        let range_check_19_g_relation = relations::RangeCheck_19_G::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check_19_gen = crate::witness::components::range_check_19::ClaimGenerator::new();
        let range_check_19_b_gen = crate::witness::components::range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_gen = crate::witness::components::range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_gen = crate::witness::components::range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_gen = crate::witness::components::range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_gen = crate::witness::components::range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_gen = crate::witness::components::range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_gen = crate::witness::components::range_check_19_h::ClaimGenerator::new();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
                    &mut mock_tree_builder,
                    &memory_address_to_id_trace_generator,
                    &memory_id_to_big_trace_generator,
                    &range_check_19_gen,
                    &range_check_19_b_gen,
                    &range_check_19_c_gen,
                    &range_check_19_d_gen,
                    &range_check_19_e_gen,
                    &range_check_19_f_gen,
                    &range_check_19_g_gen,
                    &range_check_19_h_gen,
                    &verify_instruction_trace_generator,
                );

        mock_tree_builder.finalize_interaction();

        println!("mul_opcode_claim log_size: {:?}", mul_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_19_h_relation,
                    &range_check_19_relation,
                    &range_check_19_b_relation,
                    &range_check_19_c_relation,
                    &range_check_19_d_relation,
                    &range_check_19_e_relation,
                    &range_check_19_f_relation,
                    &range_check_19_g_relation,
                    &opcodes_relation,
                 );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("mul_opcode_interaction_claim.claimed_sum: {:?}", mul_interaction_claim.claimed_sum);

        // Debug: dump base trace columns to compare with CUDA
        println!("\n=== CPU Base Trace (first 16 rows) ===");
        // trace[1] is the base trace
        for col_idx in [0, 14, 43, 72, 101, 102, 129].iter() {
            if *col_idx < trace[1].len() {
                let col_data = trace[1][*col_idx].to_cpu();
                println!("col_{}: {:?}", col_idx, &col_data[..std::cmp::min(16, col_data.len())]);
            }
        }

        // Debug: dump ALL interaction trace columns (19 QM31 columns = 76 M31 columns)
        println!("\n=== CPU Interaction Trace - All 19 QM31 columns (first 2 rows) ===");
        // trace[2] is the interaction trace, stored as 76 M31 columns (19 QM31 * 4 M31 each)
        let n_qm31_cols = trace[2].len() / 4;
        println!("Total M31 columns: {}, QM31 columns: {}", trace[2].len(), n_qm31_cols);
        for qm31_idx in 0..n_qm31_cols {
            let c0 = trace[2][qm31_idx * 4 + 0].to_cpu();
            let c1 = trace[2][qm31_idx * 4 + 1].to_cpu();
            let c2 = trace[2][qm31_idx * 4 + 2].to_cpu();
            let c3 = trace[2][qm31_idx * 4 + 3].to_cpu();
            // Print as QM31 format for first 2 rows
            println!("interaction_qm31_col_{}:", qm31_idx);
            for row in 0..std::cmp::min(2, c0.len()) {
                println!("  row[{}]: ({} + {}i) + ({} + {}i)u",
                    row, c0[row].0, c1[row].0, c2[row].0, c3[row].0);
            }
        }

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_opcode"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            mul_interaction_claim.claimed_sum,
        );

        assert_component(&mul_component, &trace)
    }

    #[test]
    fn test_mul_opcode_trace_gen_by_cpu_and_verify_by_cuda() {
        // Create mul opcode test cases - same pattern as test_mul_big
        let instructions = casm! {
            [ap] =  8, ap++;
            // 2^36 is the minimal factor value for a big mul.
            [ap] = 262144, ap++;
            [ap] = [ap-1] * 262144, ap++;      // 262144 * 262144 = 2^36 (mul_opcode_small)
            [ap] = [ap-1] * [ap-3], ap++;      // 2^36 * 8 (mul_opcode)
            [ap] = [ap-2]* 2, ap++;            // 2^36 * 2 (mul_opcode)
            [ap] = 1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have mul_opcode states
        println!("input_state.casm_states_by_opcode: {:?}", input_state.casm_states_by_opcode);
        assert!(!input_state.casm_states_by_opcode.mul_opcode.is_empty());

        let mul_gen = mul_opcode::ClaimGenerator::new(
                input_state.casm_states_by_opcode.mul_opcode,
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
        let range_check_19_h_relation = relations::RangeCheck_19_H::dummy();
        let range_check_19_relation = relations::RangeCheck_19::dummy();
        let range_check_19_b_relation = relations::RangeCheck_19_B::dummy();
        let range_check_19_c_relation = relations::RangeCheck_19_C::dummy();
        let range_check_19_d_relation = relations::RangeCheck_19_D::dummy();
        let range_check_19_e_relation = relations::RangeCheck_19_E::dummy();
        let range_check_19_f_relation = relations::RangeCheck_19_F::dummy();
        let range_check_19_g_relation = relations::RangeCheck_19_G::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check_19_gen = crate::witness::components::range_check_19::ClaimGenerator::new();
        let range_check_19_b_gen = crate::witness::components::range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_gen = crate::witness::components::range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_gen = crate::witness::components::range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_gen = crate::witness::components::range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_gen = crate::witness::components::range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_gen = crate::witness::components::range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_gen = crate::witness::components::range_check_19_h::ClaimGenerator::new();
        let (mul_claim, mul_interaction_gen) = mul_gen.write_trace(
                    &mut mock_tree_builder,
                    &memory_address_to_id_trace_generator,
                    &memory_id_to_big_trace_generator,
                    &range_check_19_gen,
                    &range_check_19_b_gen,
                    &range_check_19_c_gen,
                    &range_check_19_d_gen,
                    &range_check_19_e_gen,
                    &range_check_19_f_gen,
                    &range_check_19_g_gen,
                    &range_check_19_h_gen,
                    &verify_instruction_trace_generator,
                );

        mock_tree_builder.finalize_interaction();

        println!("mul_opcode_claims log_size: {:?}", mul_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_interaction_claim = mul_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_19_h_relation,
                    &range_check_19_relation,
                    &range_check_19_b_relation,
                    &range_check_19_c_relation,
                    &range_check_19_d_relation,
                    &range_check_19_e_relation,
                    &range_check_19_f_relation,
                    &range_check_19_g_relation,
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
                eval_id: fnv1a_eval_id_gen("mul_opcode"),
                claim: mul_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
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
