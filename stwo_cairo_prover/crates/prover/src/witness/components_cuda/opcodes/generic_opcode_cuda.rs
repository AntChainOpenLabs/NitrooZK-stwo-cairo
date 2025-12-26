#![allow(unused_parens)]
use cairo_air::components::generic_opcode::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda};
use crate::witness::components::{
    memory_address_to_id, memory_id_to_big, verify_instruction,
    range_check_9_9, range_check_9_9_b, range_check_9_9_c, range_check_9_9_d,
    range_check_9_9_e, range_check_9_9_f, range_check_9_9_g, range_check_9_9_h,
    range_check_19, range_check_19_b, range_check_19_c, range_check_19_d,
    range_check_19_e, range_check_19_f, range_check_19_g, range_check_19_h,
    range_check_18, range_check_11,
};
use stwo::prover::backend::cuda::CudaBackend;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 244;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 34;

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
        range_check_9_9_state: &range_check_9_9::ClaimGenerator,
        range_check_9_9_b_state: &range_check_9_9_b::ClaimGenerator,
        range_check_9_9_c_state: &range_check_9_9_c::ClaimGenerator,
        range_check_9_9_d_state: &range_check_9_9_d::ClaimGenerator,
        range_check_9_9_e_state: &range_check_9_9_e::ClaimGenerator,
        range_check_9_9_f_state: &range_check_9_9_f::ClaimGenerator,
        range_check_9_9_g_state: &range_check_9_9_g::ClaimGenerator,
        range_check_9_9_h_state: &range_check_9_9_h::ClaimGenerator,
        range_check_19_state: &range_check_19::ClaimGenerator,
        range_check_19_b_state: &range_check_19_b::ClaimGenerator,
        range_check_19_c_state: &range_check_19_c::ClaimGenerator,
        range_check_19_d_state: &range_check_19_d::ClaimGenerator,
        range_check_19_e_state: &range_check_19_e::ClaimGenerator,
        range_check_19_f_state: &range_check_19_f::ClaimGenerator,
        range_check_19_g_state: &range_check_19_g::ClaimGenerator,
        range_check_19_h_state: &range_check_19_h::ClaimGenerator,
        range_check_18_state: &range_check_18::ClaimGenerator,
        range_check_11_state: &range_check_11::ClaimGenerator,
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
        let n_rows = self.n_rows;
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
        // IMPORTANT: Use padded_size to match SIMD behavior - the CUDA kernel fills padding rows
        // with the first row's values, so we need to add multiplicities for all padded rows.
        // range_check_9_9: 4 lookups × 2 fields
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_0,
                1 => &lookup_data.range_check_9_9_1,
                2 => &lookup_data.range_check_9_9_2,
                3 => &lookup_data.range_check_9_9_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                range_check_9_9_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_b: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_b_0,
                1 => &lookup_data.range_check_9_9_b_1,
                2 => &lookup_data.range_check_9_9_b_2,
                3 => &lookup_data.range_check_9_9_b_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_b value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_b_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_c: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_c_0,
                1 => &lookup_data.range_check_9_9_c_1,
                2 => &lookup_data.range_check_9_9_c_2,
                3 => &lookup_data.range_check_9_9_c_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_c value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_c_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_d: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_d_0,
                1 => &lookup_data.range_check_9_9_d_1,
                2 => &lookup_data.range_check_9_9_d_2,
                3 => &lookup_data.range_check_9_9_d_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_d value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_d_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_e: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_e_0,
                1 => &lookup_data.range_check_9_9_e_1,
                2 => &lookup_data.range_check_9_9_e_2,
                3 => &lookup_data.range_check_9_9_e_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_e value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_e_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_f: 4 lookups
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_f_0,
                1 => &lookup_data.range_check_9_9_f_1,
                2 => &lookup_data.range_check_9_9_f_2,
                3 => &lookup_data.range_check_9_9_f_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_f value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_f_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_g: 2 lookups
        for i in 0..2 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_g_0,
                1 => &lookup_data.range_check_9_9_g_1,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_g value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_g_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_9_9_h: 2 lookups
        for i in 0..2 {
            let lookup = match i {
                0 => &lookup_data.range_check_9_9_h_0,
                1 => &lookup_data.range_check_9_9_h_1,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            let field1: Vec<M31> = lookup[1].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= 512 || field1[j].0 >= 512 {
                    eprintln!("WARNING: Invalid range_check_9_9_h value at lookup {} row {}: ({}, {}) - should be < 512",
                        i, j, field0[j].0, field1[j].0);
                }
                range_check_9_9_h_state.add_input(&[field0[j], field1[j]]);
            }
        }

        // range_check_19: 4 lookups × 1 field
        const RC19_MAX: u32 = 1 << 19;
        for i in 0..4 {
            let lookup = match i {
                0 => &lookup_data.range_check_19_0,
                1 => &lookup_data.range_check_19_1,
                2 => &lookup_data.range_check_19_2,
                3 => &lookup_data.range_check_19_3,
                _ => unreachable!(),
            };
            let field0: Vec<M31> = lookup[0].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19 value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_b value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_c value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_d value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_e value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_f value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_g value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
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
            for j in 0..padded_size {
                if field0[j].0 >= RC19_MAX {
                    eprintln!("WARNING: Invalid range_check_19_h value at lookup {} row {}: {} - should be < {}",
                        i, j, field0[j].0, RC19_MAX);
                }
                range_check_19_h_state.add_input(&[field0[j]]);
            }
        }

        // range_check_18: 1 lookup
        const RC18_MAX: u32 = 1 << 18;
        {
            let field0: Vec<M31> = lookup_data.range_check_18_0[0].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= RC18_MAX {
                    eprintln!("WARNING: Invalid range_check_18 value at row {}: {} - should be < {}",
                        j, field0[j].0, RC18_MAX);
                }
                range_check_18_state.add_input(&[field0[j]]);
            }
        }

        // range_check_11: 1 lookup
        const RC11_MAX: u32 = 1 << 11;
        {
            let field0: Vec<M31> = lookup_data.range_check_11_0[0].to_vec();
            for j in 0..padded_size {
                if field0[j].0 >= RC11_MAX {
                    eprintln!("WARNING: Invalid range_check_11 value at row {}: {} - should be < {}",
                        j, field0[j].0, RC11_MAX);
                }
                range_check_11_state.add_input(&[field0[j]]);
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
                // range_check_9_9: 4 lookups × 2 fields
                range_check_9_9_0: init_lookup_array!(log_size),
                range_check_9_9_1: init_lookup_array!(log_size),
                range_check_9_9_2: init_lookup_array!(log_size),
                range_check_9_9_3: init_lookup_array!(log_size),
                // range_check_9_9_b: 4 lookups × 2 fields
                range_check_9_9_b_0: init_lookup_array!(log_size),
                range_check_9_9_b_1: init_lookup_array!(log_size),
                range_check_9_9_b_2: init_lookup_array!(log_size),
                range_check_9_9_b_3: init_lookup_array!(log_size),
                // range_check_9_9_c: 4 lookups × 2 fields
                range_check_9_9_c_0: init_lookup_array!(log_size),
                range_check_9_9_c_1: init_lookup_array!(log_size),
                range_check_9_9_c_2: init_lookup_array!(log_size),
                range_check_9_9_c_3: init_lookup_array!(log_size),
                // range_check_9_9_d: 4 lookups × 2 fields
                range_check_9_9_d_0: init_lookup_array!(log_size),
                range_check_9_9_d_1: init_lookup_array!(log_size),
                range_check_9_9_d_2: init_lookup_array!(log_size),
                range_check_9_9_d_3: init_lookup_array!(log_size),
                // range_check_9_9_e: 4 lookups × 2 fields
                range_check_9_9_e_0: init_lookup_array!(log_size),
                range_check_9_9_e_1: init_lookup_array!(log_size),
                range_check_9_9_e_2: init_lookup_array!(log_size),
                range_check_9_9_e_3: init_lookup_array!(log_size),
                // range_check_9_9_f: 4 lookups × 2 fields
                range_check_9_9_f_0: init_lookup_array!(log_size),
                range_check_9_9_f_1: init_lookup_array!(log_size),
                range_check_9_9_f_2: init_lookup_array!(log_size),
                range_check_9_9_f_3: init_lookup_array!(log_size),
                // range_check_9_9_g: 2 lookups × 2 fields
                range_check_9_9_g_0: init_lookup_array!(log_size),
                range_check_9_9_g_1: init_lookup_array!(log_size),
                // range_check_9_9_h: 2 lookups × 2 fields
                range_check_9_9_h_0: init_lookup_array!(log_size),
                range_check_9_9_h_1: init_lookup_array!(log_size),
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
                // range_check_18: 1 lookup × 1 field
                range_check_18_0: init_lookup_array!(log_size),
                // range_check_11: 1 lookup × 1 field
                range_check_11_0: init_lookup_array!(log_size),
                // verify_instruction: 1 lookup × 7 fields
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

    // Collect all lookup pointers
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_opcodes_0 = collect_lookup_ptrs!(lookup_data, opcodes_0);
    let lookup_opcodes_1 = collect_lookup_ptrs!(lookup_data, opcodes_1);

    let lookup_range_check_9_9_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_0);
    let lookup_range_check_9_9_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_1);
    let lookup_range_check_9_9_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_2);
    let lookup_range_check_9_9_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_3);

    let lookup_range_check_9_9_b_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_0);
    let lookup_range_check_9_9_b_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_1);
    let lookup_range_check_9_9_b_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_2);
    let lookup_range_check_9_9_b_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_3);

    let lookup_range_check_9_9_c_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_0);
    let lookup_range_check_9_9_c_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_1);
    let lookup_range_check_9_9_c_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_2);
    let lookup_range_check_9_9_c_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_3);

    let lookup_range_check_9_9_d_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_0);
    let lookup_range_check_9_9_d_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_1);
    let lookup_range_check_9_9_d_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_2);
    let lookup_range_check_9_9_d_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_3);

    let lookup_range_check_9_9_e_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_0);
    let lookup_range_check_9_9_e_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_1);
    let lookup_range_check_9_9_e_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_2);
    let lookup_range_check_9_9_e_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_3);

    let lookup_range_check_9_9_f_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_0);
    let lookup_range_check_9_9_f_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_1);
    let lookup_range_check_9_9_f_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_2);
    let lookup_range_check_9_9_f_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_3);

    let lookup_range_check_9_9_g_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_g_0);
    let lookup_range_check_9_9_g_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_g_1);

    let lookup_range_check_9_9_h_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_h_0);
    let lookup_range_check_9_9_h_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_h_1);

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

    let lookup_range_check_18_0 = collect_lookup_ptrs!(lookup_data, range_check_18_0);
    let lookup_range_check_11_0 = collect_lookup_ptrs!(lookup_data, range_check_11_0);
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
        bindings_airs::generate_generic_opcode_traces(
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

            // range_check_9_9 (4 lookups)
            lookup_range_check_9_9_0.as_ptr(),
            lookup_range_check_9_9_1.as_ptr(),
            lookup_range_check_9_9_2.as_ptr(),
            lookup_range_check_9_9_3.as_ptr(),

            // range_check_9_9_b (4 lookups)
            lookup_range_check_9_9_b_0.as_ptr(),
            lookup_range_check_9_9_b_1.as_ptr(),
            lookup_range_check_9_9_b_2.as_ptr(),
            lookup_range_check_9_9_b_3.as_ptr(),

            // range_check_9_9_c (4 lookups)
            lookup_range_check_9_9_c_0.as_ptr(),
            lookup_range_check_9_9_c_1.as_ptr(),
            lookup_range_check_9_9_c_2.as_ptr(),
            lookup_range_check_9_9_c_3.as_ptr(),

            // range_check_9_9_d (4 lookups)
            lookup_range_check_9_9_d_0.as_ptr(),
            lookup_range_check_9_9_d_1.as_ptr(),
            lookup_range_check_9_9_d_2.as_ptr(),
            lookup_range_check_9_9_d_3.as_ptr(),

            // range_check_9_9_e (4 lookups)
            lookup_range_check_9_9_e_0.as_ptr(),
            lookup_range_check_9_9_e_1.as_ptr(),
            lookup_range_check_9_9_e_2.as_ptr(),
            lookup_range_check_9_9_e_3.as_ptr(),

            // range_check_9_9_f (4 lookups)
            lookup_range_check_9_9_f_0.as_ptr(),
            lookup_range_check_9_9_f_1.as_ptr(),
            lookup_range_check_9_9_f_2.as_ptr(),
            lookup_range_check_9_9_f_3.as_ptr(),

            // range_check_9_9_g (2 lookups)
            lookup_range_check_9_9_g_0.as_ptr(),
            lookup_range_check_9_9_g_1.as_ptr(),

            // range_check_9_9_h (2 lookups)
            lookup_range_check_9_9_h_0.as_ptr(),
            lookup_range_check_9_9_h_1.as_ptr(),

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

            // range_check_18 (1 lookup)
            lookup_range_check_18_0.as_ptr(),

            // range_check_11 (1 lookup)
            lookup_range_check_11_0.as_ptr(),

            // verify_instruction (1 lookup)
            lookup_verify_instruction_0.as_ptr(),

            // Sub-component inputs
            sub_component_inputs_verify_instruction_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),

            // Opcode inputs
            opcodes_input_vec.as_ptr(),

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
    // range_check_9_9: 4 lookups × 2 fields
    range_check_9_9_0: [BaseFieldVec; 2],
    range_check_9_9_1: [BaseFieldVec; 2],
    range_check_9_9_2: [BaseFieldVec; 2],
    range_check_9_9_3: [BaseFieldVec; 2],
    // range_check_9_9_b: 4 lookups × 2 fields
    range_check_9_9_b_0: [BaseFieldVec; 2],
    range_check_9_9_b_1: [BaseFieldVec; 2],
    range_check_9_9_b_2: [BaseFieldVec; 2],
    range_check_9_9_b_3: [BaseFieldVec; 2],
    // range_check_9_9_c: 4 lookups × 2 fields
    range_check_9_9_c_0: [BaseFieldVec; 2],
    range_check_9_9_c_1: [BaseFieldVec; 2],
    range_check_9_9_c_2: [BaseFieldVec; 2],
    range_check_9_9_c_3: [BaseFieldVec; 2],
    // range_check_9_9_d: 4 lookups × 2 fields
    range_check_9_9_d_0: [BaseFieldVec; 2],
    range_check_9_9_d_1: [BaseFieldVec; 2],
    range_check_9_9_d_2: [BaseFieldVec; 2],
    range_check_9_9_d_3: [BaseFieldVec; 2],
    // range_check_9_9_e: 4 lookups × 2 fields
    range_check_9_9_e_0: [BaseFieldVec; 2],
    range_check_9_9_e_1: [BaseFieldVec; 2],
    range_check_9_9_e_2: [BaseFieldVec; 2],
    range_check_9_9_e_3: [BaseFieldVec; 2],
    // range_check_9_9_f: 4 lookups × 2 fields
    range_check_9_9_f_0: [BaseFieldVec; 2],
    range_check_9_9_f_1: [BaseFieldVec; 2],
    range_check_9_9_f_2: [BaseFieldVec; 2],
    range_check_9_9_f_3: [BaseFieldVec; 2],
    // range_check_9_9_g: 2 lookups × 2 fields
    range_check_9_9_g_0: [BaseFieldVec; 2],
    range_check_9_9_g_1: [BaseFieldVec; 2],
    // range_check_9_9_h: 2 lookups × 2 fields
    range_check_9_9_h_0: [BaseFieldVec; 2],
    range_check_9_9_h_1: [BaseFieldVec; 2],
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
    // range_check_18: 1 lookup × 1 field
    range_check_18_0: [BaseFieldVec; 1],
    // range_check_11: 1 lookup × 1 field
    range_check_11_0: [BaseFieldVec; 1],
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
        range_check_9_9: &relations::RangeCheck_9_9,
        range_check_9_9_b: &relations::RangeCheck_9_9_B,
        range_check_9_9_c: &relations::RangeCheck_9_9_C,
        range_check_9_9_d: &relations::RangeCheck_9_9_D,
        range_check_9_9_e: &relations::RangeCheck_9_9_E,
        range_check_9_9_f: &relations::RangeCheck_9_9_F,
        range_check_9_9_g: &relations::RangeCheck_9_9_G,
        range_check_9_9_h: &relations::RangeCheck_9_9_H,
        range_check_19_h: &relations::RangeCheck_19_H,
        range_check_19: &relations::RangeCheck_19,
        range_check_19_b: &relations::RangeCheck_19_B,
        range_check_19_c: &relations::RangeCheck_19_C,
        range_check_19_d: &relations::RangeCheck_19_D,
        range_check_19_e: &relations::RangeCheck_19_E,
        range_check_19_f: &relations::RangeCheck_19_F,
        range_check_19_g: &relations::RangeCheck_19_G,
        range_check_18: &relations::RangeCheck_18,
        range_check_11: &relations::RangeCheck_11,
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

        let lookup_range_check_9_9_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_0);
        let lookup_range_check_9_9_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_1);
        let lookup_range_check_9_9_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_2);
        let lookup_range_check_9_9_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_3);

        let lookup_range_check_9_9_b_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_0);
        let lookup_range_check_9_9_b_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_1);
        let lookup_range_check_9_9_b_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_2);
        let lookup_range_check_9_9_b_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_3);

        let lookup_range_check_9_9_c_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_0);
        let lookup_range_check_9_9_c_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_1);
        let lookup_range_check_9_9_c_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_2);
        let lookup_range_check_9_9_c_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_3);

        let lookup_range_check_9_9_d_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_0);
        let lookup_range_check_9_9_d_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_1);
        let lookup_range_check_9_9_d_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_2);
        let lookup_range_check_9_9_d_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_3);

        let lookup_range_check_9_9_e_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_0);
        let lookup_range_check_9_9_e_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_1);
        let lookup_range_check_9_9_e_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_2);
        let lookup_range_check_9_9_e_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_3);

        let lookup_range_check_9_9_f_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_0);
        let lookup_range_check_9_9_f_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_1);
        let lookup_range_check_9_9_f_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_2);
        let lookup_range_check_9_9_f_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_3);

        let lookup_range_check_9_9_g_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_g_0);
        let lookup_range_check_9_9_g_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_g_1);

        let lookup_range_check_9_9_h_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_h_0);
        let lookup_range_check_9_9_h_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_9_9_h_1);

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

        let lookup_range_check_18_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_0);
        let lookup_range_check_11_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_11_0);
        let lookup_verify_instruction_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_instruction_0);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;
            let opcodes_ptr = opcodes as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_ptr = range_check_9_9 as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_b_ptr = range_check_9_9_b as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_c_ptr = range_check_9_9_c as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_d_ptr = range_check_9_9_d as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_e_ptr = range_check_9_9_e as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_f_ptr = range_check_9_9_f as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_g_ptr = range_check_9_9_g as *const _ as *mut std::os::raw::c_void;
            let range_check_9_9_h_ptr = range_check_9_9_h as *const _ as *mut std::os::raw::c_void;
            let range_check_19_ptr = range_check_19 as *const _ as *mut std::os::raw::c_void;
            let range_check_19_b_ptr = range_check_19_b as *const _ as *mut std::os::raw::c_void;
            let range_check_19_c_ptr = range_check_19_c as *const _ as *mut std::os::raw::c_void;
            let range_check_19_d_ptr = range_check_19_d as *const _ as *mut std::os::raw::c_void;
            let range_check_19_e_ptr = range_check_19_e as *const _ as *mut std::os::raw::c_void;
            let range_check_19_f_ptr = range_check_19_f as *const _ as *mut std::os::raw::c_void;
            let range_check_19_g_ptr = range_check_19_g as *const _ as *mut std::os::raw::c_void;
            let range_check_19_h_ptr = range_check_19_h as *const _ as *mut std::os::raw::c_void;
            let range_check_18_ptr = range_check_18 as *const _ as *mut std::os::raw::c_void;
            let range_check_11_ptr = range_check_11 as *const _ as *mut std::os::raw::c_void;
            let verify_instruction_ptr = verify_instruction as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_generic_opcode_interaction_traces(
                // Relation pointers (22 relations)
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                opcodes_ptr,
                range_check_9_9_ptr,
                range_check_9_9_b_ptr,
                range_check_9_9_c_ptr,
                range_check_9_9_d_ptr,
                range_check_9_9_e_ptr,
                range_check_9_9_f_ptr,
                range_check_9_9_g_ptr,
                range_check_9_9_h_ptr,
                range_check_19_ptr,
                range_check_19_b_ptr,
                range_check_19_c_ptr,
                range_check_19_d_ptr,
                range_check_19_e_ptr,
                range_check_19_f_ptr,
                range_check_19_g_ptr,
                range_check_19_h_ptr,
                range_check_18_ptr,
                range_check_11_ptr,
                verify_instruction_ptr,

                // All lookup data (67 lookups)
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_opcodes_0_vec.as_ptr(),
                lookup_opcodes_1_vec.as_ptr(),
                lookup_range_check_9_9_0_vec.as_ptr(),
                lookup_range_check_9_9_1_vec.as_ptr(),
                lookup_range_check_9_9_2_vec.as_ptr(),
                lookup_range_check_9_9_3_vec.as_ptr(),
                lookup_range_check_9_9_b_0_vec.as_ptr(),
                lookup_range_check_9_9_b_1_vec.as_ptr(),
                lookup_range_check_9_9_b_2_vec.as_ptr(),
                lookup_range_check_9_9_b_3_vec.as_ptr(),
                lookup_range_check_9_9_c_0_vec.as_ptr(),
                lookup_range_check_9_9_c_1_vec.as_ptr(),
                lookup_range_check_9_9_c_2_vec.as_ptr(),
                lookup_range_check_9_9_c_3_vec.as_ptr(),
                lookup_range_check_9_9_d_0_vec.as_ptr(),
                lookup_range_check_9_9_d_1_vec.as_ptr(),
                lookup_range_check_9_9_d_2_vec.as_ptr(),
                lookup_range_check_9_9_d_3_vec.as_ptr(),
                lookup_range_check_9_9_e_0_vec.as_ptr(),
                lookup_range_check_9_9_e_1_vec.as_ptr(),
                lookup_range_check_9_9_e_2_vec.as_ptr(),
                lookup_range_check_9_9_e_3_vec.as_ptr(),
                lookup_range_check_9_9_f_0_vec.as_ptr(),
                lookup_range_check_9_9_f_1_vec.as_ptr(),
                lookup_range_check_9_9_f_2_vec.as_ptr(),
                lookup_range_check_9_9_f_3_vec.as_ptr(),
                lookup_range_check_9_9_g_0_vec.as_ptr(),
                lookup_range_check_9_9_g_1_vec.as_ptr(),
                lookup_range_check_9_9_h_0_vec.as_ptr(),
                lookup_range_check_9_9_h_1_vec.as_ptr(),
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
                lookup_range_check_18_0_vec.as_ptr(),
                lookup_range_check_11_0_vec.as_ptr(),
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
    use crate::witness::components::{
        memory_address_to_id, memory_id_to_big, verify_instruction,
        range_check_9_9, range_check_9_9_b, range_check_9_9_c, range_check_9_9_d,
        range_check_9_9_e, range_check_9_9_f, range_check_9_9_g, range_check_9_9_h,
        range_check_19, range_check_19_b, range_check_19_c, range_check_19_d,
        range_check_19_e, range_check_19_f, range_check_19_g, range_check_19_h,
        range_check_18, range_check_11,
    };
    use crate::witness::components_cuda::{
        memory_address_to_id_cuda, memory_id_to_big_cuda, verify_instruction_cuda,
    };
    use super::CudaClaimGenerator;
    use cairo_air::relations;

    use stwo_constraint_framework::TraceLocationAllocator;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::generic_opcode::Eval;
    use stwo::core::fields::m31::M31;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
    use crate::witness::components::generic_opcode;
    use cairo_lang_casm::casm;
    use crate::test_utils::input_from_plain_casm;
    use cairo_air::components::generic_opcode::Component;
    use itertools::Itertools;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::prover::backend::Column;
    use stwo::stwo_cuda::bindings::CudaSecureField;
    use stwo::core::fields::m31::BaseField;

    #[test]
    fn test_generic_opcode_cpu_ref() {
        // Create a test case that generates generic opcode instructions
        // Generic opcode handles most Cairo VM instructions
        let instructions = casm! {
        [ap]=1, ap++;
        [ap]=2, ap++;
        jmp rel [ap-2] if [ap-1] != 0;
        [ap]=1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have generic_opcode states
        if input_state.casm_states_by_opcode.generic_opcode.is_empty() {
            println!("Warning: No generic_opcode states generated. Test will be skipped.");
            return;
        }

        let generic_opcode_gen = generic_opcode::ClaimGenerator::new(
            input_state.casm_states_by_opcode.generic_opcode,
        );

        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Initialize all range check generators
        let range_check_9_9_trace_generator = range_check_9_9::ClaimGenerator::new();
        let range_check_9_9_b_trace_generator = range_check_9_9_b::ClaimGenerator::new();
        let range_check_9_9_c_trace_generator = range_check_9_9_c::ClaimGenerator::new();
        let range_check_9_9_d_trace_generator = range_check_9_9_d::ClaimGenerator::new();
        let range_check_9_9_e_trace_generator = range_check_9_9_e::ClaimGenerator::new();
        let range_check_9_9_f_trace_generator = range_check_9_9_f::ClaimGenerator::new();
        let range_check_9_9_g_trace_generator = range_check_9_9_g::ClaimGenerator::new();
        let range_check_9_9_h_trace_generator = range_check_9_9_h::ClaimGenerator::new();
        let range_check_19_trace_generator = range_check_19::ClaimGenerator::new();
        let range_check_19_b_trace_generator = range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_trace_generator = range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_trace_generator = range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_trace_generator = range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_trace_generator = range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_trace_generator = range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_trace_generator = range_check_19_h::ClaimGenerator::new();
        let range_check_18_trace_generator = range_check_18::ClaimGenerator::new();
        let range_check_11_trace_generator = range_check_11::ClaimGenerator::new();

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

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (generic_opcode_claim, generic_opcode_interaction_gen) = generic_opcode_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_11_trace_generator,
            &range_check_18_trace_generator,
            &range_check_19_trace_generator,
            &range_check_19_b_trace_generator,
            &range_check_19_c_trace_generator,
            &range_check_19_d_trace_generator,
            &range_check_19_e_trace_generator,
            &range_check_19_f_trace_generator,
            &range_check_19_g_trace_generator,
            &range_check_19_h_trace_generator,
            &range_check_9_9_trace_generator,
            &range_check_9_9_b_trace_generator,
            &range_check_9_9_c_trace_generator,
            &range_check_9_9_d_trace_generator,
            &range_check_9_9_e_trace_generator,
            &range_check_9_9_f_trace_generator,
            &range_check_9_9_g_trace_generator,
            &range_check_9_9_h_trace_generator,
            &verify_instruction_trace_generator,
        );

        mock_tree_builder.finalize_interaction();

        println!("generic_opcode_claim log_size: {:?}", generic_opcode_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let generic_opcode_interaction_claim = generic_opcode_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &relations::RangeCheck_9_9::dummy(),
            &relations::RangeCheck_9_9_B::dummy(),
            &relations::RangeCheck_9_9_C::dummy(),
            &relations::RangeCheck_9_9_D::dummy(),
            &relations::RangeCheck_9_9_E::dummy(),
            &relations::RangeCheck_9_9_F::dummy(),
            &relations::RangeCheck_9_9_G::dummy(),
            &relations::RangeCheck_9_9_H::dummy(),
            &relations::RangeCheck_19_H::dummy(),
            &relations::RangeCheck_19::dummy(),
            &relations::RangeCheck_19_B::dummy(),
            &relations::RangeCheck_19_C::dummy(),
            &relations::RangeCheck_19_D::dummy(),
            &relations::RangeCheck_19_E::dummy(),
            &relations::RangeCheck_19_F::dummy(),
            &relations::RangeCheck_19_G::dummy(),
            &relations::RangeCheck_18::dummy(),
            &relations::RangeCheck_11::dummy(),
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("generic_opcode_interaction_claim.claimed_sum: {:?}", generic_opcode_interaction_claim.claimed_sum);

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let generic_opcode_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("generic_opcode"),
                claim: generic_opcode_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_9_9_lookup_elements: relations::RangeCheck_9_9::dummy(),
                range_check_9_9_b_lookup_elements: relations::RangeCheck_9_9_B::dummy(),
                range_check_9_9_c_lookup_elements: relations::RangeCheck_9_9_C::dummy(),
                range_check_9_9_d_lookup_elements: relations::RangeCheck_9_9_D::dummy(),
                range_check_9_9_e_lookup_elements: relations::RangeCheck_9_9_E::dummy(),
                range_check_9_9_f_lookup_elements: relations::RangeCheck_9_9_F::dummy(),
                range_check_9_9_g_lookup_elements: relations::RangeCheck_9_9_G::dummy(),
                range_check_9_9_h_lookup_elements: relations::RangeCheck_9_9_H::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            generic_opcode_interaction_claim.claimed_sum,
        );

        assert_component(&generic_opcode_component, &trace)
    }

    #[test]
    fn test_generic_opcode_trace_gen_by_cpu_and_verify_by_cuda() {
        // Create a test case that generates generic opcode instructions
        let instructions = casm! {
            [ap]=1, ap++;
            [ap]=2, ap++;
            jmp rel [ap-2] if [ap-1] != 0;
            [ap]=1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have generic_opcode states
        if input_state.casm_states_by_opcode.generic_opcode.is_empty() {
            println!("Warning: No generic_opcode states generated. Test will be skipped.");
            return;
        }

        let generic_opcode_gen = generic_opcode::ClaimGenerator::new(
            input_state.casm_states_by_opcode.generic_opcode,
        );

        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Initialize all range check generators
        let range_check_9_9_trace_generator = range_check_9_9::ClaimGenerator::new();
        let range_check_9_9_b_trace_generator = range_check_9_9_b::ClaimGenerator::new();
        let range_check_9_9_c_trace_generator = range_check_9_9_c::ClaimGenerator::new();
        let range_check_9_9_d_trace_generator = range_check_9_9_d::ClaimGenerator::new();
        let range_check_9_9_e_trace_generator = range_check_9_9_e::ClaimGenerator::new();
        let range_check_9_9_f_trace_generator = range_check_9_9_f::ClaimGenerator::new();
        let range_check_9_9_g_trace_generator = range_check_9_9_g::ClaimGenerator::new();
        let range_check_9_9_h_trace_generator = range_check_9_9_h::ClaimGenerator::new();
        let range_check_19_trace_generator = range_check_19::ClaimGenerator::new();
        let range_check_19_b_trace_generator = range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_trace_generator = range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_trace_generator = range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_trace_generator = range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_trace_generator = range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_trace_generator = range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_trace_generator = range_check_19_h::ClaimGenerator::new();
        let range_check_18_trace_generator = range_check_18::ClaimGenerator::new();
        let range_check_11_trace_generator = range_check_11::ClaimGenerator::new();

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

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (generic_opcode_claim, generic_opcode_interaction_gen) = generic_opcode_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_11_trace_generator,
            &range_check_18_trace_generator,
            &range_check_19_trace_generator,
            &range_check_19_b_trace_generator,
            &range_check_19_c_trace_generator,
            &range_check_19_d_trace_generator,
            &range_check_19_e_trace_generator,
            &range_check_19_f_trace_generator,
            &range_check_19_g_trace_generator,
            &range_check_19_h_trace_generator,
            &range_check_9_9_trace_generator,
            &range_check_9_9_b_trace_generator,
            &range_check_9_9_c_trace_generator,
            &range_check_9_9_d_trace_generator,
            &range_check_9_9_e_trace_generator,
            &range_check_9_9_f_trace_generator,
            &range_check_9_9_g_trace_generator,
            &range_check_9_9_h_trace_generator,
            &verify_instruction_trace_generator,
        );

        mock_tree_builder.finalize_interaction();

        println!("generic_opcode_claim log_size: {:?}", generic_opcode_claim.log_size);

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let generic_opcode_interaction_claim = generic_opcode_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &relations::RangeCheck_9_9::dummy(),
            &relations::RangeCheck_9_9_B::dummy(),
            &relations::RangeCheck_9_9_C::dummy(),
            &relations::RangeCheck_9_9_D::dummy(),
            &relations::RangeCheck_9_9_E::dummy(),
            &relations::RangeCheck_9_9_F::dummy(),
            &relations::RangeCheck_9_9_G::dummy(),
            &relations::RangeCheck_9_9_H::dummy(),
            &relations::RangeCheck_19_H::dummy(),
            &relations::RangeCheck_19::dummy(),
            &relations::RangeCheck_19_B::dummy(),
            &relations::RangeCheck_19_C::dummy(),
            &relations::RangeCheck_19_D::dummy(),
            &relations::RangeCheck_19_E::dummy(),
            &relations::RangeCheck_19_F::dummy(),
            &relations::RangeCheck_19_G::dummy(),
            &relations::RangeCheck_18::dummy(),
            &relations::RangeCheck_11::dummy(),
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
        let generic_opcode_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("generic_opcode"),
                claim: generic_opcode_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_9_9_lookup_elements: relations::RangeCheck_9_9::dummy(),
                range_check_9_9_b_lookup_elements: relations::RangeCheck_9_9_B::dummy(),
                range_check_9_9_c_lookup_elements: relations::RangeCheck_9_9_C::dummy(),
                range_check_9_9_d_lookup_elements: relations::RangeCheck_9_9_D::dummy(),
                range_check_9_9_e_lookup_elements: relations::RangeCheck_9_9_E::dummy(),
                range_check_9_9_f_lookup_elements: relations::RangeCheck_9_9_F::dummy(),
                range_check_9_9_g_lookup_elements: relations::RangeCheck_9_9_G::dummy(),
                range_check_9_9_h_lookup_elements: relations::RangeCheck_9_9_H::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            generic_opcode_interaction_claim.claimed_sum,
        );

        println!("n_constraints: {}", generic_opcode_component.info.n_constraints);
        println!("logup_counts: {}", generic_opcode_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>());

        let eval_ptr = &generic_opcode_component.eval as *const _ as *mut std::os::raw::c_void;
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
                generic_opcode_claim.log_size as u32,
                generic_opcode_claim.log_size as u32,
                generic_opcode_component.info.n_constraints as u32,
                generic_opcode_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    generic_opcode_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << generic_opcode_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator - must be true per user requirement
            );
        }

        println!("CUDA evaluator test completed successfully!");
    }

    #[test]
    fn test_generic_opcode_trace_gen_by_cuda_and_verify_by_cpu() {
        // Create a test case that generates generic opcode instructions
        let instructions = casm! {
            [ap]=1, ap++;
            [ap]=2, ap++;
            jmp rel [ap-2] if [ap-1] != 0;
            [ap]=1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have generic_opcode states
        if input_state.casm_states_by_opcode.generic_opcode.is_empty() {
            println!("Warning: No generic_opcode states generated. Test will be skipped.");
            return;
        }

        let generic_opcode_gen = CudaClaimGenerator::new(
            input_state.casm_states_by_opcode.generic_opcode,
        );

        let mut memory_address_to_id_cuda_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        // Initialize all range check generators
        let range_check_9_9_trace_generator = range_check_9_9::ClaimGenerator::new();
        let range_check_9_9_b_trace_generator = range_check_9_9_b::ClaimGenerator::new();
        let range_check_9_9_c_trace_generator = range_check_9_9_c::ClaimGenerator::new();
        let range_check_9_9_d_trace_generator = range_check_9_9_d::ClaimGenerator::new();
        let range_check_9_9_e_trace_generator = range_check_9_9_e::ClaimGenerator::new();
        let range_check_9_9_f_trace_generator = range_check_9_9_f::ClaimGenerator::new();
        let range_check_9_9_g_trace_generator = range_check_9_9_g::ClaimGenerator::new();
        let range_check_9_9_h_trace_generator = range_check_9_9_h::ClaimGenerator::new();
        let range_check_19_trace_generator = range_check_19::ClaimGenerator::new();
        let range_check_19_b_trace_generator = range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_trace_generator = range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_trace_generator = range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_trace_generator = range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_trace_generator = range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_trace_generator = range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_trace_generator = range_check_19_h::ClaimGenerator::new();
        let range_check_18_trace_generator = range_check_18::ClaimGenerator::new();
        let range_check_11_trace_generator = range_check_11::ClaimGenerator::new();

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

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace - using CUDA trace generation
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (generic_opcode_claim, generic_opcode_interaction_gen) = generic_opcode_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_generator,
            &memory_id_to_big_cuda_generator,
            &verify_instruction_cuda_generator,
            &memory_address_to_id_simd_generator,
            &memory_id_to_big_simd_generator,
            &verify_instruction_simd_generator,
            &range_check_9_9_trace_generator,
            &range_check_9_9_b_trace_generator,
            &range_check_9_9_c_trace_generator,
            &range_check_9_9_d_trace_generator,
            &range_check_9_9_e_trace_generator,
            &range_check_9_9_f_trace_generator,
            &range_check_9_9_g_trace_generator,
            &range_check_9_9_h_trace_generator,
            &range_check_19_trace_generator,
            &range_check_19_b_trace_generator,
            &range_check_19_c_trace_generator,
            &range_check_19_d_trace_generator,
            &range_check_19_e_trace_generator,
            &range_check_19_f_trace_generator,
            &range_check_19_g_trace_generator,
            &range_check_19_h_trace_generator,
            &range_check_18_trace_generator,
            &range_check_11_trace_generator,
        );

        mock_tree_builder.finalize_interaction();

        println!("generic_opcode_claim log_size: {:?}", generic_opcode_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let generic_opcode_interaction_claim = generic_opcode_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &relations::RangeCheck_9_9::dummy(),
            &relations::RangeCheck_9_9_B::dummy(),
            &relations::RangeCheck_9_9_C::dummy(),
            &relations::RangeCheck_9_9_D::dummy(),
            &relations::RangeCheck_9_9_E::dummy(),
            &relations::RangeCheck_9_9_F::dummy(),
            &relations::RangeCheck_9_9_G::dummy(),
            &relations::RangeCheck_9_9_H::dummy(),
            &relations::RangeCheck_19_H::dummy(),
            &relations::RangeCheck_19::dummy(),
            &relations::RangeCheck_19_B::dummy(),
            &relations::RangeCheck_19_C::dummy(),
            &relations::RangeCheck_19_D::dummy(),
            &relations::RangeCheck_19_E::dummy(),
            &relations::RangeCheck_19_F::dummy(),
            &relations::RangeCheck_19_G::dummy(),
            &relations::RangeCheck_18::dummy(),
            &relations::RangeCheck_11::dummy(),
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("generic_opcode_interaction_claim.claimed_sum: {:?}", generic_opcode_interaction_claim.claimed_sum);

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let generic_opcode_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("generic_opcode"),
                claim: generic_opcode_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_9_9_lookup_elements: relations::RangeCheck_9_9::dummy(),
                range_check_9_9_b_lookup_elements: relations::RangeCheck_9_9_B::dummy(),
                range_check_9_9_c_lookup_elements: relations::RangeCheck_9_9_C::dummy(),
                range_check_9_9_d_lookup_elements: relations::RangeCheck_9_9_D::dummy(),
                range_check_9_9_e_lookup_elements: relations::RangeCheck_9_9_E::dummy(),
                range_check_9_9_f_lookup_elements: relations::RangeCheck_9_9_F::dummy(),
                range_check_9_9_g_lookup_elements: relations::RangeCheck_9_9_G::dummy(),
                range_check_9_9_h_lookup_elements: relations::RangeCheck_9_9_H::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            generic_opcode_interaction_claim.claimed_sum,
        );

        // Verify using CPU
        assert_component(&generic_opcode_component, &trace)
    }

    #[test]
    fn test_generic_opcode_trace_gen_by_cuda_and_verify_by_cuda() {
        // Create a test case that generates generic opcode instructions
        let instructions = casm! {
            [ap]=1, ap++;
            [ap]=2, ap++;
            jmp rel [ap-2] if [ap-1] != 0;
            [ap]=1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions);
        let input_state = input.state_transitions;

        // Check that we have generic_opcode states
        if input_state.casm_states_by_opcode.generic_opcode.is_empty() {
            println!("Warning: No generic_opcode states generated. Test will be skipped.");
            return;
        }

        let generic_opcode_gen = CudaClaimGenerator::new(
            input_state.casm_states_by_opcode.generic_opcode,
        );

        let mut memory_address_to_id_cuda_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        // Initialize all range check generators
        let range_check_9_9_trace_generator = range_check_9_9::ClaimGenerator::new();
        let range_check_9_9_b_trace_generator = range_check_9_9_b::ClaimGenerator::new();
        let range_check_9_9_c_trace_generator = range_check_9_9_c::ClaimGenerator::new();
        let range_check_9_9_d_trace_generator = range_check_9_9_d::ClaimGenerator::new();
        let range_check_9_9_e_trace_generator = range_check_9_9_e::ClaimGenerator::new();
        let range_check_9_9_f_trace_generator = range_check_9_9_f::ClaimGenerator::new();
        let range_check_9_9_g_trace_generator = range_check_9_9_g::ClaimGenerator::new();
        let range_check_9_9_h_trace_generator = range_check_9_9_h::ClaimGenerator::new();
        let range_check_19_trace_generator = range_check_19::ClaimGenerator::new();
        let range_check_19_b_trace_generator = range_check_19_b::ClaimGenerator::new();
        let range_check_19_c_trace_generator = range_check_19_c::ClaimGenerator::new();
        let range_check_19_d_trace_generator = range_check_19_d::ClaimGenerator::new();
        let range_check_19_e_trace_generator = range_check_19_e::ClaimGenerator::new();
        let range_check_19_f_trace_generator = range_check_19_f::ClaimGenerator::new();
        let range_check_19_g_trace_generator = range_check_19_g::ClaimGenerator::new();
        let range_check_19_h_trace_generator = range_check_19_h::ClaimGenerator::new();
        let range_check_18_trace_generator = range_check_18::ClaimGenerator::new();
        let range_check_11_trace_generator = range_check_11::ClaimGenerator::new();

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

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let opcodes_relation = relations::Opcodes::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace - using CUDA trace generation
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (generic_opcode_claim, generic_opcode_interaction_gen) = generic_opcode_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_generator,
            &memory_id_to_big_cuda_generator,
            &verify_instruction_cuda_generator,
            &memory_address_to_id_simd_generator,
            &memory_id_to_big_simd_generator,
            &verify_instruction_simd_generator,
            &range_check_9_9_trace_generator,
            &range_check_9_9_b_trace_generator,
            &range_check_9_9_c_trace_generator,
            &range_check_9_9_d_trace_generator,
            &range_check_9_9_e_trace_generator,
            &range_check_9_9_f_trace_generator,
            &range_check_9_9_g_trace_generator,
            &range_check_9_9_h_trace_generator,
            &range_check_19_trace_generator,
            &range_check_19_b_trace_generator,
            &range_check_19_c_trace_generator,
            &range_check_19_d_trace_generator,
            &range_check_19_e_trace_generator,
            &range_check_19_f_trace_generator,
            &range_check_19_g_trace_generator,
            &range_check_19_h_trace_generator,
            &range_check_18_trace_generator,
            &range_check_11_trace_generator,
        );

        mock_tree_builder.finalize_interaction();

        println!("generic_opcode_claim log_size: {:?}", generic_opcode_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let generic_opcode_interaction_claim = generic_opcode_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &relations::RangeCheck_9_9::dummy(),
            &relations::RangeCheck_9_9_B::dummy(),
            &relations::RangeCheck_9_9_C::dummy(),
            &relations::RangeCheck_9_9_D::dummy(),
            &relations::RangeCheck_9_9_E::dummy(),
            &relations::RangeCheck_9_9_F::dummy(),
            &relations::RangeCheck_9_9_G::dummy(),
            &relations::RangeCheck_9_9_H::dummy(),
            &relations::RangeCheck_19_H::dummy(),
            &relations::RangeCheck_19::dummy(),
            &relations::RangeCheck_19_B::dummy(),
            &relations::RangeCheck_19_C::dummy(),
            &relations::RangeCheck_19_D::dummy(),
            &relations::RangeCheck_19_E::dummy(),
            &relations::RangeCheck_19_F::dummy(),
            &relations::RangeCheck_19_G::dummy(),
            &relations::RangeCheck_18::dummy(),
            &relations::RangeCheck_11::dummy(),
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
        let generic_opcode_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("generic_opcode"),
                claim: generic_opcode_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_9_9_lookup_elements: relations::RangeCheck_9_9::dummy(),
                range_check_9_9_b_lookup_elements: relations::RangeCheck_9_9_B::dummy(),
                range_check_9_9_c_lookup_elements: relations::RangeCheck_9_9_C::dummy(),
                range_check_9_9_d_lookup_elements: relations::RangeCheck_9_9_D::dummy(),
                range_check_9_9_e_lookup_elements: relations::RangeCheck_9_9_E::dummy(),
                range_check_9_9_f_lookup_elements: relations::RangeCheck_9_9_F::dummy(),
                range_check_9_9_g_lookup_elements: relations::RangeCheck_9_9_G::dummy(),
                range_check_9_9_h_lookup_elements: relations::RangeCheck_9_9_H::dummy(),
                range_check_19_h_lookup_elements: relations::RangeCheck_19_H::dummy(),
                range_check_19_lookup_elements: relations::RangeCheck_19::dummy(),
                range_check_19_b_lookup_elements: relations::RangeCheck_19_B::dummy(),
                range_check_19_c_lookup_elements: relations::RangeCheck_19_C::dummy(),
                range_check_19_d_lookup_elements: relations::RangeCheck_19_D::dummy(),
                range_check_19_e_lookup_elements: relations::RangeCheck_19_E::dummy(),
                range_check_19_f_lookup_elements: relations::RangeCheck_19_F::dummy(),
                range_check_19_g_lookup_elements: relations::RangeCheck_19_G::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
                range_check_11_lookup_elements: relations::RangeCheck_11::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            generic_opcode_interaction_claim.claimed_sum,
        );

        println!("n_constraints: {}", generic_opcode_component.info.n_constraints);
        println!("logup_counts: {}", generic_opcode_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>());

        let eval_ptr = &generic_opcode_component.eval as *const _ as *mut std::os::raw::c_void;
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
                generic_opcode_claim.log_size as u32,
                generic_opcode_claim.log_size as u32,
                generic_opcode_component.info.n_constraints as u32,
                generic_opcode_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    generic_opcode_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << generic_opcode_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator - must be true per user requirement
            );
        }

        println!("CUDA trace gen + CUDA verify test completed successfully!");
    }

    #[test]
    fn test_generic_opcode_compare_cpu_vs_cuda_trace() {
        // Create the same test case for both CPU and CUDA
        let instructions = casm! {
            [ap]=1, ap++;
            [ap]=2, ap++;
            jmp rel [ap-2] if [ap-1] != 0;
            [ap]=1, ap++;
        }
        .instructions;

        let input = input_from_plain_casm(instructions.clone());
        let input_state = input.state_transitions;

        if input_state.casm_states_by_opcode.generic_opcode.is_empty() {
            println!("Warning: No generic_opcode states generated. Test will be skipped.");
            return;
        }

        // ---------- CPU TRACE GENERATION ----------
        let cpu_generic_opcode_gen = generic_opcode::ClaimGenerator::new(
            input_state.casm_states_by_opcode.generic_opcode.clone(),
        );

        let cpu_memory_address_to_id_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let cpu_memory_id_to_big_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let cpu_verify_instruction_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        let cpu_range_check_9_9 = range_check_9_9::ClaimGenerator::new();
        let cpu_range_check_9_9_b = range_check_9_9_b::ClaimGenerator::new();
        let cpu_range_check_9_9_c = range_check_9_9_c::ClaimGenerator::new();
        let cpu_range_check_9_9_d = range_check_9_9_d::ClaimGenerator::new();
        let cpu_range_check_9_9_e = range_check_9_9_e::ClaimGenerator::new();
        let cpu_range_check_9_9_f = range_check_9_9_f::ClaimGenerator::new();
        let cpu_range_check_9_9_g = range_check_9_9_g::ClaimGenerator::new();
        let cpu_range_check_9_9_h = range_check_9_9_h::ClaimGenerator::new();
        let cpu_range_check_19 = range_check_19::ClaimGenerator::new();
        let cpu_range_check_19_b = range_check_19_b::ClaimGenerator::new();
        let cpu_range_check_19_c = range_check_19_c::ClaimGenerator::new();
        let cpu_range_check_19_d = range_check_19_d::ClaimGenerator::new();
        let cpu_range_check_19_e = range_check_19_e::ClaimGenerator::new();
        let cpu_range_check_19_f = range_check_19_f::ClaimGenerator::new();
        let cpu_range_check_19_g = range_check_19_g::ClaimGenerator::new();
        let cpu_range_check_19_h = range_check_19_h::ClaimGenerator::new();
        let cpu_range_check_18 = range_check_18::ClaimGenerator::new();
        let cpu_range_check_11 = range_check_11::ClaimGenerator::new();

        let mut cpu_mock = MockCommitmentScheme::default();
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut cpu_tree_builder = cpu_mock.tree_builder();
        cpu_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        cpu_tree_builder.finalize_interaction();

        let mut cpu_tree_builder = cpu_mock.tree_builder();
        let (_cpu_claim, cpu_interaction_gen) = cpu_generic_opcode_gen.write_trace(
            &mut cpu_tree_builder,
            &cpu_memory_address_to_id_generator,
            &cpu_memory_id_to_big_generator,
            &cpu_range_check_11,
            &cpu_range_check_18,
            &cpu_range_check_19,
            &cpu_range_check_19_b,
            &cpu_range_check_19_c,
            &cpu_range_check_19_d,
            &cpu_range_check_19_e,
            &cpu_range_check_19_f,
            &cpu_range_check_19_g,
            &cpu_range_check_19_h,
            &cpu_range_check_9_9,
            &cpu_range_check_9_9_b,
            &cpu_range_check_9_9_c,
            &cpu_range_check_9_9_d,
            &cpu_range_check_9_9_e,
            &cpu_range_check_9_9_f,
            &cpu_range_check_9_9_g,
            &cpu_range_check_9_9_h,
            &cpu_verify_instruction_generator,
        );
        cpu_tree_builder.finalize_interaction();

        // Interaction trace for CPU
        let mut cpu_interaction_tree = cpu_mock.tree_builder();
        let cpu_interaction_claim = cpu_interaction_gen.write_interaction_trace(
            &mut cpu_interaction_tree,
            &relations::VerifyInstruction::dummy(),
            &relations::MemoryAddressToId::dummy(),
            &relations::MemoryIdToBig::dummy(),
            &relations::RangeCheck_9_9::dummy(),
            &relations::RangeCheck_9_9_B::dummy(),
            &relations::RangeCheck_9_9_C::dummy(),
            &relations::RangeCheck_9_9_D::dummy(),
            &relations::RangeCheck_9_9_E::dummy(),
            &relations::RangeCheck_9_9_F::dummy(),
            &relations::RangeCheck_9_9_G::dummy(),
            &relations::RangeCheck_9_9_H::dummy(),
            &relations::RangeCheck_19_H::dummy(),
            &relations::RangeCheck_19::dummy(),
            &relations::RangeCheck_19_B::dummy(),
            &relations::RangeCheck_19_C::dummy(),
            &relations::RangeCheck_19_D::dummy(),
            &relations::RangeCheck_19_E::dummy(),
            &relations::RangeCheck_19_F::dummy(),
            &relations::RangeCheck_19_G::dummy(),
            &relations::RangeCheck_18::dummy(),
            &relations::RangeCheck_11::dummy(),
            &relations::Opcodes::dummy(),
        );
        cpu_interaction_tree.finalize_interaction();

        let cpu_trace = cpu_mock.trace_domain_evaluations();

        // ---------- CUDA TRACE GENERATION ----------
        let cuda_generic_opcode_gen = CudaClaimGenerator::new(
            input_state.casm_states_by_opcode.generic_opcode.clone(),
        );

        let mut cuda_memory_address_to_id_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let cuda_memory_id_to_big_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);
        let cuda_verify_instruction_generator = verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());

        let simd_memory_address_to_id = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let simd_memory_id_to_big = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let simd_verify_instruction = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());

        let cuda_range_check_9_9 = range_check_9_9::ClaimGenerator::new();
        let cuda_range_check_9_9_b = range_check_9_9_b::ClaimGenerator::new();
        let cuda_range_check_9_9_c = range_check_9_9_c::ClaimGenerator::new();
        let cuda_range_check_9_9_d = range_check_9_9_d::ClaimGenerator::new();
        let cuda_range_check_9_9_e = range_check_9_9_e::ClaimGenerator::new();
        let cuda_range_check_9_9_f = range_check_9_9_f::ClaimGenerator::new();
        let cuda_range_check_9_9_g = range_check_9_9_g::ClaimGenerator::new();
        let cuda_range_check_9_9_h = range_check_9_9_h::ClaimGenerator::new();
        let cuda_range_check_19 = range_check_19::ClaimGenerator::new();
        let cuda_range_check_19_b = range_check_19_b::ClaimGenerator::new();
        let cuda_range_check_19_c = range_check_19_c::ClaimGenerator::new();
        let cuda_range_check_19_d = range_check_19_d::ClaimGenerator::new();
        let cuda_range_check_19_e = range_check_19_e::ClaimGenerator::new();
        let cuda_range_check_19_f = range_check_19_f::ClaimGenerator::new();
        let cuda_range_check_19_g = range_check_19_g::ClaimGenerator::new();
        let cuda_range_check_19_h = range_check_19_h::ClaimGenerator::new();
        let cuda_range_check_18 = range_check_18::ClaimGenerator::new();
        let cuda_range_check_11 = range_check_11::ClaimGenerator::new();

        // Yield public memory
        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = cuda_memory_address_to_id_generator.get_id(addr);
            cuda_memory_address_to_id_generator.add_cuda_input(&addr);
            cuda_memory_id_to_big_generator.add_cuda_input(&id);
        }

        let mut cuda_mock = MockCommitmentScheme::default();
        let cuda_preprocessed = testing_preprocessed_tree(4);
        let mut cuda_tree_builder = cuda_mock.tree_builder();
        cuda_tree_builder.extend_evals(cuda_preprocessed.gen_trace());
        cuda_tree_builder.finalize_interaction();

        let mut cuda_tree_builder = cuda_mock.tree_builder();
        let (_cuda_claim, cuda_interaction_gen) = cuda_generic_opcode_gen.write_trace(
            &mut cuda_tree_builder,
            &mut cuda_memory_address_to_id_generator,
            &cuda_memory_id_to_big_generator,
            &cuda_verify_instruction_generator,
            &simd_memory_address_to_id,
            &simd_memory_id_to_big,
            &simd_verify_instruction,
            &cuda_range_check_9_9,
            &cuda_range_check_9_9_b,
            &cuda_range_check_9_9_c,
            &cuda_range_check_9_9_d,
            &cuda_range_check_9_9_e,
            &cuda_range_check_9_9_f,
            &cuda_range_check_9_9_g,
            &cuda_range_check_9_9_h,
            &cuda_range_check_19,
            &cuda_range_check_19_b,
            &cuda_range_check_19_c,
            &cuda_range_check_19_d,
            &cuda_range_check_19_e,
            &cuda_range_check_19_f,
            &cuda_range_check_19_g,
            &cuda_range_check_19_h,
            &cuda_range_check_18,
            &cuda_range_check_11,
        );
        cuda_tree_builder.finalize_interaction();

        // Interaction trace for CUDA
        let mut cuda_interaction_tree = cuda_mock.tree_builder();
        let cuda_interaction_claim = cuda_interaction_gen.write_interaction_trace(
            &mut cuda_interaction_tree,
            &relations::VerifyInstruction::dummy(),
            &relations::MemoryAddressToId::dummy(),
            &relations::MemoryIdToBig::dummy(),
            &relations::RangeCheck_9_9::dummy(),
            &relations::RangeCheck_9_9_B::dummy(),
            &relations::RangeCheck_9_9_C::dummy(),
            &relations::RangeCheck_9_9_D::dummy(),
            &relations::RangeCheck_9_9_E::dummy(),
            &relations::RangeCheck_9_9_F::dummy(),
            &relations::RangeCheck_9_9_G::dummy(),
            &relations::RangeCheck_9_9_H::dummy(),
            &relations::RangeCheck_19_H::dummy(),
            &relations::RangeCheck_19::dummy(),
            &relations::RangeCheck_19_B::dummy(),
            &relations::RangeCheck_19_C::dummy(),
            &relations::RangeCheck_19_D::dummy(),
            &relations::RangeCheck_19_E::dummy(),
            &relations::RangeCheck_19_F::dummy(),
            &relations::RangeCheck_19_G::dummy(),
            &relations::RangeCheck_18::dummy(),
            &relations::RangeCheck_11::dummy(),
            &relations::Opcodes::dummy(),
        );
        cuda_interaction_tree.finalize_interaction();

        let cuda_trace = cuda_mock.trace_domain_evaluations();

        println!("CPU claimed_sum: {:?}", cpu_interaction_claim.claimed_sum);
        println!("CUDA claimed_sum: {:?}", cuda_interaction_claim.claimed_sum);

        // Compare traces - Tree 0 is preprocessed, Tree 1 is base trace, Tree 2 is interaction trace
        println!("\n=== Comparing Base Trace (Tree 1) ===");
        let cpu_base = &cpu_trace[1];
        let cuda_base = &cuda_trace[1];

        let mut base_mismatches = 0;
        for (col_idx, (cpu_col, cuda_col)) in cpu_base.iter().zip(cuda_base.iter()).enumerate() {
            let cpu_data = cpu_col.to_cpu();
            let cuda_data = cuda_col.to_cpu();
            for (row, (c, g)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
                if c != g {
                    if base_mismatches < 10 {
                        println!("Base trace MISMATCH at col {}, row {}: CPU={:?}, CUDA={:?}", col_idx, row, c, g);
                    }
                    base_mismatches += 1;
                }
            }
        }
        println!("Total base trace mismatches: {}", base_mismatches);

        println!("\n=== Comparing Interaction Trace (Tree 2) ===");
        let cpu_interaction = &cpu_trace[2];
        let cuda_interaction = &cuda_trace[2];

        let mut interaction_mismatches = 0;
        for (col_idx, (cpu_col, cuda_col)) in cpu_interaction.iter().zip(cuda_interaction.iter()).enumerate() {
            let cpu_data = cpu_col.to_cpu();
            let cuda_data = cuda_col.to_cpu();
            for (row, (c, g)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
                if c != g {
                    if interaction_mismatches < 10 {
                        println!("Interaction trace MISMATCH at col {}, row {}: CPU={:?}, CUDA={:?}", col_idx, row, c, g);
                    }
                    interaction_mismatches += 1;
                }
            }
        }
        println!("Total interaction trace mismatches: {}", interaction_mismatches);

        assert_eq!(base_mismatches, 0, "Base trace has mismatches between CPU and CUDA");
        assert_eq!(interaction_mismatches, 0, "Interaction trace has mismatches between CPU and CUDA");
    }
}
