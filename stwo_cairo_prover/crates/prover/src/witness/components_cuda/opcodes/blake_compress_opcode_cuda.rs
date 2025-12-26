#![allow(unused_parens)]
use cairo_air::components::blake_compress_opcode::{Claim, InteractionClaim};
use stwo::prover::backend::cuda::CudaBackend;

use crate::witness::components_cuda::{
    blake_round_cuda, memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_7_2_5_cuda, triple_xor_32_cuda,
    verify_bitwise_xor_8_cuda, verify_instruction_cuda,
};
use crate::witness::components::{blake_round, memory_address_to_id, memory_id_to_big, range_check_7_2_5, triple_xor_32, verify_bitwise_xor_8, verify_instruction};
use crate::witness::prelude::*;
pub const N_TRACE_COLUMNS: usize = 174;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 37;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
pub type CudaPackedInputs = [BaseFieldVec; 3];
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
use itertools::Itertools;
use stwo::prover::backend::Col;
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::bindings_airs;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};


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
macro_rules! init_subcomponent_uint32_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| std::array::from_fn(|_| Uint32Vec::new_uninitialized(1 << $log_size)))
    };
}

fn init_blake_subcomponent_array<const N: usize>(log_size: u32) -> [blake_round_cuda::CudaPackedInputType; N] {
    unsafe {
        std::array::from_fn(|_| (
            BaseFieldVec::uninitialized(1 << log_size),
            BaseFieldVec::uninitialized(1 << log_size),
            (
                std::array::from_fn(|_| Uint32Vec::new_uninitialized(1 << log_size)),
                BaseFieldVec::uninitialized(1 << log_size)
            )
        ))
    }
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

fn collect_blake_sub_input_ptrs(arr: &[blake_round_cuda::CudaPackedInputType]) -> Vec<*const u32> {
    arr.iter()
        .flat_map(|(a, b, (arr16, c))| {
            std::iter::once(a.device_ptr)
                .chain(std::iter::once(b.device_ptr))
                .chain(arr16.iter().map(|u| u.device_ptr))
                .chain(std::iter::once(c.device_ptr))
        })
        .collect()
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
        blake_round_state: &mut blake_round_cuda::CudaClaimGenerator,
        memory_address_to_id_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
        triple_xor_32_state: &mut triple_xor_32_cuda::CudaClaimGenerator,
        verify_bitwise_xor_8_state: &verify_bitwise_xor_8_cuda::CudaClaimGenerator,
        verify_instruction_state: &verify_instruction_cuda::CudaClaimGenerator,
        // Also pass SIMD generators for multiplicity tracking (needed for final memory traces)
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        verify_instruction_simd_state: &verify_instruction::ClaimGenerator,
        blake_round_simd_state: &mut blake_round::ClaimGenerator,
        triple_xor_32_simd_state: &mut triple_xor_32::ClaimGenerator,
        verify_bitwise_xor_8_simd_state: &verify_bitwise_xor_8::ClaimGenerator,
        range_check_7_2_5_simd_state: &range_check_7_2_5::ClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let size = self.inputs[0].size;
        let log_size = size.ilog2();
        let packed_inputs = self.inputs;

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            self.n_rows,
            packed_inputs,
            blake_round_state,
            memory_address_to_id_state,
            memory_id_to_big_state,
            range_check_7_2_5_state,
            triple_xor_32_state,
            verify_bitwise_xor_8_state,
            verify_instruction_state,
        );

        verify_instruction_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);
        range_check_7_2_5_state.add_cuda_inputs(&sub_component_inputs.range_check_7_2_5);
        verify_bitwise_xor_8_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
        triple_xor_32_state.add_cuda_inputs(&sub_component_inputs.triple_xor_32);
        blake_round_state.add_cuda_inputs(&sub_component_inputs.blake_round);

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
        // blake_round: (BaseFieldVec, BaseFieldVec, ([Uint32Vec; 16], BaseFieldVec))
        // -> (PackedM31, PackedM31, ([PackedUInt32; 16], PackedM31))
        for input_tuple in &sub_component_inputs.blake_round {
            let field0: Vec<M31> = input_tuple.0.to_vec();
            let field1: Vec<M31> = input_tuple.1.to_vec();
            let arr16: [Vec<u32>; 16] = std::array::from_fn(|i| input_tuple.2.0[i].to_vec());
            let field2_1: Vec<M31> = input_tuple.2.1.to_vec();

            // Convert to packed format (chunks of N_LANES)
            let packed_inputs: Vec<blake_round::PackedInputType> = (0..padded_size / N_LANES)
                .map(|chunk_idx| {
                    let start = chunk_idx * N_LANES;
                    let packed0 = PackedM31::from_array(
                        std::array::from_fn(|i| field0[start + i])
                    );
                    let packed1 = PackedM31::from_array(
                        std::array::from_fn(|i| field1[start + i])
                    );
                    let packed_arr16: [PackedUInt32; 16] = std::array::from_fn(|arr_idx| {
                        PackedUInt32::from_array(
                            std::array::from_fn(|i| UInt32::from(arr16[arr_idx][start + i]))
                        )
                    });
                    let packed2_1 = PackedM31::from_array(
                        std::array::from_fn(|i| field2_1[start + i])
                    );
                    (packed0, packed1, (packed_arr16, packed2_1))
                })
                .collect();
            blake_round_simd_state.add_packed_inputs(&packed_inputs);
        }
        // triple_xor_32: [Uint32Vec; 3] -> [PackedUInt32; 3]
        for input_arr in &sub_component_inputs.triple_xor_32 {
            let arr3: [Vec<u32>; 3] = std::array::from_fn(|i| input_arr[i].to_vec());

            // Convert to packed format (chunks of N_LANES)
            let packed_inputs: Vec<triple_xor_32::PackedInputType> = (0..padded_size / N_LANES)
                .map(|chunk_idx| {
                    let start = chunk_idx * N_LANES;
                    std::array::from_fn(|arr_idx| {
                        PackedUInt32::from_array(
                            std::array::from_fn(|i| UInt32::from(arr3[arr_idx][start + i]))
                        )
                    })
                })
                .collect();
            triple_xor_32_simd_state.add_packed_inputs(&packed_inputs);
        }
        // verify_bitwise_xor_8: [BaseFieldVec; 3] -> [PackedM31; 3]
        for input_arr in &sub_component_inputs.verify_bitwise_xor_8 {
            let arr3: [Vec<M31>; 3] = std::array::from_fn(|i| input_arr[i].to_vec());

            // Convert to packed format (chunks of N_LANES)
            let packed_inputs: Vec<verify_bitwise_xor_8::PackedInputType> = (0..padded_size / N_LANES)
                .map(|chunk_idx| {
                    let start = chunk_idx * N_LANES;
                    std::array::from_fn(|arr_idx| {
                        PackedM31::from_array(
                            std::array::from_fn(|i| arr3[arr_idx][start + i])
                        )
                    })
                })
                .collect();
            verify_bitwise_xor_8_simd_state.add_packed_inputs(&packed_inputs);
        }
        // range_check_7_2_5: [BaseFieldVec; 3] -> [PackedM31; 3]
        for input_arr in &sub_component_inputs.range_check_7_2_5 {
            let arr3: [Vec<M31>; 3] = std::array::from_fn(|i| input_arr[i].to_vec());

            // Convert to packed format (chunks of N_LANES)
            let packed_inputs: Vec<range_check_7_2_5::PackedInputType> = (0..padded_size / N_LANES)
                .map(|chunk_idx| {
                    let start = chunk_idx * N_LANES;
                    std::array::from_fn(|arr_idx| {
                        PackedM31::from_array(
                            std::array::from_fn(|i| arr3[arr_idx][start + i])
                        )
                    })
                })
                .collect();
            range_check_7_2_5_simd_state.add_packed_inputs(&packed_inputs);
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
    memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 20],
    memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 20],
    range_check_7_2_5: [range_check_7_2_5_cuda::CudaPackedInputType; 17],
    verify_bitwise_xor_8: [verify_bitwise_xor_8_cuda::CudaPackedInputType; 4],
    blake_round: [blake_round_cuda::CudaPackedInputType; 10],
    triple_xor_32: [triple_xor_32_cuda::CudaPackedInputType; 8],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    n_rows: usize,
    inputs: CudaPackedInputs,
    blake_round_state: &blake_round_cuda::CudaClaimGenerator,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    range_check_7_2_5_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
    triple_xor_32_state: &triple_xor_32_cuda::CudaClaimGenerator,
    verify_bitwise_xor_8_state: &verify_bitwise_xor_8_cuda::CudaClaimGenerator,
    verify_instruction_state: &verify_instruction_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let log_size = inputs[0].size.ilog2();
    // Initialize all trace, lookup, subcomponent arrays
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                blake_round_0:     init_lookup_array!(log_size),
                blake_round_1:     init_lookup_array!(log_size),
                memory_address_to_id_0:  init_lookup_array!(log_size),
                memory_address_to_id_1:  init_lookup_array!(log_size),
                memory_address_to_id_2:  init_lookup_array!(log_size),
                memory_address_to_id_3:  init_lookup_array!(log_size),
                memory_address_to_id_4:  init_lookup_array!(log_size),
                memory_address_to_id_5:  init_lookup_array!(log_size),
                memory_address_to_id_6:  init_lookup_array!(log_size),
                memory_address_to_id_7:  init_lookup_array!(log_size),
                memory_address_to_id_8:  init_lookup_array!(log_size),
                memory_address_to_id_9:  init_lookup_array!(log_size),
                memory_address_to_id_10: init_lookup_array!(log_size),
                memory_address_to_id_11: init_lookup_array!(log_size),
                memory_address_to_id_12: init_lookup_array!(log_size),
                memory_address_to_id_13: init_lookup_array!(log_size),
                memory_address_to_id_14: init_lookup_array!(log_size),
                memory_address_to_id_15: init_lookup_array!(log_size),
                memory_address_to_id_16: init_lookup_array!(log_size),
                memory_address_to_id_17: init_lookup_array!(log_size),
                memory_address_to_id_18: init_lookup_array!(log_size),
                memory_address_to_id_19: init_lookup_array!(log_size),
                memory_id_to_big_0:  init_lookup_array!(log_size),
                memory_id_to_big_1:  init_lookup_array!(log_size),
                memory_id_to_big_2:  init_lookup_array!(log_size),
                memory_id_to_big_3:  init_lookup_array!(log_size),
                memory_id_to_big_4:  init_lookup_array!(log_size),
                memory_id_to_big_5:  init_lookup_array!(log_size),
                memory_id_to_big_6:  init_lookup_array!(log_size),
                memory_id_to_big_7:  init_lookup_array!(log_size),
                memory_id_to_big_8:  init_lookup_array!(log_size),
                memory_id_to_big_9:  init_lookup_array!(log_size),
                memory_id_to_big_10: init_lookup_array!(log_size),
                memory_id_to_big_11: init_lookup_array!(log_size),
                memory_id_to_big_12: init_lookup_array!(log_size),
                memory_id_to_big_13: init_lookup_array!(log_size),
                memory_id_to_big_14: init_lookup_array!(log_size),
                memory_id_to_big_15: init_lookup_array!(log_size),
                memory_id_to_big_16: init_lookup_array!(log_size),
                memory_id_to_big_17: init_lookup_array!(log_size),
                memory_id_to_big_18: init_lookup_array!(log_size),
                memory_id_to_big_19: init_lookup_array!(log_size),
                opcodes_0:  init_lookup_array!(log_size),
                opcodes_1:  init_lookup_array!(log_size),
                range_check_7_2_5_0:  init_lookup_array!(log_size),
                range_check_7_2_5_1:  init_lookup_array!(log_size),
                range_check_7_2_5_2:  init_lookup_array!(log_size),
                range_check_7_2_5_3:  init_lookup_array!(log_size),
                range_check_7_2_5_4:  init_lookup_array!(log_size),
                range_check_7_2_5_5:  init_lookup_array!(log_size),
                range_check_7_2_5_6:  init_lookup_array!(log_size),
                range_check_7_2_5_7:  init_lookup_array!(log_size),
                range_check_7_2_5_8:  init_lookup_array!(log_size),
                range_check_7_2_5_9:  init_lookup_array!(log_size),
                range_check_7_2_5_10: init_lookup_array!(log_size),
                range_check_7_2_5_11: init_lookup_array!(log_size),
                range_check_7_2_5_12: init_lookup_array!(log_size),
                range_check_7_2_5_13: init_lookup_array!(log_size),
                range_check_7_2_5_14: init_lookup_array!(log_size),
                range_check_7_2_5_15: init_lookup_array!(log_size),
                range_check_7_2_5_16: init_lookup_array!(log_size),
                triple_xor_32_0:  init_lookup_array!(log_size),
                triple_xor_32_1:  init_lookup_array!(log_size),
                triple_xor_32_2:  init_lookup_array!(log_size),
                triple_xor_32_3:  init_lookup_array!(log_size),
                triple_xor_32_4:  init_lookup_array!(log_size),
                triple_xor_32_5:  init_lookup_array!(log_size),
                triple_xor_32_6:  init_lookup_array!(log_size),
                triple_xor_32_7:  init_lookup_array!(log_size),
                verify_bitwise_xor_8_0: init_lookup_array!(log_size),
                verify_bitwise_xor_8_1: init_lookup_array!(log_size),
                verify_bitwise_xor_8_2: init_lookup_array!(log_size),
                verify_bitwise_xor_8_3: init_lookup_array!(log_size),
                verify_instruction_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_instruction: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                range_check_7_2_5: init_subcomponent_basefield_array!(log_size),
                verify_bitwise_xor_8: init_subcomponent_basefield_array!(log_size),
                blake_round: init_blake_subcomponent_array(log_size),
                triple_xor_32: init_subcomponent_uint32_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    macro_rules! ptrs {
        ($name:ident) => { collect_lookup_ptrs!(lookup_data, $name) }
    }
    macro_rules! sub_ptrs {
        ($name:ident) => { collect_sub_input_ptrs!(sub_component_inputs, $name) }
    }
    let lookup_blake_round_0         = ptrs!(blake_round_0);
    let lookup_blake_round_1         = ptrs!(blake_round_1);
    let lookup_memory_address_to_id_0 = ptrs!(memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = ptrs!(memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = ptrs!(memory_address_to_id_2);
    let lookup_memory_address_to_id_3 = ptrs!(memory_address_to_id_3);
    let lookup_memory_address_to_id_4 = ptrs!(memory_address_to_id_4);
    let lookup_memory_address_to_id_5 = ptrs!(memory_address_to_id_5);
    let lookup_memory_address_to_id_6 = ptrs!(memory_address_to_id_6);
    let lookup_memory_address_to_id_7 = ptrs!(memory_address_to_id_7);
    let lookup_memory_address_to_id_8 = ptrs!(memory_address_to_id_8);
    let lookup_memory_address_to_id_9 = ptrs!(memory_address_to_id_9);
    let lookup_memory_address_to_id_10 = ptrs!(memory_address_to_id_10);
    let lookup_memory_address_to_id_11 = ptrs!(memory_address_to_id_11);
    let lookup_memory_address_to_id_12 = ptrs!(memory_address_to_id_12);
    let lookup_memory_address_to_id_13 = ptrs!(memory_address_to_id_13);
    let lookup_memory_address_to_id_14 = ptrs!(memory_address_to_id_14);
    let lookup_memory_address_to_id_15 = ptrs!(memory_address_to_id_15);
    let lookup_memory_address_to_id_16 = ptrs!(memory_address_to_id_16);
    let lookup_memory_address_to_id_17 = ptrs!(memory_address_to_id_17);
    let lookup_memory_address_to_id_18 = ptrs!(memory_address_to_id_18);
    let lookup_memory_address_to_id_19 = ptrs!(memory_address_to_id_19);

    let lookup_memory_id_to_big_0 = ptrs!(memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = ptrs!(memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = ptrs!(memory_id_to_big_2);
    let lookup_memory_id_to_big_3 = ptrs!(memory_id_to_big_3);
    let lookup_memory_id_to_big_4 = ptrs!(memory_id_to_big_4);
    let lookup_memory_id_to_big_5 = ptrs!(memory_id_to_big_5);
    let lookup_memory_id_to_big_6 = ptrs!(memory_id_to_big_6);
    let lookup_memory_id_to_big_7 = ptrs!(memory_id_to_big_7);
    let lookup_memory_id_to_big_8 = ptrs!(memory_id_to_big_8);
    let lookup_memory_id_to_big_9 = ptrs!(memory_id_to_big_9);
    let lookup_memory_id_to_big_10 = ptrs!(memory_id_to_big_10);
    let lookup_memory_id_to_big_11 = ptrs!(memory_id_to_big_11);
    let lookup_memory_id_to_big_12 = ptrs!(memory_id_to_big_12);
    let lookup_memory_id_to_big_13 = ptrs!(memory_id_to_big_13);
    let lookup_memory_id_to_big_14 = ptrs!(memory_id_to_big_14);
    let lookup_memory_id_to_big_15 = ptrs!(memory_id_to_big_15);
    let lookup_memory_id_to_big_16 = ptrs!(memory_id_to_big_16);
    let lookup_memory_id_to_big_17 = ptrs!(memory_id_to_big_17);
    let lookup_memory_id_to_big_18 = ptrs!(memory_id_to_big_18);
    let lookup_memory_id_to_big_19 = ptrs!(memory_id_to_big_19);

    let lookup_opcodes_0 = ptrs!(opcodes_0);
    let lookup_opcodes_1 = ptrs!(opcodes_1);

    let lookup_range_check_7_2_5_0 = ptrs!(range_check_7_2_5_0);
    let lookup_range_check_7_2_5_1 = ptrs!(range_check_7_2_5_1);
    let lookup_range_check_7_2_5_2 = ptrs!(range_check_7_2_5_2);
    let lookup_range_check_7_2_5_3 = ptrs!(range_check_7_2_5_3);
    let lookup_range_check_7_2_5_4 = ptrs!(range_check_7_2_5_4);
    let lookup_range_check_7_2_5_5 = ptrs!(range_check_7_2_5_5);
    let lookup_range_check_7_2_5_6 = ptrs!(range_check_7_2_5_6);
    let lookup_range_check_7_2_5_7 = ptrs!(range_check_7_2_5_7);
    let lookup_range_check_7_2_5_8 = ptrs!(range_check_7_2_5_8);
    let lookup_range_check_7_2_5_9 = ptrs!(range_check_7_2_5_9);
    let lookup_range_check_7_2_5_10 = ptrs!(range_check_7_2_5_10);
    let lookup_range_check_7_2_5_11 = ptrs!(range_check_7_2_5_11);
    let lookup_range_check_7_2_5_12 = ptrs!(range_check_7_2_5_12);
    let lookup_range_check_7_2_5_13 = ptrs!(range_check_7_2_5_13);
    let lookup_range_check_7_2_5_14 = ptrs!(range_check_7_2_5_14);
    let lookup_range_check_7_2_5_15 = ptrs!(range_check_7_2_5_15);
    let lookup_range_check_7_2_5_16 = ptrs!(range_check_7_2_5_16);

    let lookup_triple_xor_32_0 = ptrs!(triple_xor_32_0);
    let lookup_triple_xor_32_1 = ptrs!(triple_xor_32_1);
    let lookup_triple_xor_32_2 = ptrs!(triple_xor_32_2);
    let lookup_triple_xor_32_3 = ptrs!(triple_xor_32_3);
    let lookup_triple_xor_32_4 = ptrs!(triple_xor_32_4);
    let lookup_triple_xor_32_5 = ptrs!(triple_xor_32_5);
    let lookup_triple_xor_32_6 = ptrs!(triple_xor_32_6);
    let lookup_triple_xor_32_7 = ptrs!(triple_xor_32_7);

    let lookup_verify_bitwise_xor_8_0 = ptrs!(verify_bitwise_xor_8_0);
    let lookup_verify_bitwise_xor_8_1 = ptrs!(verify_bitwise_xor_8_1);
    let lookup_verify_bitwise_xor_8_2 = ptrs!(verify_bitwise_xor_8_2);
    let lookup_verify_bitwise_xor_8_3 = ptrs!(verify_bitwise_xor_8_3);

    let lookup_verify_instruction_0 = ptrs!(verify_instruction_0);

    // sub_component_inputs
    let sub_verify_instruction_vec = sub_ptrs!(verify_instruction);
    let sub_memory_address_to_id_vec = sub_ptrs!(memory_address_to_id);
    let sub_memory_id_to_big_vec = sub_ptrs!(memory_id_to_big);
    let sub_range_check_7_2_5_vec = sub_ptrs!(range_check_7_2_5);
    let sub_verify_bitwise_xor_8_vec = sub_ptrs!(verify_bitwise_xor_8);
    let sub_blake_round_vec = collect_blake_sub_input_ptrs(&sub_component_inputs.blake_round);
    let sub_triple_xor_32_vec = sub_ptrs!(triple_xor_32);

    // inputs
    let opcodes_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_blake_compress_opcode_traces(
            traces_vec.as_ptr(),
            // All lookup ptrs
            lookup_blake_round_0.as_ptr(),
            lookup_blake_round_1.as_ptr(),
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            lookup_memory_address_to_id_3.as_ptr(),
            lookup_memory_address_to_id_4.as_ptr(),
            lookup_memory_address_to_id_5.as_ptr(),
            lookup_memory_address_to_id_6.as_ptr(),
            lookup_memory_address_to_id_7.as_ptr(),
            lookup_memory_address_to_id_8.as_ptr(),
            lookup_memory_address_to_id_9.as_ptr(),
            lookup_memory_address_to_id_10.as_ptr(),
            lookup_memory_address_to_id_11.as_ptr(),
            lookup_memory_address_to_id_12.as_ptr(),
            lookup_memory_address_to_id_13.as_ptr(),
            lookup_memory_address_to_id_14.as_ptr(),
            lookup_memory_address_to_id_15.as_ptr(),
            lookup_memory_address_to_id_16.as_ptr(),
            lookup_memory_address_to_id_17.as_ptr(),
            lookup_memory_address_to_id_18.as_ptr(),
            lookup_memory_address_to_id_19.as_ptr(),
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            lookup_memory_id_to_big_3.as_ptr(),
            lookup_memory_id_to_big_4.as_ptr(),
            lookup_memory_id_to_big_5.as_ptr(),
            lookup_memory_id_to_big_6.as_ptr(),
            lookup_memory_id_to_big_7.as_ptr(),
            lookup_memory_id_to_big_8.as_ptr(),
            lookup_memory_id_to_big_9.as_ptr(),
            lookup_memory_id_to_big_10.as_ptr(),
            lookup_memory_id_to_big_11.as_ptr(),
            lookup_memory_id_to_big_12.as_ptr(),
            lookup_memory_id_to_big_13.as_ptr(),
            lookup_memory_id_to_big_14.as_ptr(),
            lookup_memory_id_to_big_15.as_ptr(),
            lookup_memory_id_to_big_16.as_ptr(),
            lookup_memory_id_to_big_17.as_ptr(),
            lookup_memory_id_to_big_18.as_ptr(),
            lookup_memory_id_to_big_19.as_ptr(),
            lookup_opcodes_0.as_ptr(),
            lookup_opcodes_1.as_ptr(),
            lookup_range_check_7_2_5_0.as_ptr(),
            lookup_range_check_7_2_5_1.as_ptr(),
            lookup_range_check_7_2_5_2.as_ptr(),
            lookup_range_check_7_2_5_3.as_ptr(),
            lookup_range_check_7_2_5_4.as_ptr(),
            lookup_range_check_7_2_5_5.as_ptr(),
            lookup_range_check_7_2_5_6.as_ptr(),
            lookup_range_check_7_2_5_7.as_ptr(),
            lookup_range_check_7_2_5_8.as_ptr(),
            lookup_range_check_7_2_5_9.as_ptr(),
            lookup_range_check_7_2_5_10.as_ptr(),
            lookup_range_check_7_2_5_11.as_ptr(),
            lookup_range_check_7_2_5_12.as_ptr(),
            lookup_range_check_7_2_5_13.as_ptr(),
            lookup_range_check_7_2_5_14.as_ptr(),
            lookup_range_check_7_2_5_15.as_ptr(),
            lookup_range_check_7_2_5_16.as_ptr(),
            lookup_triple_xor_32_0.as_ptr(),
            lookup_triple_xor_32_1.as_ptr(),
            lookup_triple_xor_32_2.as_ptr(),
            lookup_triple_xor_32_3.as_ptr(),
            lookup_triple_xor_32_4.as_ptr(),
            lookup_triple_xor_32_5.as_ptr(),
            lookup_triple_xor_32_6.as_ptr(),
            lookup_triple_xor_32_7.as_ptr(),
            lookup_verify_bitwise_xor_8_0.as_ptr(),
            lookup_verify_bitwise_xor_8_1.as_ptr(),
            lookup_verify_bitwise_xor_8_2.as_ptr(),
            lookup_verify_bitwise_xor_8_3.as_ptr(),
            lookup_verify_instruction_0.as_ptr(),
            // subcomponent ptrs
            sub_verify_instruction_vec.as_ptr(),
            sub_memory_address_to_id_vec.as_ptr(),
            sub_memory_id_to_big_vec.as_ptr(),
            sub_range_check_7_2_5_vec.as_ptr(),
            sub_verify_bitwise_xor_8_vec.as_ptr(),
            sub_blake_round_vec.as_ptr(),
            sub_triple_xor_32_vec.as_ptr(),
            // inputs
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

#[repr(C)]
struct CudaLookupData {
    blake_round_0: [BaseFieldVec; 35],
    blake_round_1: [BaseFieldVec; 35],
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    memory_address_to_id_3: [BaseFieldVec; 2],
    memory_address_to_id_4: [BaseFieldVec; 2],
    memory_address_to_id_5: [BaseFieldVec; 2],
    memory_address_to_id_6: [BaseFieldVec; 2],
    memory_address_to_id_7: [BaseFieldVec; 2],
    memory_address_to_id_8: [BaseFieldVec; 2],
    memory_address_to_id_9: [BaseFieldVec; 2],
    memory_address_to_id_10: [BaseFieldVec; 2],
    memory_address_to_id_11: [BaseFieldVec; 2],
    memory_address_to_id_12: [BaseFieldVec; 2],
    memory_address_to_id_13: [BaseFieldVec; 2],
    memory_address_to_id_14: [BaseFieldVec; 2],
    memory_address_to_id_15: [BaseFieldVec; 2],
    memory_address_to_id_16: [BaseFieldVec; 2],
    memory_address_to_id_17: [BaseFieldVec; 2],
    memory_address_to_id_18: [BaseFieldVec; 2],
    memory_address_to_id_19: [BaseFieldVec; 2],
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    memory_id_to_big_3: [BaseFieldVec; 29],
    memory_id_to_big_4: [BaseFieldVec; 29],
    memory_id_to_big_5: [BaseFieldVec; 29],
    memory_id_to_big_6: [BaseFieldVec; 29],
    memory_id_to_big_7: [BaseFieldVec; 29],
    memory_id_to_big_8: [BaseFieldVec; 29],
    memory_id_to_big_9: [BaseFieldVec; 29],
    memory_id_to_big_10: [BaseFieldVec; 29],
    memory_id_to_big_11: [BaseFieldVec; 29],
    memory_id_to_big_12: [BaseFieldVec; 29],
    memory_id_to_big_13: [BaseFieldVec; 29],
    memory_id_to_big_14: [BaseFieldVec; 29],
    memory_id_to_big_15: [BaseFieldVec; 29],
    memory_id_to_big_16: [BaseFieldVec; 29],
    memory_id_to_big_17: [BaseFieldVec; 29],
    memory_id_to_big_18: [BaseFieldVec; 29],
    memory_id_to_big_19: [BaseFieldVec; 29],
    opcodes_0: [BaseFieldVec; 3],
    opcodes_1: [BaseFieldVec; 3],
    range_check_7_2_5_0: [BaseFieldVec; 3],
    range_check_7_2_5_1: [BaseFieldVec; 3],
    range_check_7_2_5_2: [BaseFieldVec; 3],
    range_check_7_2_5_3: [BaseFieldVec; 3],
    range_check_7_2_5_4: [BaseFieldVec; 3],
    range_check_7_2_5_5: [BaseFieldVec; 3],
    range_check_7_2_5_6: [BaseFieldVec; 3],
    range_check_7_2_5_7: [BaseFieldVec; 3],
    range_check_7_2_5_8: [BaseFieldVec; 3],
    range_check_7_2_5_9: [BaseFieldVec; 3],
    range_check_7_2_5_10: [BaseFieldVec; 3],
    range_check_7_2_5_11: [BaseFieldVec; 3],
    range_check_7_2_5_12: [BaseFieldVec; 3],
    range_check_7_2_5_13: [BaseFieldVec; 3],
    range_check_7_2_5_14: [BaseFieldVec; 3],
    range_check_7_2_5_15: [BaseFieldVec; 3],
    range_check_7_2_5_16: [BaseFieldVec; 3],
    triple_xor_32_0: [BaseFieldVec; 8],
    triple_xor_32_1: [BaseFieldVec; 8],
    triple_xor_32_2: [BaseFieldVec; 8],
    triple_xor_32_3: [BaseFieldVec; 8],
    triple_xor_32_4: [BaseFieldVec; 8],
    triple_xor_32_5: [BaseFieldVec; 8],
    triple_xor_32_6: [BaseFieldVec; 8],
    triple_xor_32_7: [BaseFieldVec; 8],
    verify_bitwise_xor_8_0: [BaseFieldVec; 3],
    verify_bitwise_xor_8_1: [BaseFieldVec; 3],
    verify_bitwise_xor_8_2: [BaseFieldVec; 3],
    verify_bitwise_xor_8_3: [BaseFieldVec; 3],
    verify_instruction_0: [BaseFieldVec; 7],
}

#[repr(C)]
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
        range_check_7_2_5: &relations::RangeCheck_7_2_5,
        verify_bitwise_xor_8: &relations::VerifyBitwiseXor_8,
        blake_round: &relations::BlakeRound,
        triple_xor_32: &relations::TripleXor32,
        opcodes: &relations::Opcodes,
    ) ->InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Allocate interaction trace, one Col per column
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect all lookup ptrs
        macro_rules! ptrs {
            ($name:ident) => { collect_lookup_ptrs!(self.lookup_data, $name) }
        }
        let _lookup_blake_round_0         = ptrs!(blake_round_0);
        let _lookup_blake_round_1         = ptrs!(blake_round_1);
        let _lookup_memory_address_to_id_0 = ptrs!(memory_address_to_id_0);
        let _lookup_memory_address_to_id_1 = ptrs!(memory_address_to_id_1);
        let _lookup_memory_address_to_id_2 = ptrs!(memory_address_to_id_2);
        let _lookup_memory_address_to_id_3 = ptrs!(memory_address_to_id_3);
        let _lookup_memory_address_to_id_4 = ptrs!(memory_address_to_id_4);
        let _lookup_memory_address_to_id_5 = ptrs!(memory_address_to_id_5);
        let _lookup_memory_address_to_id_6 = ptrs!(memory_address_to_id_6);
        let _lookup_memory_address_to_id_7 = ptrs!(memory_address_to_id_7);
        let _lookup_memory_address_to_id_8 = ptrs!(memory_address_to_id_8);
        let _lookup_memory_address_to_id_9 = ptrs!(memory_address_to_id_9);
        let _lookup_memory_address_to_id_10 = ptrs!(memory_address_to_id_10);
        let _lookup_memory_address_to_id_11 = ptrs!(memory_address_to_id_11);
        let _lookup_memory_address_to_id_12 = ptrs!(memory_address_to_id_12);
        let _lookup_memory_address_to_id_13 = ptrs!(memory_address_to_id_13);
        let _lookup_memory_address_to_id_14 = ptrs!(memory_address_to_id_14);
        let _lookup_memory_address_to_id_15 = ptrs!(memory_address_to_id_15);
        let _lookup_memory_address_to_id_16 = ptrs!(memory_address_to_id_16);
        let _lookup_memory_address_to_id_17 = ptrs!(memory_address_to_id_17);
        let _lookup_memory_address_to_id_18 = ptrs!(memory_address_to_id_18);
        let _lookup_memory_address_to_id_19 = ptrs!(memory_address_to_id_19);

        let _lookup_memory_id_to_big_0 = ptrs!(memory_id_to_big_0);
        let _lookup_memory_id_to_big_1 = ptrs!(memory_id_to_big_1);
        let _lookup_memory_id_to_big_2 = ptrs!(memory_id_to_big_2);
        let _lookup_memory_id_to_big_3 = ptrs!(memory_id_to_big_3);
        let _lookup_memory_id_to_big_4 = ptrs!(memory_id_to_big_4);
        let _lookup_memory_id_to_big_5 = ptrs!(memory_id_to_big_5);
        let _lookup_memory_id_to_big_6 = ptrs!(memory_id_to_big_6);
        let _lookup_memory_id_to_big_7 = ptrs!(memory_id_to_big_7);
        let _lookup_memory_id_to_big_8 = ptrs!(memory_id_to_big_8);
        let _lookup_memory_id_to_big_9 = ptrs!(memory_id_to_big_9);
        let _lookup_memory_id_to_big_10 = ptrs!(memory_id_to_big_10);
        let _lookup_memory_id_to_big_11 = ptrs!(memory_id_to_big_11);
        let _lookup_memory_id_to_big_12 = ptrs!(memory_id_to_big_12);
        let _lookup_memory_id_to_big_13 = ptrs!(memory_id_to_big_13);
        let _lookup_memory_id_to_big_14 = ptrs!(memory_id_to_big_14);
        let _lookup_memory_id_to_big_15 = ptrs!(memory_id_to_big_15);
        let _lookup_memory_id_to_big_16 = ptrs!(memory_id_to_big_16);
        let _lookup_memory_id_to_big_17 = ptrs!(memory_id_to_big_17);
        let _lookup_memory_id_to_big_18 = ptrs!(memory_id_to_big_18);
        let _lookup_memory_id_to_big_19 = ptrs!(memory_id_to_big_19);

        let _lookup_opcodes_0 = ptrs!(opcodes_0);
        let _lookup_opcodes_1 = ptrs!(opcodes_1);

        let _lookup_range_check_7_2_5_0 = ptrs!(range_check_7_2_5_0);
        let _lookup_range_check_7_2_5_1 = ptrs!(range_check_7_2_5_1);
        let _lookup_range_check_7_2_5_2 = ptrs!(range_check_7_2_5_2);
        let _lookup_range_check_7_2_5_3 = ptrs!(range_check_7_2_5_3);
        let _lookup_range_check_7_2_5_4 = ptrs!(range_check_7_2_5_4);
        let _lookup_range_check_7_2_5_5 = ptrs!(range_check_7_2_5_5);
        let _lookup_range_check_7_2_5_6 = ptrs!(range_check_7_2_5_6);
        let _lookup_range_check_7_2_5_7 = ptrs!(range_check_7_2_5_7);
        let _lookup_range_check_7_2_5_8 = ptrs!(range_check_7_2_5_8);
        let _lookup_range_check_7_2_5_9 = ptrs!(range_check_7_2_5_9);
        let _lookup_range_check_7_2_5_10 = ptrs!(range_check_7_2_5_10);
        let _lookup_range_check_7_2_5_11 = ptrs!(range_check_7_2_5_11);
        let _lookup_range_check_7_2_5_12 = ptrs!(range_check_7_2_5_12);
        let _lookup_range_check_7_2_5_13 = ptrs!(range_check_7_2_5_13);
        let _lookup_range_check_7_2_5_14 = ptrs!(range_check_7_2_5_14);
        let _lookup_range_check_7_2_5_15 = ptrs!(range_check_7_2_5_15);
        let _lookup_range_check_7_2_5_16 = ptrs!(range_check_7_2_5_16);

        let _lookup_triple_xor_32_0 = ptrs!(triple_xor_32_0);
        let _lookup_triple_xor_32_1 = ptrs!(triple_xor_32_1);
        let _lookup_triple_xor_32_2 = ptrs!(triple_xor_32_2);
        let _lookup_triple_xor_32_3 = ptrs!(triple_xor_32_3);
        let _lookup_triple_xor_32_4 = ptrs!(triple_xor_32_4);
        let _lookup_triple_xor_32_5 = ptrs!(triple_xor_32_5);
        let _lookup_triple_xor_32_6 = ptrs!(triple_xor_32_6);
        let _lookup_triple_xor_32_7 = ptrs!(triple_xor_32_7);

        let _lookup_verify_bitwise_xor_8_0 = ptrs!(verify_bitwise_xor_8_0);
        let _lookup_verify_bitwise_xor_8_1 = ptrs!(verify_bitwise_xor_8_1);
        let _lookup_verify_bitwise_xor_8_2 = ptrs!(verify_bitwise_xor_8_2);
        let _lookup_verify_bitwise_xor_8_3 = ptrs!(verify_bitwise_xor_8_3);

        let _lookup_verify_instruction_0 = ptrs!(verify_instruction_0);

        let _interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let blake_round_ptr = blake_round as *const _ as *mut std::os::raw::c_void;
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;
            let opcodes_ptr = opcodes as *const _ as *mut std::os::raw::c_void;
            let range_check_7_2_5_ptr = range_check_7_2_5 as *const _ as *mut std::os::raw::c_void;
            let triple_xor_32_ptr = triple_xor_32 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_8_ptr = verify_bitwise_xor_8 as *const _ as *mut std::os::raw::c_void;
            let verify_instruction_ptr = verify_instruction as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_blake_compress_opcode_interaction_traces(
                blake_round_ptr,
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                opcodes_ptr,
                range_check_7_2_5_ptr,
                triple_xor_32_ptr,
                verify_bitwise_xor_8_ptr,
                verify_instruction_ptr,

                _lookup_blake_round_0.as_ptr(),
                _lookup_blake_round_1.as_ptr(),
                _lookup_memory_address_to_id_0.as_ptr(),
                _lookup_memory_address_to_id_1.as_ptr(),
                _lookup_memory_address_to_id_2.as_ptr(),
                _lookup_memory_address_to_id_3.as_ptr(),
                _lookup_memory_address_to_id_4.as_ptr(),
                _lookup_memory_address_to_id_5.as_ptr(),
                _lookup_memory_address_to_id_6.as_ptr(),
                _lookup_memory_address_to_id_7.as_ptr(),
                _lookup_memory_address_to_id_8.as_ptr(),
                _lookup_memory_address_to_id_9.as_ptr(),
                _lookup_memory_address_to_id_10.as_ptr(),
                _lookup_memory_address_to_id_11.as_ptr(),
                _lookup_memory_address_to_id_12.as_ptr(),
                _lookup_memory_address_to_id_13.as_ptr(),
                _lookup_memory_address_to_id_14.as_ptr(),
                _lookup_memory_address_to_id_15.as_ptr(),
                _lookup_memory_address_to_id_16.as_ptr(),
                _lookup_memory_address_to_id_17.as_ptr(),
                _lookup_memory_address_to_id_18.as_ptr(),
                _lookup_memory_address_to_id_19.as_ptr(),
                _lookup_memory_id_to_big_0.as_ptr(),
                _lookup_memory_id_to_big_1.as_ptr(),
                _lookup_memory_id_to_big_2.as_ptr(),
                _lookup_memory_id_to_big_3.as_ptr(),
                _lookup_memory_id_to_big_4.as_ptr(),
                _lookup_memory_id_to_big_5.as_ptr(),
                _lookup_memory_id_to_big_6.as_ptr(),
                _lookup_memory_id_to_big_7.as_ptr(),
                _lookup_memory_id_to_big_8.as_ptr(),
                _lookup_memory_id_to_big_9.as_ptr(),
                _lookup_memory_id_to_big_10.as_ptr(),
                _lookup_memory_id_to_big_11.as_ptr(),
                _lookup_memory_id_to_big_12.as_ptr(),
                _lookup_memory_id_to_big_13.as_ptr(),
                _lookup_memory_id_to_big_14.as_ptr(),
                _lookup_memory_id_to_big_15.as_ptr(),
                _lookup_memory_id_to_big_16.as_ptr(),
                _lookup_memory_id_to_big_17.as_ptr(),
                _lookup_memory_id_to_big_18.as_ptr(),
                _lookup_memory_id_to_big_19.as_ptr(),
                _lookup_opcodes_0.as_ptr(),
                _lookup_opcodes_1.as_ptr(),
                _lookup_range_check_7_2_5_0.as_ptr(),
                _lookup_range_check_7_2_5_1.as_ptr(),
                _lookup_range_check_7_2_5_2.as_ptr(),
                _lookup_range_check_7_2_5_3.as_ptr(),
                _lookup_range_check_7_2_5_4.as_ptr(),
                _lookup_range_check_7_2_5_5.as_ptr(),
                _lookup_range_check_7_2_5_6.as_ptr(),
                _lookup_range_check_7_2_5_7.as_ptr(),
                _lookup_range_check_7_2_5_8.as_ptr(),
                _lookup_range_check_7_2_5_9.as_ptr(),
                _lookup_range_check_7_2_5_10.as_ptr(),
                _lookup_range_check_7_2_5_11.as_ptr(),
                _lookup_range_check_7_2_5_12.as_ptr(),
                _lookup_range_check_7_2_5_13.as_ptr(),
                _lookup_range_check_7_2_5_14.as_ptr(),
                _lookup_range_check_7_2_5_15.as_ptr(),
                _lookup_range_check_7_2_5_16.as_ptr(),
                _lookup_triple_xor_32_0.as_ptr(),
                _lookup_triple_xor_32_1.as_ptr(),
                _lookup_triple_xor_32_2.as_ptr(),
                _lookup_triple_xor_32_3.as_ptr(),
                _lookup_triple_xor_32_4.as_ptr(),
                _lookup_triple_xor_32_5.as_ptr(),
                _lookup_triple_xor_32_6.as_ptr(),
                _lookup_triple_xor_32_7.as_ptr(),
                _lookup_verify_bitwise_xor_8_0.as_ptr(),
                _lookup_verify_bitwise_xor_8_1.as_ptr(),
                _lookup_verify_bitwise_xor_8_2.as_ptr(),
                _lookup_verify_bitwise_xor_8_3.as_ptr(),
                _lookup_verify_instruction_0.as_ptr(),

                self.n_rows as u32,
                trace_log_size as u32,
                _interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([
            claimed_sum_vec[0],
            claimed_sum_vec[1],
            claimed_sum_vec[2],
            claimed_sum_vec[3],
        ]);

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
    use crate::witness::components::blake_round;
    use crate::witness::components::range_check_7_2_5;
    use crate::witness::components::triple_xor_32;
    use crate::witness::components::verify_bitwise_xor_8;
    use crate::witness::components_cuda::{
        blake_round_cuda, memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_7_2_5_cuda,
        triple_xor_32_cuda, verify_bitwise_xor_8_cuda, verify_instruction_cuda,
    };
    use super::CudaClaimGenerator;
    use cairo_air::relations;

    use stwo_constraint_framework::TraceLocationAllocator;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::blake_compress_opcode::Eval;
    use cairo_air::components::blake_compress_opcode::Component;
    use crate::witness::components::{memory_id_to_big, memory_address_to_id};
    use stwo::core::fields::m31::M31;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
    use crate::witness::components::blake_compress_opcode;
    use crate::witness::components::verify_instruction;
    use stwo::stwo_cuda::base_field_vec::Uint32Vec;
    use stwo::prover::backend::Column;
    use crate::test_utils::prover_input_from_compiled_cairo_program;

    #[test]
    fn test_blake_compress_opcode_cpu_ref() {
        let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
        let input_state = input.state_transitions;
        assert!(!input_state.casm_states_by_opcode.blake_compress_opcode.is_empty());

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

        let blake_compress = blake_compress_opcode::ClaimGenerator::new(input_state.casm_states_by_opcode.blake_compress_opcode);

        let mut blake_round_trace_generator = blake_round::ClaimGenerator::new(input.memory);
        let mut triple_xor_32_trace_generator = triple_xor_32::ClaimGenerator::new();
        let verify_instruction_trace_generator = verify_instruction::ClaimGenerator::new(input.inst_cache);
        let range_check_7_2_5_trace_generator = range_check_7_2_5::ClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let blake_round_relation = relations::BlakeRound::dummy();
        let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();
        let triple_xor_32_relation = relations::TripleXor32::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let opcodes_relation = relations::Opcodes::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_compress_claim, blake_compress_interaction_gen) = blake_compress.write_trace(
                    &mut mock_tree_builder,
                    &mut blake_round_trace_generator,
                    &memory_address_to_id_trace_generator,
                    &memory_id_to_big_trace_generator,
                    &range_check_7_2_5_trace_generator,
                    &mut triple_xor_32_trace_generator,
                    &verify_bitwise_xor_8_trace_generator,
                    &verify_instruction_trace_generator,
                );
        mock_tree_builder.finalize_interaction();

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_compress_interaction_claim = blake_compress_interaction_gen.write_interaction_trace(
                    &mut mock_tree_builder,
                    &verify_instruction_relation,
                    &memory_address_to_id_relation,
                    &memory_id_to_big_relation,
                    &range_check_7_2_5_relation,
                    &verify_bitwise_xor_8_relation,
                    &blake_round_relation,
                    &triple_xor_32_relation,
                    &opcodes_relation,
                );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let blake_compress_components = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_compress_opcode"),
                claim: blake_compress_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                blake_round_lookup_elements: relations::BlakeRound::dummy(),
                triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            blake_compress_interaction_claim.claimed_sum,
        );

        assert_component(&blake_compress_components, &trace)
    }


    // #[test]
    // fn test_blake_compress_opcode_trace_gen_by_cpu_and_verify_by_cuda() {

    //     let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
    //     let input_state = input.state_transitions;
    //     // println{}!("instruction_by_pc:{:?}", instruction_by_pc);

    //     let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
    //     let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);

    //     // Yield public memory.
    //     for addr in input
    //         .public_memory_addresses
    //         .iter()
    //         .copied()
    //         .map(M31::from_u32_unchecked)
    //     {
    //         let id = memory_address_to_id_trace_generator.get_id(addr);
    //         memory_address_to_id_trace_generator.add_input(&addr);
    //         memory_id_to_big_trace_generator.add_input(&id);
    //     }

    //     let blake_compress = blake_compress_opcode::ClaimGenerator::new(input_state.casm_states_by_opcode.blake_compress_opcode);

    //     let mut blake_round_trace_generator = blake_round::ClaimGenerator::new(input.memory);
    //     let mut triple_xor_32_trace_generator = triple_xor_32::ClaimGenerator::new();
    //     let verify_instruction_trace_generator = verify_instruction::ClaimGenerator::new(input.inst_cache);
    //     let range_check_7_2_5_trace_generator = range_check_7_2_5::ClaimGenerator::new();
    //     let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();


    //     let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
    //     let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
    //     let verify_instruction_relation = relations::VerifyInstruction::dummy();
    //     let blake_round_relation = relations::BlakeRound::dummy();
    //     let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();
    //     let triple_xor_32_relation = relations::TripleXor32::dummy();
    //     let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
    //     let opcodes_relation = relations::Opcodes::dummy();

    //     let mut mock_commitment_scheme = MockCommitmentScheme::default();

    //     // Preprocessed.
    //     let preprocessed_trace = testing_preprocessed_tree(4);
    //     let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
    //     mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
    //     mock_tree_builder.finalize_interaction();

    //     // Base trace.
    //     let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
    //     let (blake_compress_claim, blake_compress_interaction_gen) = blake_compress.write_trace(
    //                 &mut mock_tree_builder,
    //                 &mut blake_round_trace_generator,
    //                 &memory_address_to_id_trace_generator,
    //                 &memory_id_to_big_trace_generator,
    //                 &range_check_7_2_5_trace_generator,
    //                 &mut triple_xor_32_trace_generator,
    //                 &verify_bitwise_xor_8_trace_generator,
    //                 &verify_instruction_trace_generator,
    //             );

    //     mock_tree_builder.finalize_interaction();


    //     // Interaction trace.
    //     let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
    //     let blake_compress_interaction_claim = blake_compress_interaction_gen.write_interaction_trace(
    //                 &mut mock_tree_builder,
    //                 &verify_instruction_relation,
    //                 &memory_address_to_id_relation,
    //                 &memory_id_to_big_relation,
    //                 &range_check_7_2_5_relation,
    //                 &verify_bitwise_xor_8_relation,
    //                 &blake_round_relation,
    //                 &triple_xor_32_relation,
    //                 &opcodes_relation,
    //             );
    //     mock_tree_builder.finalize_interaction();
    //     let trace = mock_commitment_scheme.trace_domain_evaluations();

    //     // for i in 0..trace[0].len() {
    //     // }

    //     // for i in 0..trace[1].len() {
    //     // }

    //     // for i in 0..trace[2].len() {
    //     // }


    //     let trace0_vec: Vec<_> = trace[0].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
    //     let trace1_vec: Vec<_> = trace[1].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
    //     let trace2_vec: Vec<_> = trace[2].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();

    //     let trace0_evaluations_vec = trace0_vec
    //         .iter()
    //         .map(|column_evaluations| column_evaluations.device_ptr)
    //         .collect_vec();
    //     let trace1_evaluations_vec = trace1_vec
    //         .iter()
    //         .map(|column_evaluations| column_evaluations.device_ptr)
    //         .collect_vec();

    //     let mut trace2_evaluations_vec = vec![];
    //     if trace.len() != 2 {
    //         trace2_evaluations_vec = trace2_vec
    //             .iter()
    //             .map(|column_evaluations| column_evaluations.device_ptr)
    //             .collect_vec();
    //     }

    //     let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
    //     let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

    //     let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
    //     let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
    //     let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
    //     let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

    //     let tree_span_provider = &mut TraceLocationAllocator::default();
    //     let blake_compress_components = Component::new(
    //         tree_span_provider,
    //         Eval {
    //             eval_id: fnv1a_eval_id_gen("blake_compress_opcode"),
    //             claim: blake_compress_claim.clone(),
    //             verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
    //             memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
    //             memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
    //             range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
    //             verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
    //             blake_round_lookup_elements: relations::BlakeRound::dummy(),
    //             triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
    //             opcodes_lookup_elements: relations::Opcodes::dummy(),
    //         },
    //         blake_compress_interaction_claim.claimed_sum,
    //     );


    //     let eval_ptr = &blake_compress_components.eval as *const _ as *mut std::os::raw::c_void;
    //     unsafe {
    //         stwo::stwo_cuda::bindings::evaluate_constraint_quotients_on_domain(
    //             mock_accum_col_columns_0.device_ptr,
    //             mock_accum_col_columns_1.device_ptr,
    //             mock_accum_col_columns_2.device_ptr,
    //             mock_accum_col_columns_3.device_ptr,
    //             trace0_evaluations_vec.as_ptr(),
    //             trace0_evaluations_vec.len() as u32,
    //             trace1_evaluations_vec.as_ptr(),
    //             trace1_evaluations_vec.len() as u32,
    //             trace2_evaluations_vec.as_ptr(),
    //             trace2_evaluations_vec.len() as u32,
    //             mock_random_coeff_powers.device_ptr,
    //             mock_gpu_denom_inv.device_ptr,
    //             blake_compress_claim.log_size as u32,
    //             blake_compress_claim.log_size as u32,
    //             blake_compress_components.info.n_constraints as u32,
    //             blake_compress_components.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
    //             eval_ptr,
    //             CudaSecureField::from(
    //                 blake_compress_interaction_claim.claimed_sum
    //                     / BaseField::from_u32_unchecked(1 << blake_compress_claim.log_size)
    //             ),
    //             false,
    //             true,
    //         );
    //     }
    // }

    #[test]
    fn test_blake_compress_opcode_trace_gen_by_cuda_and_verify_by_cpu() {
        let input = prover_input_from_compiled_cairo_program("test_prove_verify_all_opcode_components");
        let input_state = input.state_transitions;
        assert!(!input_state.casm_states_by_opcode.blake_compress_opcode.is_empty());

        let mut blake_round_trace_generator = blake_round_cuda::CudaClaimGenerator::new(input.memory.clone());
        let mut triple_xor_32_trace_generator = triple_xor_32_cuda::CudaClaimGenerator::new();
        let mut memory_address_to_id_trace_generator =
            memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_trace_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);
        // Pad address_to_raw_id and multiplicities to power of 2 to avoid CUDA out of bounds access
        let padded_len = memory_address_to_id_trace_generator
            .address_to_raw_id
            .size
            .next_power_of_two();
        if padded_len > memory_address_to_id_trace_generator.address_to_raw_id.size {
            let mut addr = memory_address_to_id_trace_generator.address_to_raw_id.to_vec();
            addr.resize(padded_len, 0);
            memory_address_to_id_trace_generator.address_to_raw_id = Uint32Vec::from_vec(addr);

            let mut mults = memory_address_to_id_trace_generator.multiplicities.to_vec();
            mults.resize(padded_len, 0);
            memory_address_to_id_trace_generator.multiplicities = Uint32Vec::from_vec(mults);
        }
        let range_check_7_2_5_trace_generator = range_check_7_2_5_cuda::CudaClaimGenerator::new_rc_7_2_5();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
        let verify_instruction_trace_generator =
            verify_instruction_cuda::CudaClaimGenerator::new(input.inst_cache.clone());

        // Public memory setup
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_trace_generator.get_id(addr);
            memory_address_to_id_trace_generator.add_cuda_input(&addr);
            memory_id_to_big_trace_generator.add_cuda_input(&id);
        }

        let blake_compress =
            CudaClaimGenerator::new(input_state.casm_states_by_opcode.blake_compress_opcode);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_generator = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_generator = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_instruction_simd_generator = verify_instruction::ClaimGenerator::new(input.inst_cache.clone());
        let mut blake_round_simd_generator = blake_round::ClaimGenerator::new(input.memory.clone());
        let mut triple_xor_32_simd_generator = triple_xor_32::ClaimGenerator::new();
        let verify_bitwise_xor_8_simd_generator = verify_bitwise_xor_8::ClaimGenerator::new();
        let range_check_7_2_5_simd_generator = range_check_7_2_5::ClaimGenerator::new();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let verify_instruction_relation = relations::VerifyInstruction::dummy();
        let blake_round_relation = relations::BlakeRound::dummy();
        let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();
        let triple_xor_32_relation = relations::TripleXor32::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let opcodes_relation = relations::Opcodes::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace generated by CUDA
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_compress_claim, blake_compress_interaction_gen) = blake_compress.write_trace(
            &mut mock_tree_builder,
            &mut blake_round_trace_generator,
            &mut memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_7_2_5_trace_generator,
            &mut triple_xor_32_trace_generator,
            &verify_bitwise_xor_8_trace_generator,
            &verify_instruction_trace_generator,
            &memory_address_to_id_simd_generator,
            &memory_id_to_big_simd_generator,
            &verify_instruction_simd_generator,
            &mut blake_round_simd_generator,
            &mut triple_xor_32_simd_generator,
            &verify_bitwise_xor_8_simd_generator,
            &range_check_7_2_5_simd_generator,
        );
        mock_tree_builder.finalize_interaction();

        // Interaction trace generated by CUDA, verified by CPU
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_compress_interaction_claim = blake_compress_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_instruction_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_7_2_5_relation,
            &verify_bitwise_xor_8_relation,
            &blake_round_relation,
            &triple_xor_32_relation,
            &opcodes_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let blake_compress_components = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_compress_opcode"),
                claim: blake_compress_claim.clone(),
                verify_instruction_lookup_elements: relations::VerifyInstruction::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                blake_round_lookup_elements: relations::BlakeRound::dummy(),
                triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
                opcodes_lookup_elements: relations::Opcodes::dummy(),
            },
            blake_compress_interaction_claim.claimed_sum,
        );

        assert_component(&blake_compress_components, &trace)
    }
}
