#![allow(unused_parens)]
use cairo_air::components::blake_compress_opcode::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::prover::backend::cuda::CudaBackend;

use crate::witness::components_cuda::{
    blake_round_cuda, memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_7_2_5_cuda,
    triple_xor_32_cuda, vbx_8_cuda as verify_bitwise_xor_8_cuda, verify_instruction_cuda,
};
use crate::witness::prelude::*;
pub const N_TRACE_COLUMNS: usize = 174;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 37;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
pub type CudaPackedInputs = [BaseFieldVec; 3];
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}
macro_rules! init_subcomponent_basefield_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
        })
    };
}
macro_rules! init_subcomponent_uint32_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| Uint32Vec::new_uninitialized(1 << $log_size))
        })
    };
}

/// CUDA format for blake_round inputs (used locally, not imported from blake_round_cuda)
pub type BlakeRoundCudaInputType = (BaseFieldVec, BaseFieldVec, ([Uint32Vec; 16], BaseFieldVec));

fn init_blake_subcomponent_array<const N: usize>(log_size: u32) -> [BlakeRoundCudaInputType; N] {
    unsafe {
        std::array::from_fn(|_| {
            (
                BaseFieldVec::uninitialized(1 << log_size),
                BaseFieldVec::uninitialized(1 << log_size),
                (
                    std::array::from_fn(|_| Uint32Vec::new_uninitialized(1 << log_size)),
                    BaseFieldVec::uninitialized(1 << log_size),
                ),
            )
        })
    }
}

macro_rules! collect_lookup_ptrs {
    ($lookup_data:expr, $field:ident) => {
        $lookup_data
            .$field
            .iter()
            .map(|x| x.device_ptr)
            .collect_vec()
    };
}

macro_rules! collect_sub_input_ptrs {
    ($sub_inputs:expr, $field:ident) => {
        $sub_inputs
            .$field
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec()
    };
}

fn collect_blake_sub_input_ptrs(arr: &[BlakeRoundCudaInputType]) -> Vec<*const u32> {
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
            inputs: [pc_vec, ap_vec, fp_vec],
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
        triple_xor_32_state: &mut triple_xor_32_cuda::CudaClaimGenerator,
        verify_bitwise_xor_8_state: &verify_bitwise_xor_8_cuda::CudaClaimGenerator,
        verify_instruction_state: &verify_instruction_cuda::CudaClaimGenerator,
        // CUDA blake_round generator (pure CUDA approach)
        blake_round_cuda_state: &mut blake_round_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let size = self.inputs[0].size;
        let log_size = size.ilog2();
        let packed_inputs = self.inputs;

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            self.n_rows,
            packed_inputs,
            memory_address_to_id_state,
            memory_id_to_big_state,
            range_check_7_2_5_state,
            triple_xor_32_state,
            verify_bitwise_xor_8_state,
            verify_instruction_state,
        );

        // Add to CUDA generators
        verify_instruction_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);
        range_check_7_2_5_state.add_cuda_inputs(&sub_component_inputs.range_check_7_2_5);
        verify_bitwise_xor_8_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
        triple_xor_32_state.add_cuda_inputs(&sub_component_inputs.triple_xor_32);

        // Add blake_round inputs to CUDA generator (pure CUDA approach)
        // Pass all rows including padding (like SIMD does) - blake_round's enabler handles padding
        // Add CUDA inputs directly to CUDA blake_round generator (all rows including padding)
        blake_round_cuda_state.add_cuda_inputs(&sub_component_inputs.blake_round);

        // Add to verify_instruction SIMD generator (still uses SIMD trace generation)
        let padded_size = 1usize << log_size;

        // Note: range_check_7_2_5 multiplicities are already added to CUDA generator (line 164).
        // No need to also add to SIMD generator - that would cause double-counting when merged.

        tree_builder.extend_evals(trace.to_evals().to_vec());

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
    memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 20],
    memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 20],
    range_check_7_2_5: [range_check_7_2_5_cuda::CudaPackedInputType; 17],
    verify_bitwise_xor_8: [verify_bitwise_xor_8_cuda::CudaPackedInputType; 4],
    blake_round: [BlakeRoundCudaInputType; 10],
    triple_xor_32: [triple_xor_32_cuda::CudaPackedInputType; 8],
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
    // 初始化所有trace、lookup、subcomponent数组
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                blake_round_0: init_lookup_array!(log_size),
                blake_round_1: init_lookup_array!(log_size),
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                memory_address_to_id_3: init_lookup_array!(log_size),
                memory_address_to_id_4: init_lookup_array!(log_size),
                memory_address_to_id_5: init_lookup_array!(log_size),
                memory_address_to_id_6: init_lookup_array!(log_size),
                memory_address_to_id_7: init_lookup_array!(log_size),
                memory_address_to_id_8: init_lookup_array!(log_size),
                memory_address_to_id_9: init_lookup_array!(log_size),
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
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                memory_id_to_big_3: init_lookup_array!(log_size),
                memory_id_to_big_4: init_lookup_array!(log_size),
                memory_id_to_big_5: init_lookup_array!(log_size),
                memory_id_to_big_6: init_lookup_array!(log_size),
                memory_id_to_big_7: init_lookup_array!(log_size),
                memory_id_to_big_8: init_lookup_array!(log_size),
                memory_id_to_big_9: init_lookup_array!(log_size),
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
                opcodes_0: init_lookup_array!(log_size),
                opcodes_1: init_lookup_array!(log_size),
                range_check_7_2_5_0: init_lookup_array!(log_size),
                range_check_7_2_5_1: init_lookup_array!(log_size),
                range_check_7_2_5_2: init_lookup_array!(log_size),
                range_check_7_2_5_3: init_lookup_array!(log_size),
                range_check_7_2_5_4: init_lookup_array!(log_size),
                range_check_7_2_5_5: init_lookup_array!(log_size),
                range_check_7_2_5_6: init_lookup_array!(log_size),
                range_check_7_2_5_7: init_lookup_array!(log_size),
                range_check_7_2_5_8: init_lookup_array!(log_size),
                range_check_7_2_5_9: init_lookup_array!(log_size),
                range_check_7_2_5_10: init_lookup_array!(log_size),
                range_check_7_2_5_11: init_lookup_array!(log_size),
                range_check_7_2_5_12: init_lookup_array!(log_size),
                range_check_7_2_5_13: init_lookup_array!(log_size),
                range_check_7_2_5_14: init_lookup_array!(log_size),
                range_check_7_2_5_15: init_lookup_array!(log_size),
                range_check_7_2_5_16: init_lookup_array!(log_size),
                triple_xor_32_0: init_lookup_array!(log_size),
                triple_xor_32_1: init_lookup_array!(log_size),
                triple_xor_32_2: init_lookup_array!(log_size),
                triple_xor_32_3: init_lookup_array!(log_size),
                triple_xor_32_4: init_lookup_array!(log_size),
                triple_xor_32_5: init_lookup_array!(log_size),
                triple_xor_32_6: init_lookup_array!(log_size),
                triple_xor_32_7: init_lookup_array!(log_size),
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
        ($name:ident) => {
            collect_lookup_ptrs!(lookup_data, $name)
        };
    }
    macro_rules! sub_ptrs {
        ($name:ident) => {
            collect_sub_input_ptrs!(sub_component_inputs, $name)
        };
    }
    let lookup_blake_round_0 = ptrs!(blake_round_0);
    let lookup_blake_round_1 = ptrs!(blake_round_1);
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
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_blake_compress_opcode_traces(
            traces_vec.as_ptr(),
            // 所有 lookup ptrs
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
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // 分配 interaction trace，每列一个 Col
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // 收集所有 lookup ptrs
        macro_rules! ptrs {
            ($name:ident) => {
                collect_lookup_ptrs!(self.lookup_data, $name)
            };
        }
        let _lookup_blake_round_0 = ptrs!(blake_round_0);
        let _lookup_blake_round_1 = ptrs!(blake_round_1);
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
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_blake_round =
                create_modified_lookup_for_cuda(lookup_elements, BLAKE_ROUND_RELATION_ID);
            let mod_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
            let mod_ops = create_modified_lookup_for_cuda(lookup_elements, OPCODES_RELATION_ID);
            let mod_rc725 = create_modified_lookup_for_cuda(lookup_elements, RC_7_2_5_RELATION_ID);
            let mod_txor32 =
                create_modified_lookup_for_cuda(lookup_elements, TRIPLE_XOR_32_RELATION_ID);
            let mod_vbx8 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_8_RELATION_ID);
            let mod_vi =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_INSTRUCTION_RELATION_ID);

            bindings_airs::generate_blake_compress_opcode_interaction_traces(
                &mod_blake_round as *const _ as *mut std::os::raw::c_void,
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_ops as *const _ as *mut std::os::raw::c_void,
                &mod_rc725 as *const _ as *mut std::os::raw::c_void,
                &mod_txor32 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx8 as *const _ as *mut std::os::raw::c_void,
                &mod_vi as *const _ as *mut std::os::raw::c_void,
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
