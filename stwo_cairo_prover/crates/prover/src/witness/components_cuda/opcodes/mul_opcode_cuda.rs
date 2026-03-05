#![allow(unused_parens)]
use cairo_air::components::mul_opcode::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use super::super::{
    memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_20_cuda,
    verify_instruction_cuda,
};
use crate::witness::prelude::*;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo::core::fields::m31::BaseField;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
pub const N_TRACE_COLUMNS: usize = 130;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 19;

pub type CudaPackedInputs = [BaseFieldVec; 3];
use itertools::Itertools;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

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

    #[allow(clippy::too_many_arguments)]
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        verify_instruction_cuda_state: &verify_instruction_cuda::CudaClaimGenerator,
        // Range check generator (CUDA - use relation_index to distinguish instances)
        range_check_20_cuda_state: &range_check_20_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let size = self.inputs[0].size;
        let log_size = size.ilog2();
        let packed_inputs = self.inputs;

        let (trace, mut lookup_data, sub_component_inputs) = write_trace_cuda(
            self.n_rows,
            packed_inputs,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
            verify_instruction_cuda_state,
        );

        // Normalize range_check_19_h_0 offset: CUDA kernel stores k_col+262144,
        // subtract 131072 to make uniform base offset of 131072 (like all other carries).
        {
            let mut h0_vals: Vec<M31> = lookup_data.range_check_19_h_0[0].to_vec();
            for v in &mut h0_vals {
                *v = *v - M31(131072);
            }
            lookup_data.range_check_19_h_0[0] = BaseFieldVec::from_vec(h0_vals);
        }

        // Add to CUDA generators for multiplicity accumulation
        verify_instruction_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // NOTE: Do NOT add to memory SIMD generators here - that would cause double-counting
        // when merge_simd_multiplicities() is called later.

        // Add range check inputs via CUDA path with offset correction.
        // CUDA stores rc_19 values with base offset 131072; rc_20 expects 524288.
        // Correction: 524288 - 131072 = 393216.
        let rc20_offset = M31(393216);

        // CUDA _h → relation 0 (k_col, carry_7, carry_15, carry_23)
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_h_0.clone(), lookup_data.range_check_19_h_1.clone(),
              lookup_data.range_check_19_h_2.clone(), lookup_data.range_check_19_h_3.clone()],
            0, rc20_offset,
        );
        // CUDA _0 → relation 1 (carry_0, carry_8, carry_16, carry_24)
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_0.clone(), lookup_data.range_check_19_1.clone(),
              lookup_data.range_check_19_2.clone(), lookup_data.range_check_19_3.clone()],
            1, rc20_offset,
        );
        // CUDA _b → relation 2
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_b_0.clone(), lookup_data.range_check_19_b_1.clone(),
              lookup_data.range_check_19_b_2.clone(), lookup_data.range_check_19_b_3.clone()],
            2, rc20_offset,
        );
        // CUDA _c → relation 3
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_c_0.clone(), lookup_data.range_check_19_c_1.clone(),
              lookup_data.range_check_19_c_2.clone(), lookup_data.range_check_19_c_3.clone()],
            3, rc20_offset,
        );
        // CUDA _d → relation 4
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_d_0.clone(), lookup_data.range_check_19_d_1.clone(),
              lookup_data.range_check_19_d_2.clone()],
            4, rc20_offset,
        );
        // CUDA _e → relation 5
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_e_0.clone(), lookup_data.range_check_19_e_1.clone(),
              lookup_data.range_check_19_e_2.clone()],
            5, rc20_offset,
        );
        // CUDA _f → relation 6
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_f_0.clone(), lookup_data.range_check_19_f_1.clone(),
              lookup_data.range_check_19_f_2.clone()],
            6, rc20_offset,
        );
        // CUDA _g → relation 7
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[lookup_data.range_check_19_g_0.clone(), lookup_data.range_check_19_g_1.clone(),
              lookup_data.range_check_19_g_2.clone()],
            7, rc20_offset,
        );

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

    let sub_component_inputs_verify_instruction_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_instruction);
    let sub_component_inputs_memory_address_to_id_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_range_check_19_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19);
    let sub_component_inputs_range_check_19_b_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_b);
    let sub_component_inputs_range_check_19_c_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_c);
    let sub_component_inputs_range_check_19_d_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_d);
    let sub_component_inputs_range_check_19_e_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_e);
    let sub_component_inputs_range_check_19_f_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_f);
    let sub_component_inputs_range_check_19_g_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_g);
    let sub_component_inputs_range_check_19_h_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_19_h);

    let mul_opcode_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
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
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect all lookup pointers for interaction trace generation
        let lookup_memory_address_to_id_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_id_to_big_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_opcodes_0_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_0);
        let lookup_opcodes_1_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_1);

        let lookup_range_check_19_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_0);
        let lookup_range_check_19_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_1);
        let lookup_range_check_19_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_2);
        let lookup_range_check_19_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_3);

        let lookup_range_check_19_b_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_0);
        let lookup_range_check_19_b_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_1);
        let lookup_range_check_19_b_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_2);
        let lookup_range_check_19_b_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_3);

        let lookup_range_check_19_c_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_0);
        let lookup_range_check_19_c_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_1);
        let lookup_range_check_19_c_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_2);
        let lookup_range_check_19_c_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_3);

        let lookup_range_check_19_d_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_d_0);
        let lookup_range_check_19_d_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_d_1);
        let lookup_range_check_19_d_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_d_2);

        let lookup_range_check_19_e_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_e_0);
        let lookup_range_check_19_e_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_e_1);
        let lookup_range_check_19_e_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_e_2);

        let lookup_range_check_19_f_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_f_0);
        let lookup_range_check_19_f_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_f_1);
        let lookup_range_check_19_f_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_f_2);

        let lookup_range_check_19_g_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_g_0);
        let lookup_range_check_19_g_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_g_1);
        let lookup_range_check_19_g_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_g_2);

        let lookup_range_check_19_h_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_0);
        let lookup_range_check_19_h_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_1);
        let lookup_range_check_19_h_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_2);
        let lookup_range_check_19_h_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_3);

        let lookup_verify_instruction_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_instruction_0);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
            let mod_ops = create_modified_lookup_for_cuda(lookup_elements, OPCODES_RELATION_ID);
            let mod_vi =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_INSTRUCTION_RELATION_ID);
            // CUDA rc_19 variant → correct "now" relation constant (shifted mapping).
            // CUDA _h → relation 0, _0 → relation 1, _b → relation 2, ..., _g → relation 7.
            // Also compensate for offset mismatch: CUDA uses 131072, AIR expects 524288.
            let rc20_offset_corr = M31(393216); // 524288 - 131072
            let mod_rc19 = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_B_RELATION_ID,
                rc20_offset_corr,
            ); // _0 → rel 1
            let mod_rc19b = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_C_RELATION_ID,
                rc20_offset_corr,
            ); // _b → rel 2
            let mod_rc19c = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_D_RELATION_ID,
                rc20_offset_corr,
            ); // _c → rel 3
            let mod_rc19d = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_E_RELATION_ID,
                rc20_offset_corr,
            ); // _d → rel 4
            let mod_rc19e = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_F_RELATION_ID,
                rc20_offset_corr,
            ); // _e → rel 5
            let mod_rc19f = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_G_RELATION_ID,
                rc20_offset_corr,
            ); // _f → rel 6
            let mod_rc19g = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_H_RELATION_ID,
                rc20_offset_corr,
            ); // _g → rel 7
            let mod_rc19h = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_RELATION_ID,
                rc20_offset_corr,
            ); // _h → rel 0

            bindings_airs::generate_mul_opcode_interaction_traces(
                // Relation pointers (12 relations)
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_ops as *const _ as *mut std::os::raw::c_void,
                &mod_vi as *const _ as *mut std::os::raw::c_void,
                &mod_rc19 as *const _ as *mut std::os::raw::c_void,
                &mod_rc19b as *const _ as *mut std::os::raw::c_void,
                &mod_rc19c as *const _ as *mut std::os::raw::c_void,
                &mod_rc19d as *const _ as *mut std::os::raw::c_void,
                &mod_rc19e as *const _ as *mut std::os::raw::c_void,
                &mod_rc19f as *const _ as *mut std::os::raw::c_void,
                &mod_rc19g as *const _ as *mut std::os::raw::c_void,
                &mod_rc19h as *const _ as *mut std::os::raw::c_void,
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
