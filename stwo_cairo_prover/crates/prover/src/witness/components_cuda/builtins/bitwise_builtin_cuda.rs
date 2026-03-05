// CUDA witness generation for bitwise_builtin component
// This handles AND/OR/XOR bitwise operations on 252-bit integers

#![allow(unused_parens)]
use cairo_air::components::bitwise_builtin::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda, vbx_8_cuda, vbx_9_cuda};
use crate::witness::prelude::*;
pub const N_TRACE_COLUMNS: usize = 89;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 19;

use itertools::Itertools;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::{Col, Column};
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

pub struct CudaClaimGenerator {
    pub log_size: u32,
    pub segment_start: u32,
}

impl CudaClaimGenerator {
    pub fn new(log_size: u32, segment_start: u32) -> Self {
        Self {
            log_size,
            segment_start,
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        verify_bitwise_xor_8_cuda_state: &vbx_8_cuda::CudaClaimGenerator,
        verify_bitwise_xor_9_cuda_state: &vbx_9_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        let n_rows = 1usize << log_size;

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            n_rows,
            log_size,
            self.segment_start,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
        );

        // Add to CUDA generators directly (no SIMD merge needed)
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add verify_bitwise_xor multiplicities directly to CUDA generators
        add_verify_bitwise_xor_multiplicities_cuda(
            &sub_component_inputs,
            verify_bitwise_xor_8_cuda_state,
            verify_bitwise_xor_9_cuda_state,
        );

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim {
                log_size,
                bitwise_builtin_segment_start: self.segment_start,
            },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

/// Add verify_bitwise_xor multiplicities directly to CUDA generators.
fn add_verify_bitwise_xor_multiplicities_cuda(
    sub_component_inputs: &CudaSubComponentInputs,
    verify_bitwise_xor_8_cuda_state: &vbx_8_cuda::CudaClaimGenerator,
    verify_bitwise_xor_9_cuda_state: &vbx_9_cuda::CudaClaimGenerator,
) {
    // Add verify_bitwise_xor_9 inputs (27 lookups)
    verify_bitwise_xor_9_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_9);

    // Add verify_bitwise_xor_8 inputs (1 lookup)
    verify_bitwise_xor_8_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
}

pub struct CudaSubComponentInputs {
    // 5 memory_address_to_id lookups
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 5],
    // 5 memory_id_to_big lookups
    pub memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 5],
    // 27 verify_bitwise_xor_9 lookups (3 elements each)
    pub verify_bitwise_xor_9: [[BaseFieldVec; 3]; 27],
    // 1 verify_bitwise_xor_8 lookup (3 elements each)
    pub verify_bitwise_xor_8: [[BaseFieldVec; 3]; 1],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    n_rows: usize,
    log_size: u32,
    segment_start: u32,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                // 5 memory_address_to_id lookups (2 elements each)
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                memory_address_to_id_3: init_lookup_array!(log_size),
                memory_address_to_id_4: init_lookup_array!(log_size),
                // 5 memory_id_to_big lookups (29 elements each)
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                memory_id_to_big_3: init_lookup_array!(log_size),
                memory_id_to_big_4: init_lookup_array!(log_size),
                // 27 verify_bitwise_xor_9 lookups (3 elements each)
                verify_bitwise_xor_9_0: init_lookup_array!(log_size),
                verify_bitwise_xor_9_1: init_lookup_array!(log_size),
                verify_bitwise_xor_9_2: init_lookup_array!(log_size),
                verify_bitwise_xor_9_3: init_lookup_array!(log_size),
                verify_bitwise_xor_9_4: init_lookup_array!(log_size),
                verify_bitwise_xor_9_5: init_lookup_array!(log_size),
                verify_bitwise_xor_9_6: init_lookup_array!(log_size),
                verify_bitwise_xor_9_7: init_lookup_array!(log_size),
                verify_bitwise_xor_9_8: init_lookup_array!(log_size),
                verify_bitwise_xor_9_9: init_lookup_array!(log_size),
                verify_bitwise_xor_9_10: init_lookup_array!(log_size),
                verify_bitwise_xor_9_11: init_lookup_array!(log_size),
                verify_bitwise_xor_9_12: init_lookup_array!(log_size),
                verify_bitwise_xor_9_13: init_lookup_array!(log_size),
                verify_bitwise_xor_9_14: init_lookup_array!(log_size),
                verify_bitwise_xor_9_15: init_lookup_array!(log_size),
                verify_bitwise_xor_9_16: init_lookup_array!(log_size),
                verify_bitwise_xor_9_17: init_lookup_array!(log_size),
                verify_bitwise_xor_9_18: init_lookup_array!(log_size),
                verify_bitwise_xor_9_19: init_lookup_array!(log_size),
                verify_bitwise_xor_9_20: init_lookup_array!(log_size),
                verify_bitwise_xor_9_21: init_lookup_array!(log_size),
                verify_bitwise_xor_9_22: init_lookup_array!(log_size),
                verify_bitwise_xor_9_23: init_lookup_array!(log_size),
                verify_bitwise_xor_9_24: init_lookup_array!(log_size),
                verify_bitwise_xor_9_25: init_lookup_array!(log_size),
                verify_bitwise_xor_9_26: init_lookup_array!(log_size),
                // 1 verify_bitwise_xor_8 lookup (3 elements each)
                verify_bitwise_xor_8_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                verify_bitwise_xor_9: std::array::from_fn(|_| {
                    std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << log_size))
                }),
                verify_bitwise_xor_8: std::array::from_fn(|_| {
                    std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << log_size))
                }),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers - 5 memory_address_to_id
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_address_to_id_3 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_3);
    let lookup_memory_address_to_id_4 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_4);

    // Collect lookup data pointers - 5 memory_id_to_big
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_memory_id_to_big_3 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_3);
    let lookup_memory_id_to_big_4 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_4);

    // Collect lookup data pointers - 27 verify_bitwise_xor_9
    let lookup_verify_bitwise_xor_9_0 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_0);
    let lookup_verify_bitwise_xor_9_1 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_1);
    let lookup_verify_bitwise_xor_9_2 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_2);
    let lookup_verify_bitwise_xor_9_3 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_3);
    let lookup_verify_bitwise_xor_9_4 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_4);
    let lookup_verify_bitwise_xor_9_5 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_5);
    let lookup_verify_bitwise_xor_9_6 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_6);
    let lookup_verify_bitwise_xor_9_7 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_7);
    let lookup_verify_bitwise_xor_9_8 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_8);
    let lookup_verify_bitwise_xor_9_9 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_9);
    let lookup_verify_bitwise_xor_9_10 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_10);
    let lookup_verify_bitwise_xor_9_11 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_11);
    let lookup_verify_bitwise_xor_9_12 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_12);
    let lookup_verify_bitwise_xor_9_13 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_13);
    let lookup_verify_bitwise_xor_9_14 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_14);
    let lookup_verify_bitwise_xor_9_15 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_15);
    let lookup_verify_bitwise_xor_9_16 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_16);
    let lookup_verify_bitwise_xor_9_17 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_17);
    let lookup_verify_bitwise_xor_9_18 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_18);
    let lookup_verify_bitwise_xor_9_19 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_19);
    let lookup_verify_bitwise_xor_9_20 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_20);
    let lookup_verify_bitwise_xor_9_21 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_21);
    let lookup_verify_bitwise_xor_9_22 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_22);
    let lookup_verify_bitwise_xor_9_23 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_23);
    let lookup_verify_bitwise_xor_9_24 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_24);
    let lookup_verify_bitwise_xor_9_25 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_25);
    let lookup_verify_bitwise_xor_9_26 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_26);

    // Collect lookup data pointers - 1 verify_bitwise_xor_8
    let lookup_verify_bitwise_xor_8_0 = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_0);

    let sub_component_inputs_memory_address_to_id_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_verify_bitwise_xor_9_vec: Vec<_> = sub_component_inputs
        .verify_bitwise_xor_9
        .iter()
        .flat_map(|row| row.iter().map(|x| x.device_ptr))
        .collect();
    let sub_component_inputs_verify_bitwise_xor_8_vec: Vec<_> = sub_component_inputs
        .verify_bitwise_xor_8
        .iter()
        .flat_map(|row| row.iter().map(|x| x.device_ptr))
        .collect();

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_bitwise_builtin_traces(
            traces_vec.as_ptr(),
            // 5 memory_address_to_id lookups
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            lookup_memory_address_to_id_3.as_ptr(),
            lookup_memory_address_to_id_4.as_ptr(),
            // 5 memory_id_to_big lookups
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            lookup_memory_id_to_big_3.as_ptr(),
            lookup_memory_id_to_big_4.as_ptr(),
            // 27 verify_bitwise_xor_9 lookups
            lookup_verify_bitwise_xor_9_0.as_ptr(),
            lookup_verify_bitwise_xor_9_1.as_ptr(),
            lookup_verify_bitwise_xor_9_2.as_ptr(),
            lookup_verify_bitwise_xor_9_3.as_ptr(),
            lookup_verify_bitwise_xor_9_4.as_ptr(),
            lookup_verify_bitwise_xor_9_5.as_ptr(),
            lookup_verify_bitwise_xor_9_6.as_ptr(),
            lookup_verify_bitwise_xor_9_7.as_ptr(),
            lookup_verify_bitwise_xor_9_8.as_ptr(),
            lookup_verify_bitwise_xor_9_9.as_ptr(),
            lookup_verify_bitwise_xor_9_10.as_ptr(),
            lookup_verify_bitwise_xor_9_11.as_ptr(),
            lookup_verify_bitwise_xor_9_12.as_ptr(),
            lookup_verify_bitwise_xor_9_13.as_ptr(),
            lookup_verify_bitwise_xor_9_14.as_ptr(),
            lookup_verify_bitwise_xor_9_15.as_ptr(),
            lookup_verify_bitwise_xor_9_16.as_ptr(),
            lookup_verify_bitwise_xor_9_17.as_ptr(),
            lookup_verify_bitwise_xor_9_18.as_ptr(),
            lookup_verify_bitwise_xor_9_19.as_ptr(),
            lookup_verify_bitwise_xor_9_20.as_ptr(),
            lookup_verify_bitwise_xor_9_21.as_ptr(),
            lookup_verify_bitwise_xor_9_22.as_ptr(),
            lookup_verify_bitwise_xor_9_23.as_ptr(),
            lookup_verify_bitwise_xor_9_24.as_ptr(),
            lookup_verify_bitwise_xor_9_25.as_ptr(),
            lookup_verify_bitwise_xor_9_26.as_ptr(),
            // 1 verify_bitwise_xor_8 lookup
            lookup_verify_bitwise_xor_8_0.as_ptr(),
            // Sub-component inputs
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_9_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_8_vec.as_ptr(),
            // Builtin segment info
            segment_start,
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
    // 5 memory_address_to_id lookups (2 elements each)
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    memory_address_to_id_3: [BaseFieldVec; 2],
    memory_address_to_id_4: [BaseFieldVec; 2],
    // 5 memory_id_to_big lookups (29 elements each)
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    memory_id_to_big_3: [BaseFieldVec; 29],
    memory_id_to_big_4: [BaseFieldVec; 29],
    // 27 verify_bitwise_xor_9 lookups (3 elements each)
    verify_bitwise_xor_9_0: [BaseFieldVec; 3],
    verify_bitwise_xor_9_1: [BaseFieldVec; 3],
    verify_bitwise_xor_9_2: [BaseFieldVec; 3],
    verify_bitwise_xor_9_3: [BaseFieldVec; 3],
    verify_bitwise_xor_9_4: [BaseFieldVec; 3],
    verify_bitwise_xor_9_5: [BaseFieldVec; 3],
    verify_bitwise_xor_9_6: [BaseFieldVec; 3],
    verify_bitwise_xor_9_7: [BaseFieldVec; 3],
    verify_bitwise_xor_9_8: [BaseFieldVec; 3],
    verify_bitwise_xor_9_9: [BaseFieldVec; 3],
    verify_bitwise_xor_9_10: [BaseFieldVec; 3],
    verify_bitwise_xor_9_11: [BaseFieldVec; 3],
    verify_bitwise_xor_9_12: [BaseFieldVec; 3],
    verify_bitwise_xor_9_13: [BaseFieldVec; 3],
    verify_bitwise_xor_9_14: [BaseFieldVec; 3],
    verify_bitwise_xor_9_15: [BaseFieldVec; 3],
    verify_bitwise_xor_9_16: [BaseFieldVec; 3],
    verify_bitwise_xor_9_17: [BaseFieldVec; 3],
    verify_bitwise_xor_9_18: [BaseFieldVec; 3],
    verify_bitwise_xor_9_19: [BaseFieldVec; 3],
    verify_bitwise_xor_9_20: [BaseFieldVec; 3],
    verify_bitwise_xor_9_21: [BaseFieldVec; 3],
    verify_bitwise_xor_9_22: [BaseFieldVec; 3],
    verify_bitwise_xor_9_23: [BaseFieldVec; 3],
    verify_bitwise_xor_9_24: [BaseFieldVec; 3],
    verify_bitwise_xor_9_25: [BaseFieldVec; 3],
    verify_bitwise_xor_9_26: [BaseFieldVec; 3],
    // 1 verify_bitwise_xor_8 lookup (3 elements each)
    verify_bitwise_xor_8_0: [BaseFieldVec; 3],
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

        // 19 pairs × 4 m31 columns = 76 columns
        // Each pair stores: 4 m31 for accumulated QM31 value
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect lookup data pointers - 5 memory_address_to_id
        let lookup_memory_address_to_id_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_address_to_id_3_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_3);
        let lookup_memory_address_to_id_4_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_4);

        // Collect lookup data pointers - 5 memory_id_to_big
        let lookup_memory_id_to_big_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_memory_id_to_big_3_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_3);
        let lookup_memory_id_to_big_4_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_4);

        // Collect lookup data pointers - 27 verify_bitwise_xor_9
        let lookup_verify_bitwise_xor_9_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_0);
        let lookup_verify_bitwise_xor_9_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_1);
        let lookup_verify_bitwise_xor_9_2_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_2);
        let lookup_verify_bitwise_xor_9_3_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_3);
        let lookup_verify_bitwise_xor_9_4_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_4);
        let lookup_verify_bitwise_xor_9_5_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_5);
        let lookup_verify_bitwise_xor_9_6_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_6);
        let lookup_verify_bitwise_xor_9_7_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_7);
        let lookup_verify_bitwise_xor_9_8_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_8);
        let lookup_verify_bitwise_xor_9_9_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_9);
        let lookup_verify_bitwise_xor_9_10_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_10);
        let lookup_verify_bitwise_xor_9_11_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_11);
        let lookup_verify_bitwise_xor_9_12_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_12);
        let lookup_verify_bitwise_xor_9_13_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_13);
        let lookup_verify_bitwise_xor_9_14_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_14);
        let lookup_verify_bitwise_xor_9_15_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_15);
        let lookup_verify_bitwise_xor_9_16_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_16);
        let lookup_verify_bitwise_xor_9_17_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_17);
        let lookup_verify_bitwise_xor_9_18_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_18);
        let lookup_verify_bitwise_xor_9_19_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_19);
        let lookup_verify_bitwise_xor_9_20_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_20);
        let lookup_verify_bitwise_xor_9_21_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_21);
        let lookup_verify_bitwise_xor_9_22_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_22);
        let lookup_verify_bitwise_xor_9_23_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_23);
        let lookup_verify_bitwise_xor_9_24_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_24);
        let lookup_verify_bitwise_xor_9_25_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_25);
        let lookup_verify_bitwise_xor_9_26_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_26);

        // Collect lookup data pointers - 1 verify_bitwise_xor_8
        let lookup_verify_bitwise_xor_8_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_0);

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
            let mod_xor9 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_9_RELATION_ID);
            let mod_xor8 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_8_RELATION_ID);

            bindings_airs::generate_bitwise_builtin_interaction_traces(
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_xor9 as *const _ as *mut std::os::raw::c_void,
                &mod_xor8 as *const _ as *mut std::os::raw::c_void,
                // 5 memory_address_to_id lookups
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_address_to_id_3_vec.as_ptr(),
                lookup_memory_address_to_id_4_vec.as_ptr(),
                // 5 memory_id_to_big lookups
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_memory_id_to_big_3_vec.as_ptr(),
                lookup_memory_id_to_big_4_vec.as_ptr(),
                // 27 verify_bitwise_xor_9 lookups
                lookup_verify_bitwise_xor_9_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_2_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_3_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_4_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_5_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_6_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_7_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_8_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_9_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_10_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_11_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_12_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_13_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_14_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_15_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_16_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_17_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_18_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_19_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_20_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_21_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_22_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_23_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_24_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_25_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_26_vec.as_ptr(),
                // 1 verify_bitwise_xor_8 lookup
                lookup_verify_bitwise_xor_8_0_vec.as_ptr(),
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
