#![allow(unused_parens)]
use cairo_air::components::add_mod_builtin::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::prelude::*;
pub const N_TRACE_COLUMNS: usize = 267;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 27;

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

        // Add to CUDA generators - SIMD multiplicities are handled by the merge in cairo_cuda.rs
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim {
                log_size,
                add_mod_builtin_segment_start: self.segment_start,
            },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

pub struct CudaSubComponentInputs {
    // 29 memory_address_to_id lookups
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 29],
    // 24 memory_id_to_big lookups
    pub memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 24],
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
                // 29 memory_address_to_id lookups (2 elements each)
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
                memory_address_to_id_20: init_lookup_array!(log_size),
                memory_address_to_id_21: init_lookup_array!(log_size),
                memory_address_to_id_22: init_lookup_array!(log_size),
                memory_address_to_id_23: init_lookup_array!(log_size),
                memory_address_to_id_24: init_lookup_array!(log_size),
                memory_address_to_id_25: init_lookup_array!(log_size),
                memory_address_to_id_26: init_lookup_array!(log_size),
                memory_address_to_id_27: init_lookup_array!(log_size),
                memory_address_to_id_28: init_lookup_array!(log_size),
                // 24 memory_id_to_big lookups (29 elements each)
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
                memory_id_to_big_20: init_lookup_array!(log_size),
                memory_id_to_big_21: init_lookup_array!(log_size),
                memory_id_to_big_22: init_lookup_array!(log_size),
                memory_id_to_big_23: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers - 29 memory_address_to_id
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_address_to_id_3 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_3);
    let lookup_memory_address_to_id_4 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_4);
    let lookup_memory_address_to_id_5 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_5);
    let lookup_memory_address_to_id_6 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_6);
    let lookup_memory_address_to_id_7 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_7);
    let lookup_memory_address_to_id_8 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_8);
    let lookup_memory_address_to_id_9 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_9);
    let lookup_memory_address_to_id_10 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_10);
    let lookup_memory_address_to_id_11 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_11);
    let lookup_memory_address_to_id_12 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_12);
    let lookup_memory_address_to_id_13 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_13);
    let lookup_memory_address_to_id_14 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_14);
    let lookup_memory_address_to_id_15 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_15);
    let lookup_memory_address_to_id_16 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_16);
    let lookup_memory_address_to_id_17 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_17);
    let lookup_memory_address_to_id_18 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_18);
    let lookup_memory_address_to_id_19 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_19);
    let lookup_memory_address_to_id_20 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_20);
    let lookup_memory_address_to_id_21 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_21);
    let lookup_memory_address_to_id_22 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_22);
    let lookup_memory_address_to_id_23 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_23);
    let lookup_memory_address_to_id_24 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_24);
    let lookup_memory_address_to_id_25 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_25);
    let lookup_memory_address_to_id_26 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_26);
    let lookup_memory_address_to_id_27 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_27);
    let lookup_memory_address_to_id_28 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_28);

    // Collect lookup data pointers - 24 memory_id_to_big
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_memory_id_to_big_3 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_3);
    let lookup_memory_id_to_big_4 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_4);
    let lookup_memory_id_to_big_5 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_5);
    let lookup_memory_id_to_big_6 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_6);
    let lookup_memory_id_to_big_7 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_7);
    let lookup_memory_id_to_big_8 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_8);
    let lookup_memory_id_to_big_9 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_9);
    let lookup_memory_id_to_big_10 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_10);
    let lookup_memory_id_to_big_11 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_11);
    let lookup_memory_id_to_big_12 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_12);
    let lookup_memory_id_to_big_13 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_13);
    let lookup_memory_id_to_big_14 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_14);
    let lookup_memory_id_to_big_15 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_15);
    let lookup_memory_id_to_big_16 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_16);
    let lookup_memory_id_to_big_17 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_17);
    let lookup_memory_id_to_big_18 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_18);
    let lookup_memory_id_to_big_19 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_19);
    let lookup_memory_id_to_big_20 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_20);
    let lookup_memory_id_to_big_21 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_21);
    let lookup_memory_id_to_big_22 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_22);
    let lookup_memory_id_to_big_23 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_23);

    let sub_component_inputs_memory_address_to_id_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_add_mod_builtin_traces(
            traces_vec.as_ptr(),
            // 29 memory_address_to_id lookups
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
            lookup_memory_address_to_id_20.as_ptr(),
            lookup_memory_address_to_id_21.as_ptr(),
            lookup_memory_address_to_id_22.as_ptr(),
            lookup_memory_address_to_id_23.as_ptr(),
            lookup_memory_address_to_id_24.as_ptr(),
            lookup_memory_address_to_id_25.as_ptr(),
            lookup_memory_address_to_id_26.as_ptr(),
            lookup_memory_address_to_id_27.as_ptr(),
            lookup_memory_address_to_id_28.as_ptr(),
            // 24 memory_id_to_big lookups
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
            lookup_memory_id_to_big_20.as_ptr(),
            lookup_memory_id_to_big_21.as_ptr(),
            lookup_memory_id_to_big_22.as_ptr(),
            lookup_memory_id_to_big_23.as_ptr(),
            // Sub-component inputs
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
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
    // 29 memory_address_to_id lookups (2 elements each)
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
    memory_address_to_id_20: [BaseFieldVec; 2],
    memory_address_to_id_21: [BaseFieldVec; 2],
    memory_address_to_id_22: [BaseFieldVec; 2],
    memory_address_to_id_23: [BaseFieldVec; 2],
    memory_address_to_id_24: [BaseFieldVec; 2],
    memory_address_to_id_25: [BaseFieldVec; 2],
    memory_address_to_id_26: [BaseFieldVec; 2],
    memory_address_to_id_27: [BaseFieldVec; 2],
    memory_address_to_id_28: [BaseFieldVec; 2],
    // 24 memory_id_to_big lookups (29 elements each)
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
    memory_id_to_big_20: [BaseFieldVec; 29],
    memory_id_to_big_21: [BaseFieldVec; 29],
    memory_id_to_big_22: [BaseFieldVec; 29],
    memory_id_to_big_23: [BaseFieldVec; 29],
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

        // Collect lookup data pointers - 29 memory_address_to_id
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
        let lookup_memory_address_to_id_5_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_5);
        let lookup_memory_address_to_id_6_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_6);
        let lookup_memory_address_to_id_7_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_7);
        let lookup_memory_address_to_id_8_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_8);
        let lookup_memory_address_to_id_9_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_9);
        let lookup_memory_address_to_id_10_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_10);
        let lookup_memory_address_to_id_11_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_11);
        let lookup_memory_address_to_id_12_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_12);
        let lookup_memory_address_to_id_13_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_13);
        let lookup_memory_address_to_id_14_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_14);
        let lookup_memory_address_to_id_15_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_15);
        let lookup_memory_address_to_id_16_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_16);
        let lookup_memory_address_to_id_17_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_17);
        let lookup_memory_address_to_id_18_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_18);
        let lookup_memory_address_to_id_19_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_19);
        let lookup_memory_address_to_id_20_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_20);
        let lookup_memory_address_to_id_21_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_21);
        let lookup_memory_address_to_id_22_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_22);
        let lookup_memory_address_to_id_23_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_23);
        let lookup_memory_address_to_id_24_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_24);
        let lookup_memory_address_to_id_25_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_25);
        let lookup_memory_address_to_id_26_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_26);
        let lookup_memory_address_to_id_27_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_27);
        let lookup_memory_address_to_id_28_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_28);

        // Collect lookup data pointers - 24 memory_id_to_big
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
        let lookup_memory_id_to_big_5_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_5);
        let lookup_memory_id_to_big_6_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_6);
        let lookup_memory_id_to_big_7_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_7);
        let lookup_memory_id_to_big_8_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_8);
        let lookup_memory_id_to_big_9_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_9);
        let lookup_memory_id_to_big_10_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_10);
        let lookup_memory_id_to_big_11_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_11);
        let lookup_memory_id_to_big_12_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_12);
        let lookup_memory_id_to_big_13_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_13);
        let lookup_memory_id_to_big_14_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_14);
        let lookup_memory_id_to_big_15_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_15);
        let lookup_memory_id_to_big_16_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_16);
        let lookup_memory_id_to_big_17_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_17);
        let lookup_memory_id_to_big_18_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_18);
        let lookup_memory_id_to_big_19_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_19);
        let lookup_memory_id_to_big_20_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_20);
        let lookup_memory_id_to_big_21_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_21);
        let lookup_memory_id_to_big_22_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_22);
        let lookup_memory_id_to_big_23_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_23);

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

            bindings_airs::generate_add_mod_builtin_interaction_traces(
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                // 29 memory_address_to_id lookups
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_address_to_id_3_vec.as_ptr(),
                lookup_memory_address_to_id_4_vec.as_ptr(),
                lookup_memory_address_to_id_5_vec.as_ptr(),
                lookup_memory_address_to_id_6_vec.as_ptr(),
                lookup_memory_address_to_id_7_vec.as_ptr(),
                lookup_memory_address_to_id_8_vec.as_ptr(),
                lookup_memory_address_to_id_9_vec.as_ptr(),
                lookup_memory_address_to_id_10_vec.as_ptr(),
                lookup_memory_address_to_id_11_vec.as_ptr(),
                lookup_memory_address_to_id_12_vec.as_ptr(),
                lookup_memory_address_to_id_13_vec.as_ptr(),
                lookup_memory_address_to_id_14_vec.as_ptr(),
                lookup_memory_address_to_id_15_vec.as_ptr(),
                lookup_memory_address_to_id_16_vec.as_ptr(),
                lookup_memory_address_to_id_17_vec.as_ptr(),
                lookup_memory_address_to_id_18_vec.as_ptr(),
                lookup_memory_address_to_id_19_vec.as_ptr(),
                lookup_memory_address_to_id_20_vec.as_ptr(),
                lookup_memory_address_to_id_21_vec.as_ptr(),
                lookup_memory_address_to_id_22_vec.as_ptr(),
                lookup_memory_address_to_id_23_vec.as_ptr(),
                lookup_memory_address_to_id_24_vec.as_ptr(),
                lookup_memory_address_to_id_25_vec.as_ptr(),
                lookup_memory_address_to_id_26_vec.as_ptr(),
                lookup_memory_address_to_id_27_vec.as_ptr(),
                lookup_memory_address_to_id_28_vec.as_ptr(),
                // 24 memory_id_to_big lookups
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_memory_id_to_big_3_vec.as_ptr(),
                lookup_memory_id_to_big_4_vec.as_ptr(),
                lookup_memory_id_to_big_5_vec.as_ptr(),
                lookup_memory_id_to_big_6_vec.as_ptr(),
                lookup_memory_id_to_big_7_vec.as_ptr(),
                lookup_memory_id_to_big_8_vec.as_ptr(),
                lookup_memory_id_to_big_9_vec.as_ptr(),
                lookup_memory_id_to_big_10_vec.as_ptr(),
                lookup_memory_id_to_big_11_vec.as_ptr(),
                lookup_memory_id_to_big_12_vec.as_ptr(),
                lookup_memory_id_to_big_13_vec.as_ptr(),
                lookup_memory_id_to_big_14_vec.as_ptr(),
                lookup_memory_id_to_big_15_vec.as_ptr(),
                lookup_memory_id_to_big_16_vec.as_ptr(),
                lookup_memory_id_to_big_17_vec.as_ptr(),
                lookup_memory_id_to_big_18_vec.as_ptr(),
                lookup_memory_id_to_big_19_vec.as_ptr(),
                lookup_memory_id_to_big_20_vec.as_ptr(),
                lookup_memory_id_to_big_21_vec.as_ptr(),
                lookup_memory_id_to_big_22_vec.as_ptr(),
                lookup_memory_id_to_big_23_vec.as_ptr(),
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
