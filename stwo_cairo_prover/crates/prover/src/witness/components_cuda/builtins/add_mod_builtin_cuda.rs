#![allow(unused_parens)]
use cairo_air::components::add_mod_builtin::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big};
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 267;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 27;

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
        // Also pass SIMD generators for multiplicity tracking (needed for final memory traces)
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
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

        // Add to CUDA generators
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        // Copy GPU data to CPU and add to SIMD generators
        // Skip zero values (padding sentinel) and out-of-bounds addresses
        let memory_size = memory_address_to_id_simd_state.memory_size();
        for input_arr in &sub_component_inputs.memory_address_to_id {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for addr in cpu_data.iter().take(n_rows) {
                // Skip zero (padding sentinel) and addresses exceeding memory bounds
                if addr.0 != 0 && addr.0 <= memory_size {
                    memory_address_to_id_simd_state.add_input(addr);
                }
            }
        }
        for input_arr in &sub_component_inputs.memory_id_to_big {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for id in cpu_data.iter().take(n_rows) {
                // Skip zero (padding sentinel) - memory_id_to_big handles bounds internally
                if id.0 != 0 {
                    memory_id_to_big_simd_state.add_input(id);
                }
            }
        }

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size, add_mod_builtin_segment_start: self.segment_start },
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
    pub memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 24],
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

    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
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
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect lookup data pointers - 29 memory_address_to_id
        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_address_to_id_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_3);
        let lookup_memory_address_to_id_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_4);
        let lookup_memory_address_to_id_5_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_5);
        let lookup_memory_address_to_id_6_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_6);
        let lookup_memory_address_to_id_7_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_7);
        let lookup_memory_address_to_id_8_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_8);
        let lookup_memory_address_to_id_9_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_9);
        let lookup_memory_address_to_id_10_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_10);
        let lookup_memory_address_to_id_11_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_11);
        let lookup_memory_address_to_id_12_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_12);
        let lookup_memory_address_to_id_13_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_13);
        let lookup_memory_address_to_id_14_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_14);
        let lookup_memory_address_to_id_15_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_15);
        let lookup_memory_address_to_id_16_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_16);
        let lookup_memory_address_to_id_17_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_17);
        let lookup_memory_address_to_id_18_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_18);
        let lookup_memory_address_to_id_19_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_19);
        let lookup_memory_address_to_id_20_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_20);
        let lookup_memory_address_to_id_21_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_21);
        let lookup_memory_address_to_id_22_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_22);
        let lookup_memory_address_to_id_23_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_23);
        let lookup_memory_address_to_id_24_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_24);
        let lookup_memory_address_to_id_25_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_25);
        let lookup_memory_address_to_id_26_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_26);
        let lookup_memory_address_to_id_27_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_27);
        let lookup_memory_address_to_id_28_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_28);

        // Collect lookup data pointers - 24 memory_id_to_big
        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_memory_id_to_big_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_3);
        let lookup_memory_id_to_big_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_4);
        let lookup_memory_id_to_big_5_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_5);
        let lookup_memory_id_to_big_6_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_6);
        let lookup_memory_id_to_big_7_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_7);
        let lookup_memory_id_to_big_8_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_8);
        let lookup_memory_id_to_big_9_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_9);
        let lookup_memory_id_to_big_10_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_10);
        let lookup_memory_id_to_big_11_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_11);
        let lookup_memory_id_to_big_12_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_12);
        let lookup_memory_id_to_big_13_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_13);
        let lookup_memory_id_to_big_14_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_14);
        let lookup_memory_id_to_big_15_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_15);
        let lookup_memory_id_to_big_16_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_16);
        let lookup_memory_id_to_big_17_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_17);
        let lookup_memory_id_to_big_18_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_18);
        let lookup_memory_id_to_big_19_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_19);
        let lookup_memory_id_to_big_20_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_20);
        let lookup_memory_id_to_big_21_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_21);
        let lookup_memory_id_to_big_22_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_22);
        let lookup_memory_id_to_big_23_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_23);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_add_mod_builtin_interaction_traces(
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,

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
    use dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_adapter::utils::{run_program_and_adapter, ProgramType};
    use stwo::core::fields::m31::M31;
    use crate::witness::components::{
        memory_address_to_id, memory_id_to_big,
        add_mod_builtin,
    };
    use crate::witness::components_cuda::{
        memory_address_to_id_cuda, memory_id_to_big_cuda,
        add_mod_builtin_cuda,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use crate::debug_tools::assert_constraints::assert_component;

    #[test]
    fn test_add_mod_builtin_cpu_ref() {
        use cairo_air::relations;
        use cairo_air::components::add_mod_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_add_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for add_mod builtin segment
        use stwo_cairo_adapter::builtins::ADD_MOD_MEMORY_CELLS;

        let add_mod_segment = input.builtin_segments.add_mod.as_ref()
            .expect("Expected add_mod builtin segment");

        let segment_length = add_mod_segment.stop_ptr - add_mod_segment.begin_addr;
        let n_instances = segment_length / ADD_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = add_mod_segment.begin_addr as u32;

        println!("add_mod_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_state.get_id(addr);
            memory_address_to_id_state.add_input(&addr);
            memory_id_to_big_state.add_input(&id);
        }

        // Create add_mod_builtin claim generator
        let add_mod_claim_gen = add_mod_builtin::ClaimGenerator::new(log_size, segment_start);

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (add_mod_claim, add_mod_interaction_gen) = add_mod_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
        );

        mock_tree_builder.finalize_interaction();

        println!("add_mod_builtin_claim log_size: {:?}", add_mod_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let add_mod_interaction_claim = add_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("add_mod_builtin_interaction_claim.claimed_sum: {:?}", add_mod_interaction_claim.claimed_sum);

        // Create component and verify with assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let add_mod_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("add_mod_builtin"),
                claim: add_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
            },
            add_mod_interaction_claim.claimed_sum,
        );

        assert_component(&add_mod_component, &trace)
    }

    #[test]
    fn test_add_mod_builtin_trace_gen_by_cpu_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::add_mod_builtin::{Component, Eval};
        use stwo::core::fields::m31::BaseField;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_add_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for add_mod builtin segment
        use stwo_cairo_adapter::builtins::ADD_MOD_MEMORY_CELLS;

        let add_mod_segment = input.builtin_segments.add_mod.as_ref()
            .expect("Expected add_mod builtin segment");

        let segment_length = add_mod_segment.stop_ptr - add_mod_segment.begin_addr;
        let n_instances = segment_length / ADD_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = add_mod_segment.begin_addr as u32;

        println!("add_mod_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_state.get_id(addr);
            memory_address_to_id_state.add_input(&addr);
            memory_id_to_big_state.add_input(&id);
        }

        // Create add_mod_builtin claim generator
        let add_mod_claim_gen = add_mod_builtin::ClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (add_mod_claim, add_mod_interaction_gen) = add_mod_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
        );

        mock_tree_builder.finalize_interaction();

        println!("add_mod_builtin_claim log_size: {:?}", add_mod_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let add_mod_interaction_claim = add_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("add_mod_builtin_interaction_claim.claimed_sum: {:?}", add_mod_interaction_claim.claimed_sum);

        // Convert trace to CUDA format
        let trace0_vec: Vec<_> = if !trace[0].is_empty() {
            trace[0].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace1_vec: Vec<_> = if trace.len() > 1 && !trace[1].is_empty() {
            trace[1].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace2_vec: Vec<_> = if trace.len() > 2 && !trace[2].is_empty() {
            trace[2].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace0_evaluations_vec: Vec<_> = trace0_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();
        let trace1_evaluations_vec: Vec<_> = trace1_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();
        let trace2_evaluations_vec: Vec<_> = trace2_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();

        // Create component for CUDA evaluator first to get actual constraint count
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let add_mod_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("add_mod_builtin"),
                claim: add_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
            },
            add_mod_interaction_claim.claimed_sum,
        );

        // Create mock CUDA buffers with correct size based on actual constraint count
        let domain_size = 1 << add_mod_claim.log_size;
        let n_constraints = add_mod_component.info.n_constraints.max(500);

        let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        // Call CUDA evaluator
        let eval_ptr = &add_mod_component.eval as *const _ as *mut std::os::raw::c_void;
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
                add_mod_claim.log_size as u32,
                add_mod_claim.log_size as u32,
                add_mod_component.info.n_constraints as u32,
                add_mod_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    add_mod_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << add_mod_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }

        println!("add_mod_builtin CUDA evaluator test completed successfully!");
    }

    /// CUDA trace generation + CPU constraint verification.
    #[test]
    fn test_add_mod_builtin_trace_gen_by_cuda_and_verify_by_cpu() {
        use cairo_air::relations;
        use cairo_air::components::add_mod_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_add_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for add_mod builtin segment
        use stwo_cairo_adapter::builtins::ADD_MOD_MEMORY_CELLS;

        let add_mod_segment = input.builtin_segments.add_mod.as_ref()
            .expect("Expected add_mod builtin segment");

        let segment_length = add_mod_segment.stop_ptr - add_mod_segment.begin_addr;
        let n_instances = segment_length / ADD_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = add_mod_segment.begin_addr as u32;

        println!("add_mod_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize CUDA generators
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_state.get_id(addr);
            memory_address_to_id_cuda_state.add_cuda_input(&addr);
            memory_id_to_big_cuda_state.add_cuda_input(&id);
        }

        // Create CUDA claim generator
        let add_mod_cuda_gen = add_mod_builtin_cuda::CudaClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (add_mod_claim, add_mod_interaction_gen) = add_mod_cuda_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        println!("add_mod_builtin_claim log_size: {:?}", add_mod_claim.log_size);

        // Interaction trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let add_mod_interaction_claim = add_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("add_mod_builtin_interaction_claim.claimed_sum: {:?}", add_mod_interaction_claim.claimed_sum);

        // Verify with CPU
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let add_mod_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("add_mod_builtin"),
                claim: add_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
            },
            add_mod_interaction_claim.claimed_sum,
        );

        assert_component(&add_mod_component, &trace);
        println!("add_mod_builtin CUDA trace gen + CPU verify test completed successfully!");
    }

    /// CUDA trace generation + CUDA constraint verification.
    #[test]
    fn test_add_mod_builtin_trace_gen_by_cuda_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::add_mod_builtin::{Component, Eval};
        use stwo::core::fields::m31::BaseField;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;
        use itertools::Itertools;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_add_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for add_mod builtin segment
        use stwo_cairo_adapter::builtins::ADD_MOD_MEMORY_CELLS;

        let add_mod_segment = input.builtin_segments.add_mod.as_ref()
            .expect("Expected add_mod builtin segment");

        let segment_length = add_mod_segment.stop_ptr - add_mod_segment.begin_addr;
        let n_instances = segment_length / ADD_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = add_mod_segment.begin_addr as u32;

        println!("add_mod_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize CUDA generators
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_state.get_id(addr);
            memory_address_to_id_cuda_state.add_cuda_input(&addr);
            memory_id_to_big_cuda_state.add_cuda_input(&id);
        }

        // Create CUDA claim generator
        let add_mod_cuda_gen = add_mod_builtin_cuda::CudaClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (add_mod_claim, add_mod_interaction_gen) = add_mod_cuda_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        println!("add_mod_builtin_claim log_size: {:?}", add_mod_claim.log_size);

        // Interaction trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let add_mod_interaction_claim = add_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("add_mod_builtin_interaction_claim.claimed_sum: {:?}", add_mod_interaction_claim.claimed_sum);

        // Convert trace to CUDA format for CUDA evaluator
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

        // Create component first to get actual constraint count
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let add_mod_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("add_mod_builtin"),
                claim: add_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
            },
            add_mod_interaction_claim.claimed_sum,
        );

        let domain_size = 1 << add_mod_claim.log_size;
        let n_constraints = add_mod_component.info.n_constraints.max(500);

        let mock_random_coeff_powers =
            BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv =
            BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 =
            BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 =
            BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 =
            BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 =
            BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let eval_ptr = &add_mod_component.eval as *const _ as *mut std::os::raw::c_void;
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
                add_mod_claim.log_size as u32,
                add_mod_claim.log_size as u32,
                add_mod_component.info.n_constraints as u32,
                add_mod_component
                    .info
                    .logup_counts
                    .iter()
                    .map(|(_, &count)| count)
                    .sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    add_mod_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << add_mod_claim.log_size),
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }
        println!("add_mod_builtin CUDA trace gen + CUDA verify test completed successfully!");
    }

    /// Compare CPU and CUDA traces column by column to identify discrepancies.
    #[test]
    fn test_add_mod_builtin_compare_cpu_vs_cuda_traces() {
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_add_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for add_mod builtin segment
        use stwo_cairo_adapter::builtins::ADD_MOD_MEMORY_CELLS;

        let add_mod_segment = input.builtin_segments.add_mod.as_ref()
            .expect("Expected add_mod builtin segment");

        let segment_length = add_mod_segment.stop_ptr - add_mod_segment.begin_addr;
        let n_instances = segment_length / ADD_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = add_mod_segment.begin_addr as u32;

        println!("add_mod_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // ============ CPU trace generation ============
        let memory_address_to_id_state_cpu = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state_cpu = memory_id_to_big::ClaimGenerator::new(&input.memory);

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_state_cpu.get_id(addr);
            memory_address_to_id_state_cpu.add_input(&addr);
            memory_id_to_big_state_cpu.add_input(&id);
        }

        let add_mod_claim_gen_cpu = add_mod_builtin::ClaimGenerator::new(log_size, segment_start);

        let mut mock_commitment_scheme_cpu = MockCommitmentScheme::default();
        let preprocessed_trace_cpu = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        mock_tree_builder_cpu.extend_evals(preprocessed_trace_cpu.gen_trace());
        mock_tree_builder_cpu.finalize_interaction();

        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        let (cpu_claim, _cpu_interaction_gen) = add_mod_claim_gen_cpu.write_trace(
            &mut mock_tree_builder_cpu,
            &memory_address_to_id_state_cpu,
            &memory_id_to_big_state_cpu,
        );
        mock_tree_builder_cpu.finalize_interaction();
        let cpu_trace = mock_commitment_scheme_cpu.trace_domain_evaluations();

        // ============ CUDA trace generation ============
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_cuda_state.get_id(addr);
            memory_address_to_id_cuda_state.add_cuda_input(&addr);
            memory_id_to_big_cuda_state.add_cuda_input(&id);
        }

        let add_mod_cuda_gen = add_mod_builtin_cuda::CudaClaimGenerator::new(log_size, segment_start);

        let mut mock_commitment_scheme_cuda = MockCommitmentScheme::default();
        let preprocessed_trace_cuda = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        mock_tree_builder_cuda.extend_evals(preprocessed_trace_cuda.gen_trace());
        mock_tree_builder_cuda.finalize_interaction();

        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        let (cuda_claim, _cuda_interaction_gen) = add_mod_cuda_gen.write_trace(
            &mut mock_tree_builder_cuda,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder_cuda.finalize_interaction();
        let cuda_trace = mock_commitment_scheme_cuda.trace_domain_evaluations();

        // ============ Compare traces ============
        println!("CPU log_size: {:?}, CUDA log_size: {:?}", cpu_claim.log_size, cuda_claim.log_size);
        assert_eq!(cpu_claim.log_size, cuda_claim.log_size);

        let trace_size = 1usize << cpu_claim.log_size;

        // Compare base trace columns (Tree 1)
        let cpu_base_trace = &cpu_trace[1];
        let cuda_base_trace = &cuda_trace[1];

        println!("CPU base trace columns: {:?}", cpu_base_trace.len());
        println!("CUDA base trace columns: {:?}", cuda_base_trace.len());

        let mut different_cols = Vec::new();
        for col in 0..std::cmp::min(cpu_base_trace.len(), cuda_base_trace.len()) {
            let cpu_col = &cpu_base_trace[col];
            let cuda_col = &cuda_base_trace[col];
            let cpu_vals: Vec<M31> = cpu_col.to_cpu().to_vec();
            let cuda_vals: Vec<M31> = cuda_col.to_cpu().to_vec();

            let mut diff_rows = Vec::new();
            for row in 0..trace_size {
                if cpu_vals[row] != cuda_vals[row] {
                    diff_rows.push((row, cpu_vals[row], cuda_vals[row]));
                }
            }

            if !diff_rows.is_empty() {
                different_cols.push(col);
                println!("\n=== Column {} has {} differences ===", col, diff_rows.len());
                for (row, cpu_val, cuda_val) in diff_rows.iter().take(5) {
                    println!("  Row {:3}: CPU={:10}, CUDA={:10}", row, cpu_val.0, cuda_val.0);
                }
                if diff_rows.len() > 5 {
                    println!("  ... and {} more differences", diff_rows.len() - 5);
                }
            }
        }

        if different_cols.is_empty() {
            println!("\nAll {} base trace columns match!", cpu_base_trace.len());
        } else {
            println!("\n=== SUMMARY: {} columns with differences: {:?} ===", different_cols.len(), different_cols);
        }

        assert!(different_cols.is_empty(), "Found differences in {} columns", different_cols.len());
    }
}
