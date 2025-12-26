#![allow(unused_parens)]
use cairo_air::components::mul_mod_builtin::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big};
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 426;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 94;

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
            Claim { log_size, mul_mod_builtin_segment_start: self.segment_start },
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
    Box<CudaLookupData>,
    CudaSubComponentInputs,
) {
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            Box::new(CudaLookupData {
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
                // 32 range_check_12 lookups (1 element each)
                range_check_12_0: init_lookup_array!(log_size),
                range_check_12_1: init_lookup_array!(log_size),
                range_check_12_2: init_lookup_array!(log_size),
                range_check_12_3: init_lookup_array!(log_size),
                range_check_12_4: init_lookup_array!(log_size),
                range_check_12_5: init_lookup_array!(log_size),
                range_check_12_6: init_lookup_array!(log_size),
                range_check_12_7: init_lookup_array!(log_size),
                range_check_12_8: init_lookup_array!(log_size),
                range_check_12_9: init_lookup_array!(log_size),
                range_check_12_10: init_lookup_array!(log_size),
                range_check_12_11: init_lookup_array!(log_size),
                range_check_12_12: init_lookup_array!(log_size),
                range_check_12_13: init_lookup_array!(log_size),
                range_check_12_14: init_lookup_array!(log_size),
                range_check_12_15: init_lookup_array!(log_size),
                range_check_12_16: init_lookup_array!(log_size),
                range_check_12_17: init_lookup_array!(log_size),
                range_check_12_18: init_lookup_array!(log_size),
                range_check_12_19: init_lookup_array!(log_size),
                range_check_12_20: init_lookup_array!(log_size),
                range_check_12_21: init_lookup_array!(log_size),
                range_check_12_22: init_lookup_array!(log_size),
                range_check_12_23: init_lookup_array!(log_size),
                range_check_12_24: init_lookup_array!(log_size),
                range_check_12_25: init_lookup_array!(log_size),
                range_check_12_26: init_lookup_array!(log_size),
                range_check_12_27: init_lookup_array!(log_size),
                range_check_12_28: init_lookup_array!(log_size),
                range_check_12_29: init_lookup_array!(log_size),
                range_check_12_30: init_lookup_array!(log_size),
                range_check_12_31: init_lookup_array!(log_size),
                // 62 range_check_18 lookups (1 element each)
                range_check_18_0: init_lookup_array!(log_size),
                range_check_18_1: init_lookup_array!(log_size),
                range_check_18_2: init_lookup_array!(log_size),
                range_check_18_3: init_lookup_array!(log_size),
                range_check_18_4: init_lookup_array!(log_size),
                range_check_18_5: init_lookup_array!(log_size),
                range_check_18_6: init_lookup_array!(log_size),
                range_check_18_7: init_lookup_array!(log_size),
                range_check_18_8: init_lookup_array!(log_size),
                range_check_18_9: init_lookup_array!(log_size),
                range_check_18_10: init_lookup_array!(log_size),
                range_check_18_11: init_lookup_array!(log_size),
                range_check_18_12: init_lookup_array!(log_size),
                range_check_18_13: init_lookup_array!(log_size),
                range_check_18_14: init_lookup_array!(log_size),
                range_check_18_15: init_lookup_array!(log_size),
                range_check_18_16: init_lookup_array!(log_size),
                range_check_18_17: init_lookup_array!(log_size),
                range_check_18_18: init_lookup_array!(log_size),
                range_check_18_19: init_lookup_array!(log_size),
                range_check_18_20: init_lookup_array!(log_size),
                range_check_18_21: init_lookup_array!(log_size),
                range_check_18_22: init_lookup_array!(log_size),
                range_check_18_23: init_lookup_array!(log_size),
                range_check_18_24: init_lookup_array!(log_size),
                range_check_18_25: init_lookup_array!(log_size),
                range_check_18_26: init_lookup_array!(log_size),
                range_check_18_27: init_lookup_array!(log_size),
                range_check_18_28: init_lookup_array!(log_size),
                range_check_18_29: init_lookup_array!(log_size),
                range_check_18_30: init_lookup_array!(log_size),
                range_check_18_31: init_lookup_array!(log_size),
                range_check_18_32: init_lookup_array!(log_size),
                range_check_18_33: init_lookup_array!(log_size),
                range_check_18_34: init_lookup_array!(log_size),
                range_check_18_35: init_lookup_array!(log_size),
                range_check_18_36: init_lookup_array!(log_size),
                range_check_18_37: init_lookup_array!(log_size),
                range_check_18_38: init_lookup_array!(log_size),
                range_check_18_39: init_lookup_array!(log_size),
                range_check_18_40: init_lookup_array!(log_size),
                range_check_18_41: init_lookup_array!(log_size),
                range_check_18_42: init_lookup_array!(log_size),
                range_check_18_43: init_lookup_array!(log_size),
                range_check_18_44: init_lookup_array!(log_size),
                range_check_18_45: init_lookup_array!(log_size),
                range_check_18_46: init_lookup_array!(log_size),
                range_check_18_47: init_lookup_array!(log_size),
                range_check_18_48: init_lookup_array!(log_size),
                range_check_18_49: init_lookup_array!(log_size),
                range_check_18_50: init_lookup_array!(log_size),
                range_check_18_51: init_lookup_array!(log_size),
                range_check_18_52: init_lookup_array!(log_size),
                range_check_18_53: init_lookup_array!(log_size),
                range_check_18_54: init_lookup_array!(log_size),
                range_check_18_55: init_lookup_array!(log_size),
                range_check_18_56: init_lookup_array!(log_size),
                range_check_18_57: init_lookup_array!(log_size),
                range_check_18_58: init_lookup_array!(log_size),
                range_check_18_59: init_lookup_array!(log_size),
                range_check_18_60: init_lookup_array!(log_size),
                range_check_18_61: init_lookup_array!(log_size),
                // 40 range_check_3_6_6_3 lookups (4 elements each)
                range_check_3_6_6_3_0: init_lookup_array!(log_size),
                range_check_3_6_6_3_1: init_lookup_array!(log_size),
                range_check_3_6_6_3_2: init_lookup_array!(log_size),
                range_check_3_6_6_3_3: init_lookup_array!(log_size),
                range_check_3_6_6_3_4: init_lookup_array!(log_size),
                range_check_3_6_6_3_5: init_lookup_array!(log_size),
                range_check_3_6_6_3_6: init_lookup_array!(log_size),
                range_check_3_6_6_3_7: init_lookup_array!(log_size),
                range_check_3_6_6_3_8: init_lookup_array!(log_size),
                range_check_3_6_6_3_9: init_lookup_array!(log_size),
                range_check_3_6_6_3_10: init_lookup_array!(log_size),
                range_check_3_6_6_3_11: init_lookup_array!(log_size),
                range_check_3_6_6_3_12: init_lookup_array!(log_size),
                range_check_3_6_6_3_13: init_lookup_array!(log_size),
                range_check_3_6_6_3_14: init_lookup_array!(log_size),
                range_check_3_6_6_3_15: init_lookup_array!(log_size),
                range_check_3_6_6_3_16: init_lookup_array!(log_size),
                range_check_3_6_6_3_17: init_lookup_array!(log_size),
                range_check_3_6_6_3_18: init_lookup_array!(log_size),
                range_check_3_6_6_3_19: init_lookup_array!(log_size),
                range_check_3_6_6_3_20: init_lookup_array!(log_size),
                range_check_3_6_6_3_21: init_lookup_array!(log_size),
                range_check_3_6_6_3_22: init_lookup_array!(log_size),
                range_check_3_6_6_3_23: init_lookup_array!(log_size),
                range_check_3_6_6_3_24: init_lookup_array!(log_size),
                range_check_3_6_6_3_25: init_lookup_array!(log_size),
                range_check_3_6_6_3_26: init_lookup_array!(log_size),
                range_check_3_6_6_3_27: init_lookup_array!(log_size),
                range_check_3_6_6_3_28: init_lookup_array!(log_size),
                range_check_3_6_6_3_29: init_lookup_array!(log_size),
                range_check_3_6_6_3_30: init_lookup_array!(log_size),
                range_check_3_6_6_3_31: init_lookup_array!(log_size),
                range_check_3_6_6_3_32: init_lookup_array!(log_size),
                range_check_3_6_6_3_33: init_lookup_array!(log_size),
                range_check_3_6_6_3_34: init_lookup_array!(log_size),
                range_check_3_6_6_3_35: init_lookup_array!(log_size),
                range_check_3_6_6_3_36: init_lookup_array!(log_size),
                range_check_3_6_6_3_37: init_lookup_array!(log_size),
                range_check_3_6_6_3_38: init_lookup_array!(log_size),
                range_check_3_6_6_3_39: init_lookup_array!(log_size),
            }),
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

    // Collect lookup data pointers - 32 range_check_12
    let lookup_range_check_12_0 = collect_lookup_ptrs!(lookup_data, range_check_12_0);
    let lookup_range_check_12_1 = collect_lookup_ptrs!(lookup_data, range_check_12_1);
    let lookup_range_check_12_2 = collect_lookup_ptrs!(lookup_data, range_check_12_2);
    let lookup_range_check_12_3 = collect_lookup_ptrs!(lookup_data, range_check_12_3);
    let lookup_range_check_12_4 = collect_lookup_ptrs!(lookup_data, range_check_12_4);
    let lookup_range_check_12_5 = collect_lookup_ptrs!(lookup_data, range_check_12_5);
    let lookup_range_check_12_6 = collect_lookup_ptrs!(lookup_data, range_check_12_6);
    let lookup_range_check_12_7 = collect_lookup_ptrs!(lookup_data, range_check_12_7);
    let lookup_range_check_12_8 = collect_lookup_ptrs!(lookup_data, range_check_12_8);
    let lookup_range_check_12_9 = collect_lookup_ptrs!(lookup_data, range_check_12_9);
    let lookup_range_check_12_10 = collect_lookup_ptrs!(lookup_data, range_check_12_10);
    let lookup_range_check_12_11 = collect_lookup_ptrs!(lookup_data, range_check_12_11);
    let lookup_range_check_12_12 = collect_lookup_ptrs!(lookup_data, range_check_12_12);
    let lookup_range_check_12_13 = collect_lookup_ptrs!(lookup_data, range_check_12_13);
    let lookup_range_check_12_14 = collect_lookup_ptrs!(lookup_data, range_check_12_14);
    let lookup_range_check_12_15 = collect_lookup_ptrs!(lookup_data, range_check_12_15);
    let lookup_range_check_12_16 = collect_lookup_ptrs!(lookup_data, range_check_12_16);
    let lookup_range_check_12_17 = collect_lookup_ptrs!(lookup_data, range_check_12_17);
    let lookup_range_check_12_18 = collect_lookup_ptrs!(lookup_data, range_check_12_18);
    let lookup_range_check_12_19 = collect_lookup_ptrs!(lookup_data, range_check_12_19);
    let lookup_range_check_12_20 = collect_lookup_ptrs!(lookup_data, range_check_12_20);
    let lookup_range_check_12_21 = collect_lookup_ptrs!(lookup_data, range_check_12_21);
    let lookup_range_check_12_22 = collect_lookup_ptrs!(lookup_data, range_check_12_22);
    let lookup_range_check_12_23 = collect_lookup_ptrs!(lookup_data, range_check_12_23);
    let lookup_range_check_12_24 = collect_lookup_ptrs!(lookup_data, range_check_12_24);
    let lookup_range_check_12_25 = collect_lookup_ptrs!(lookup_data, range_check_12_25);
    let lookup_range_check_12_26 = collect_lookup_ptrs!(lookup_data, range_check_12_26);
    let lookup_range_check_12_27 = collect_lookup_ptrs!(lookup_data, range_check_12_27);
    let lookup_range_check_12_28 = collect_lookup_ptrs!(lookup_data, range_check_12_28);
    let lookup_range_check_12_29 = collect_lookup_ptrs!(lookup_data, range_check_12_29);
    let lookup_range_check_12_30 = collect_lookup_ptrs!(lookup_data, range_check_12_30);
    let lookup_range_check_12_31 = collect_lookup_ptrs!(lookup_data, range_check_12_31);

    // Collect lookup data pointers - 62 range_check_18
    let lookup_range_check_18_0 = collect_lookup_ptrs!(lookup_data, range_check_18_0);
    let lookup_range_check_18_1 = collect_lookup_ptrs!(lookup_data, range_check_18_1);
    let lookup_range_check_18_2 = collect_lookup_ptrs!(lookup_data, range_check_18_2);
    let lookup_range_check_18_3 = collect_lookup_ptrs!(lookup_data, range_check_18_3);
    let lookup_range_check_18_4 = collect_lookup_ptrs!(lookup_data, range_check_18_4);
    let lookup_range_check_18_5 = collect_lookup_ptrs!(lookup_data, range_check_18_5);
    let lookup_range_check_18_6 = collect_lookup_ptrs!(lookup_data, range_check_18_6);
    let lookup_range_check_18_7 = collect_lookup_ptrs!(lookup_data, range_check_18_7);
    let lookup_range_check_18_8 = collect_lookup_ptrs!(lookup_data, range_check_18_8);
    let lookup_range_check_18_9 = collect_lookup_ptrs!(lookup_data, range_check_18_9);
    let lookup_range_check_18_10 = collect_lookup_ptrs!(lookup_data, range_check_18_10);
    let lookup_range_check_18_11 = collect_lookup_ptrs!(lookup_data, range_check_18_11);
    let lookup_range_check_18_12 = collect_lookup_ptrs!(lookup_data, range_check_18_12);
    let lookup_range_check_18_13 = collect_lookup_ptrs!(lookup_data, range_check_18_13);
    let lookup_range_check_18_14 = collect_lookup_ptrs!(lookup_data, range_check_18_14);
    let lookup_range_check_18_15 = collect_lookup_ptrs!(lookup_data, range_check_18_15);
    let lookup_range_check_18_16 = collect_lookup_ptrs!(lookup_data, range_check_18_16);
    let lookup_range_check_18_17 = collect_lookup_ptrs!(lookup_data, range_check_18_17);
    let lookup_range_check_18_18 = collect_lookup_ptrs!(lookup_data, range_check_18_18);
    let lookup_range_check_18_19 = collect_lookup_ptrs!(lookup_data, range_check_18_19);
    let lookup_range_check_18_20 = collect_lookup_ptrs!(lookup_data, range_check_18_20);
    let lookup_range_check_18_21 = collect_lookup_ptrs!(lookup_data, range_check_18_21);
    let lookup_range_check_18_22 = collect_lookup_ptrs!(lookup_data, range_check_18_22);
    let lookup_range_check_18_23 = collect_lookup_ptrs!(lookup_data, range_check_18_23);
    let lookup_range_check_18_24 = collect_lookup_ptrs!(lookup_data, range_check_18_24);
    let lookup_range_check_18_25 = collect_lookup_ptrs!(lookup_data, range_check_18_25);
    let lookup_range_check_18_26 = collect_lookup_ptrs!(lookup_data, range_check_18_26);
    let lookup_range_check_18_27 = collect_lookup_ptrs!(lookup_data, range_check_18_27);
    let lookup_range_check_18_28 = collect_lookup_ptrs!(lookup_data, range_check_18_28);
    let lookup_range_check_18_29 = collect_lookup_ptrs!(lookup_data, range_check_18_29);
    let lookup_range_check_18_30 = collect_lookup_ptrs!(lookup_data, range_check_18_30);
    let lookup_range_check_18_31 = collect_lookup_ptrs!(lookup_data, range_check_18_31);
    let lookup_range_check_18_32 = collect_lookup_ptrs!(lookup_data, range_check_18_32);
    let lookup_range_check_18_33 = collect_lookup_ptrs!(lookup_data, range_check_18_33);
    let lookup_range_check_18_34 = collect_lookup_ptrs!(lookup_data, range_check_18_34);
    let lookup_range_check_18_35 = collect_lookup_ptrs!(lookup_data, range_check_18_35);
    let lookup_range_check_18_36 = collect_lookup_ptrs!(lookup_data, range_check_18_36);
    let lookup_range_check_18_37 = collect_lookup_ptrs!(lookup_data, range_check_18_37);
    let lookup_range_check_18_38 = collect_lookup_ptrs!(lookup_data, range_check_18_38);
    let lookup_range_check_18_39 = collect_lookup_ptrs!(lookup_data, range_check_18_39);
    let lookup_range_check_18_40 = collect_lookup_ptrs!(lookup_data, range_check_18_40);
    let lookup_range_check_18_41 = collect_lookup_ptrs!(lookup_data, range_check_18_41);
    let lookup_range_check_18_42 = collect_lookup_ptrs!(lookup_data, range_check_18_42);
    let lookup_range_check_18_43 = collect_lookup_ptrs!(lookup_data, range_check_18_43);
    let lookup_range_check_18_44 = collect_lookup_ptrs!(lookup_data, range_check_18_44);
    let lookup_range_check_18_45 = collect_lookup_ptrs!(lookup_data, range_check_18_45);
    let lookup_range_check_18_46 = collect_lookup_ptrs!(lookup_data, range_check_18_46);
    let lookup_range_check_18_47 = collect_lookup_ptrs!(lookup_data, range_check_18_47);
    let lookup_range_check_18_48 = collect_lookup_ptrs!(lookup_data, range_check_18_48);
    let lookup_range_check_18_49 = collect_lookup_ptrs!(lookup_data, range_check_18_49);
    let lookup_range_check_18_50 = collect_lookup_ptrs!(lookup_data, range_check_18_50);
    let lookup_range_check_18_51 = collect_lookup_ptrs!(lookup_data, range_check_18_51);
    let lookup_range_check_18_52 = collect_lookup_ptrs!(lookup_data, range_check_18_52);
    let lookup_range_check_18_53 = collect_lookup_ptrs!(lookup_data, range_check_18_53);
    let lookup_range_check_18_54 = collect_lookup_ptrs!(lookup_data, range_check_18_54);
    let lookup_range_check_18_55 = collect_lookup_ptrs!(lookup_data, range_check_18_55);
    let lookup_range_check_18_56 = collect_lookup_ptrs!(lookup_data, range_check_18_56);
    let lookup_range_check_18_57 = collect_lookup_ptrs!(lookup_data, range_check_18_57);
    let lookup_range_check_18_58 = collect_lookup_ptrs!(lookup_data, range_check_18_58);
    let lookup_range_check_18_59 = collect_lookup_ptrs!(lookup_data, range_check_18_59);
    let lookup_range_check_18_60 = collect_lookup_ptrs!(lookup_data, range_check_18_60);
    let lookup_range_check_18_61 = collect_lookup_ptrs!(lookup_data, range_check_18_61);

    // Collect lookup data pointers - 40 range_check_3_6_6_3
    let lookup_range_check_3_6_6_3_0 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_0);
    let lookup_range_check_3_6_6_3_1 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_1);
    let lookup_range_check_3_6_6_3_2 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_2);
    let lookup_range_check_3_6_6_3_3 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_3);
    let lookup_range_check_3_6_6_3_4 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_4);
    let lookup_range_check_3_6_6_3_5 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_5);
    let lookup_range_check_3_6_6_3_6 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_6);
    let lookup_range_check_3_6_6_3_7 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_7);
    let lookup_range_check_3_6_6_3_8 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_8);
    let lookup_range_check_3_6_6_3_9 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_9);
    let lookup_range_check_3_6_6_3_10 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_10);
    let lookup_range_check_3_6_6_3_11 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_11);
    let lookup_range_check_3_6_6_3_12 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_12);
    let lookup_range_check_3_6_6_3_13 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_13);
    let lookup_range_check_3_6_6_3_14 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_14);
    let lookup_range_check_3_6_6_3_15 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_15);
    let lookup_range_check_3_6_6_3_16 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_16);
    let lookup_range_check_3_6_6_3_17 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_17);
    let lookup_range_check_3_6_6_3_18 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_18);
    let lookup_range_check_3_6_6_3_19 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_19);
    let lookup_range_check_3_6_6_3_20 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_20);
    let lookup_range_check_3_6_6_3_21 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_21);
    let lookup_range_check_3_6_6_3_22 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_22);
    let lookup_range_check_3_6_6_3_23 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_23);
    let lookup_range_check_3_6_6_3_24 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_24);
    let lookup_range_check_3_6_6_3_25 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_25);
    let lookup_range_check_3_6_6_3_26 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_26);
    let lookup_range_check_3_6_6_3_27 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_27);
    let lookup_range_check_3_6_6_3_28 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_28);
    let lookup_range_check_3_6_6_3_29 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_29);
    let lookup_range_check_3_6_6_3_30 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_30);
    let lookup_range_check_3_6_6_3_31 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_31);
    let lookup_range_check_3_6_6_3_32 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_32);
    let lookup_range_check_3_6_6_3_33 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_33);
    let lookup_range_check_3_6_6_3_34 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_34);
    let lookup_range_check_3_6_6_3_35 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_35);
    let lookup_range_check_3_6_6_3_36 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_36);
    let lookup_range_check_3_6_6_3_37 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_37);
    let lookup_range_check_3_6_6_3_38 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_38);
    let lookup_range_check_3_6_6_3_39 = collect_lookup_ptrs!(lookup_data, range_check_3_6_6_3_39);

    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    // Range check sub-component inputs are not needed for mul_mod_builtin base trace generation
    // (range checks don't have sub-component inputs to pass to the kernel)
    let sub_component_inputs_range_check_12_vec: Vec<*const u32> = vec![];
    let sub_component_inputs_range_check_18_vec: Vec<*const u32> = vec![];
    let sub_component_inputs_range_check_3_6_6_3_vec: Vec<*const u32> = vec![];

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_mul_mod_builtin_traces(
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

            // 32 range_check_12 lookups
            lookup_range_check_12_0.as_ptr(),
            lookup_range_check_12_1.as_ptr(),
            lookup_range_check_12_2.as_ptr(),
            lookup_range_check_12_3.as_ptr(),
            lookup_range_check_12_4.as_ptr(),
            lookup_range_check_12_5.as_ptr(),
            lookup_range_check_12_6.as_ptr(),
            lookup_range_check_12_7.as_ptr(),
            lookup_range_check_12_8.as_ptr(),
            lookup_range_check_12_9.as_ptr(),
            lookup_range_check_12_10.as_ptr(),
            lookup_range_check_12_11.as_ptr(),
            lookup_range_check_12_12.as_ptr(),
            lookup_range_check_12_13.as_ptr(),
            lookup_range_check_12_14.as_ptr(),
            lookup_range_check_12_15.as_ptr(),
            lookup_range_check_12_16.as_ptr(),
            lookup_range_check_12_17.as_ptr(),
            lookup_range_check_12_18.as_ptr(),
            lookup_range_check_12_19.as_ptr(),
            lookup_range_check_12_20.as_ptr(),
            lookup_range_check_12_21.as_ptr(),
            lookup_range_check_12_22.as_ptr(),
            lookup_range_check_12_23.as_ptr(),
            lookup_range_check_12_24.as_ptr(),
            lookup_range_check_12_25.as_ptr(),
            lookup_range_check_12_26.as_ptr(),
            lookup_range_check_12_27.as_ptr(),
            lookup_range_check_12_28.as_ptr(),
            lookup_range_check_12_29.as_ptr(),
            lookup_range_check_12_30.as_ptr(),
            lookup_range_check_12_31.as_ptr(),

            // 62 range_check_18 lookups
            lookup_range_check_18_0.as_ptr(),
            lookup_range_check_18_1.as_ptr(),
            lookup_range_check_18_2.as_ptr(),
            lookup_range_check_18_3.as_ptr(),
            lookup_range_check_18_4.as_ptr(),
            lookup_range_check_18_5.as_ptr(),
            lookup_range_check_18_6.as_ptr(),
            lookup_range_check_18_7.as_ptr(),
            lookup_range_check_18_8.as_ptr(),
            lookup_range_check_18_9.as_ptr(),
            lookup_range_check_18_10.as_ptr(),
            lookup_range_check_18_11.as_ptr(),
            lookup_range_check_18_12.as_ptr(),
            lookup_range_check_18_13.as_ptr(),
            lookup_range_check_18_14.as_ptr(),
            lookup_range_check_18_15.as_ptr(),
            lookup_range_check_18_16.as_ptr(),
            lookup_range_check_18_17.as_ptr(),
            lookup_range_check_18_18.as_ptr(),
            lookup_range_check_18_19.as_ptr(),
            lookup_range_check_18_20.as_ptr(),
            lookup_range_check_18_21.as_ptr(),
            lookup_range_check_18_22.as_ptr(),
            lookup_range_check_18_23.as_ptr(),
            lookup_range_check_18_24.as_ptr(),
            lookup_range_check_18_25.as_ptr(),
            lookup_range_check_18_26.as_ptr(),
            lookup_range_check_18_27.as_ptr(),
            lookup_range_check_18_28.as_ptr(),
            lookup_range_check_18_29.as_ptr(),
            lookup_range_check_18_30.as_ptr(),
            lookup_range_check_18_31.as_ptr(),
            lookup_range_check_18_32.as_ptr(),
            lookup_range_check_18_33.as_ptr(),
            lookup_range_check_18_34.as_ptr(),
            lookup_range_check_18_35.as_ptr(),
            lookup_range_check_18_36.as_ptr(),
            lookup_range_check_18_37.as_ptr(),
            lookup_range_check_18_38.as_ptr(),
            lookup_range_check_18_39.as_ptr(),
            lookup_range_check_18_40.as_ptr(),
            lookup_range_check_18_41.as_ptr(),
            lookup_range_check_18_42.as_ptr(),
            lookup_range_check_18_43.as_ptr(),
            lookup_range_check_18_44.as_ptr(),
            lookup_range_check_18_45.as_ptr(),
            lookup_range_check_18_46.as_ptr(),
            lookup_range_check_18_47.as_ptr(),
            lookup_range_check_18_48.as_ptr(),
            lookup_range_check_18_49.as_ptr(),
            lookup_range_check_18_50.as_ptr(),
            lookup_range_check_18_51.as_ptr(),
            lookup_range_check_18_52.as_ptr(),
            lookup_range_check_18_53.as_ptr(),
            lookup_range_check_18_54.as_ptr(),
            lookup_range_check_18_55.as_ptr(),
            lookup_range_check_18_56.as_ptr(),
            lookup_range_check_18_57.as_ptr(),
            lookup_range_check_18_58.as_ptr(),
            lookup_range_check_18_59.as_ptr(),
            lookup_range_check_18_60.as_ptr(),
            lookup_range_check_18_61.as_ptr(),

            // 40 range_check_3_6_6_3 lookups
            lookup_range_check_3_6_6_3_0.as_ptr(),
            lookup_range_check_3_6_6_3_1.as_ptr(),
            lookup_range_check_3_6_6_3_2.as_ptr(),
            lookup_range_check_3_6_6_3_3.as_ptr(),
            lookup_range_check_3_6_6_3_4.as_ptr(),
            lookup_range_check_3_6_6_3_5.as_ptr(),
            lookup_range_check_3_6_6_3_6.as_ptr(),
            lookup_range_check_3_6_6_3_7.as_ptr(),
            lookup_range_check_3_6_6_3_8.as_ptr(),
            lookup_range_check_3_6_6_3_9.as_ptr(),
            lookup_range_check_3_6_6_3_10.as_ptr(),
            lookup_range_check_3_6_6_3_11.as_ptr(),
            lookup_range_check_3_6_6_3_12.as_ptr(),
            lookup_range_check_3_6_6_3_13.as_ptr(),
            lookup_range_check_3_6_6_3_14.as_ptr(),
            lookup_range_check_3_6_6_3_15.as_ptr(),
            lookup_range_check_3_6_6_3_16.as_ptr(),
            lookup_range_check_3_6_6_3_17.as_ptr(),
            lookup_range_check_3_6_6_3_18.as_ptr(),
            lookup_range_check_3_6_6_3_19.as_ptr(),
            lookup_range_check_3_6_6_3_20.as_ptr(),
            lookup_range_check_3_6_6_3_21.as_ptr(),
            lookup_range_check_3_6_6_3_22.as_ptr(),
            lookup_range_check_3_6_6_3_23.as_ptr(),
            lookup_range_check_3_6_6_3_24.as_ptr(),
            lookup_range_check_3_6_6_3_25.as_ptr(),
            lookup_range_check_3_6_6_3_26.as_ptr(),
            lookup_range_check_3_6_6_3_27.as_ptr(),
            lookup_range_check_3_6_6_3_28.as_ptr(),
            lookup_range_check_3_6_6_3_29.as_ptr(),
            lookup_range_check_3_6_6_3_30.as_ptr(),
            lookup_range_check_3_6_6_3_31.as_ptr(),
            lookup_range_check_3_6_6_3_32.as_ptr(),
            lookup_range_check_3_6_6_3_33.as_ptr(),
            lookup_range_check_3_6_6_3_34.as_ptr(),
            lookup_range_check_3_6_6_3_35.as_ptr(),
            lookup_range_check_3_6_6_3_36.as_ptr(),
            lookup_range_check_3_6_6_3_37.as_ptr(),
            lookup_range_check_3_6_6_3_38.as_ptr(),
            lookup_range_check_3_6_6_3_39.as_ptr(),

            // Sub-component inputs
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            sub_component_inputs_range_check_12_vec.as_ptr(),
            sub_component_inputs_range_check_18_vec.as_ptr(),
            sub_component_inputs_range_check_3_6_6_3_vec.as_ptr(),

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
    // 32 range_check_12 lookups (1 element each)
    range_check_12_0: [BaseFieldVec; 1],
    range_check_12_1: [BaseFieldVec; 1],
    range_check_12_2: [BaseFieldVec; 1],
    range_check_12_3: [BaseFieldVec; 1],
    range_check_12_4: [BaseFieldVec; 1],
    range_check_12_5: [BaseFieldVec; 1],
    range_check_12_6: [BaseFieldVec; 1],
    range_check_12_7: [BaseFieldVec; 1],
    range_check_12_8: [BaseFieldVec; 1],
    range_check_12_9: [BaseFieldVec; 1],
    range_check_12_10: [BaseFieldVec; 1],
    range_check_12_11: [BaseFieldVec; 1],
    range_check_12_12: [BaseFieldVec; 1],
    range_check_12_13: [BaseFieldVec; 1],
    range_check_12_14: [BaseFieldVec; 1],
    range_check_12_15: [BaseFieldVec; 1],
    range_check_12_16: [BaseFieldVec; 1],
    range_check_12_17: [BaseFieldVec; 1],
    range_check_12_18: [BaseFieldVec; 1],
    range_check_12_19: [BaseFieldVec; 1],
    range_check_12_20: [BaseFieldVec; 1],
    range_check_12_21: [BaseFieldVec; 1],
    range_check_12_22: [BaseFieldVec; 1],
    range_check_12_23: [BaseFieldVec; 1],
    range_check_12_24: [BaseFieldVec; 1],
    range_check_12_25: [BaseFieldVec; 1],
    range_check_12_26: [BaseFieldVec; 1],
    range_check_12_27: [BaseFieldVec; 1],
    range_check_12_28: [BaseFieldVec; 1],
    range_check_12_29: [BaseFieldVec; 1],
    range_check_12_30: [BaseFieldVec; 1],
    range_check_12_31: [BaseFieldVec; 1],
    // 62 range_check_18 lookups (1 element each)
    range_check_18_0: [BaseFieldVec; 1],
    range_check_18_1: [BaseFieldVec; 1],
    range_check_18_2: [BaseFieldVec; 1],
    range_check_18_3: [BaseFieldVec; 1],
    range_check_18_4: [BaseFieldVec; 1],
    range_check_18_5: [BaseFieldVec; 1],
    range_check_18_6: [BaseFieldVec; 1],
    range_check_18_7: [BaseFieldVec; 1],
    range_check_18_8: [BaseFieldVec; 1],
    range_check_18_9: [BaseFieldVec; 1],
    range_check_18_10: [BaseFieldVec; 1],
    range_check_18_11: [BaseFieldVec; 1],
    range_check_18_12: [BaseFieldVec; 1],
    range_check_18_13: [BaseFieldVec; 1],
    range_check_18_14: [BaseFieldVec; 1],
    range_check_18_15: [BaseFieldVec; 1],
    range_check_18_16: [BaseFieldVec; 1],
    range_check_18_17: [BaseFieldVec; 1],
    range_check_18_18: [BaseFieldVec; 1],
    range_check_18_19: [BaseFieldVec; 1],
    range_check_18_20: [BaseFieldVec; 1],
    range_check_18_21: [BaseFieldVec; 1],
    range_check_18_22: [BaseFieldVec; 1],
    range_check_18_23: [BaseFieldVec; 1],
    range_check_18_24: [BaseFieldVec; 1],
    range_check_18_25: [BaseFieldVec; 1],
    range_check_18_26: [BaseFieldVec; 1],
    range_check_18_27: [BaseFieldVec; 1],
    range_check_18_28: [BaseFieldVec; 1],
    range_check_18_29: [BaseFieldVec; 1],
    range_check_18_30: [BaseFieldVec; 1],
    range_check_18_31: [BaseFieldVec; 1],
    range_check_18_32: [BaseFieldVec; 1],
    range_check_18_33: [BaseFieldVec; 1],
    range_check_18_34: [BaseFieldVec; 1],
    range_check_18_35: [BaseFieldVec; 1],
    range_check_18_36: [BaseFieldVec; 1],
    range_check_18_37: [BaseFieldVec; 1],
    range_check_18_38: [BaseFieldVec; 1],
    range_check_18_39: [BaseFieldVec; 1],
    range_check_18_40: [BaseFieldVec; 1],
    range_check_18_41: [BaseFieldVec; 1],
    range_check_18_42: [BaseFieldVec; 1],
    range_check_18_43: [BaseFieldVec; 1],
    range_check_18_44: [BaseFieldVec; 1],
    range_check_18_45: [BaseFieldVec; 1],
    range_check_18_46: [BaseFieldVec; 1],
    range_check_18_47: [BaseFieldVec; 1],
    range_check_18_48: [BaseFieldVec; 1],
    range_check_18_49: [BaseFieldVec; 1],
    range_check_18_50: [BaseFieldVec; 1],
    range_check_18_51: [BaseFieldVec; 1],
    range_check_18_52: [BaseFieldVec; 1],
    range_check_18_53: [BaseFieldVec; 1],
    range_check_18_54: [BaseFieldVec; 1],
    range_check_18_55: [BaseFieldVec; 1],
    range_check_18_56: [BaseFieldVec; 1],
    range_check_18_57: [BaseFieldVec; 1],
    range_check_18_58: [BaseFieldVec; 1],
    range_check_18_59: [BaseFieldVec; 1],
    range_check_18_60: [BaseFieldVec; 1],
    range_check_18_61: [BaseFieldVec; 1],
    // 40 range_check_3_6_6_3 lookups (4 elements each)
    range_check_3_6_6_3_0: [BaseFieldVec; 4],
    range_check_3_6_6_3_1: [BaseFieldVec; 4],
    range_check_3_6_6_3_2: [BaseFieldVec; 4],
    range_check_3_6_6_3_3: [BaseFieldVec; 4],
    range_check_3_6_6_3_4: [BaseFieldVec; 4],
    range_check_3_6_6_3_5: [BaseFieldVec; 4],
    range_check_3_6_6_3_6: [BaseFieldVec; 4],
    range_check_3_6_6_3_7: [BaseFieldVec; 4],
    range_check_3_6_6_3_8: [BaseFieldVec; 4],
    range_check_3_6_6_3_9: [BaseFieldVec; 4],
    range_check_3_6_6_3_10: [BaseFieldVec; 4],
    range_check_3_6_6_3_11: [BaseFieldVec; 4],
    range_check_3_6_6_3_12: [BaseFieldVec; 4],
    range_check_3_6_6_3_13: [BaseFieldVec; 4],
    range_check_3_6_6_3_14: [BaseFieldVec; 4],
    range_check_3_6_6_3_15: [BaseFieldVec; 4],
    range_check_3_6_6_3_16: [BaseFieldVec; 4],
    range_check_3_6_6_3_17: [BaseFieldVec; 4],
    range_check_3_6_6_3_18: [BaseFieldVec; 4],
    range_check_3_6_6_3_19: [BaseFieldVec; 4],
    range_check_3_6_6_3_20: [BaseFieldVec; 4],
    range_check_3_6_6_3_21: [BaseFieldVec; 4],
    range_check_3_6_6_3_22: [BaseFieldVec; 4],
    range_check_3_6_6_3_23: [BaseFieldVec; 4],
    range_check_3_6_6_3_24: [BaseFieldVec; 4],
    range_check_3_6_6_3_25: [BaseFieldVec; 4],
    range_check_3_6_6_3_26: [BaseFieldVec; 4],
    range_check_3_6_6_3_27: [BaseFieldVec; 4],
    range_check_3_6_6_3_28: [BaseFieldVec; 4],
    range_check_3_6_6_3_29: [BaseFieldVec; 4],
    range_check_3_6_6_3_30: [BaseFieldVec; 4],
    range_check_3_6_6_3_31: [BaseFieldVec; 4],
    range_check_3_6_6_3_32: [BaseFieldVec; 4],
    range_check_3_6_6_3_33: [BaseFieldVec; 4],
    range_check_3_6_6_3_34: [BaseFieldVec; 4],
    range_check_3_6_6_3_35: [BaseFieldVec; 4],
    range_check_3_6_6_3_36: [BaseFieldVec; 4],
    range_check_3_6_6_3_37: [BaseFieldVec; 4],
    range_check_3_6_6_3_38: [BaseFieldVec; 4],
    range_check_3_6_6_3_39: [BaseFieldVec; 4],
}

pub struct CudaInteractionClaimGenerator {
    n_rows: usize,
    log_size: u32,
    lookup_data: Box<CudaLookupData>,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        range_check_12: &relations::RangeCheck_12,
        range_check_3_6_6_3: &relations::RangeCheck_3_6_6_3,
        range_check_18: &relations::RangeCheck_18,
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

        // Collect lookup data pointers - 32 range_check_12
        let lookup_range_check_12_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_0);
        let lookup_range_check_12_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_1);
        let lookup_range_check_12_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_2);
        let lookup_range_check_12_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_3);
        let lookup_range_check_12_4_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_4);
        let lookup_range_check_12_5_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_5);
        let lookup_range_check_12_6_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_6);
        let lookup_range_check_12_7_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_7);
        let lookup_range_check_12_8_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_8);
        let lookup_range_check_12_9_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_9);
        let lookup_range_check_12_10_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_10);
        let lookup_range_check_12_11_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_11);
        let lookup_range_check_12_12_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_12);
        let lookup_range_check_12_13_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_13);
        let lookup_range_check_12_14_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_14);
        let lookup_range_check_12_15_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_15);
        let lookup_range_check_12_16_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_16);
        let lookup_range_check_12_17_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_17);
        let lookup_range_check_12_18_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_18);
        let lookup_range_check_12_19_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_19);
        let lookup_range_check_12_20_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_20);
        let lookup_range_check_12_21_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_21);
        let lookup_range_check_12_22_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_22);
        let lookup_range_check_12_23_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_23);
        let lookup_range_check_12_24_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_24);
        let lookup_range_check_12_25_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_25);
        let lookup_range_check_12_26_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_26);
        let lookup_range_check_12_27_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_27);
        let lookup_range_check_12_28_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_28);
        let lookup_range_check_12_29_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_29);
        let lookup_range_check_12_30_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_30);
        let lookup_range_check_12_31_vec = collect_lookup_ptrs!(self.lookup_data, range_check_12_31);

        // Collect lookup data pointers - 62 range_check_18
        let lookup_range_check_18_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_0);
        let lookup_range_check_18_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_1);
        let lookup_range_check_18_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_2);
        let lookup_range_check_18_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_3);
        let lookup_range_check_18_4_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_4);
        let lookup_range_check_18_5_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_5);
        let lookup_range_check_18_6_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_6);
        let lookup_range_check_18_7_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_7);
        let lookup_range_check_18_8_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_8);
        let lookup_range_check_18_9_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_9);
        let lookup_range_check_18_10_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_10);
        let lookup_range_check_18_11_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_11);
        let lookup_range_check_18_12_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_12);
        let lookup_range_check_18_13_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_13);
        let lookup_range_check_18_14_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_14);
        let lookup_range_check_18_15_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_15);
        let lookup_range_check_18_16_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_16);
        let lookup_range_check_18_17_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_17);
        let lookup_range_check_18_18_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_18);
        let lookup_range_check_18_19_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_19);
        let lookup_range_check_18_20_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_20);
        let lookup_range_check_18_21_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_21);
        let lookup_range_check_18_22_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_22);
        let lookup_range_check_18_23_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_23);
        let lookup_range_check_18_24_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_24);
        let lookup_range_check_18_25_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_25);
        let lookup_range_check_18_26_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_26);
        let lookup_range_check_18_27_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_27);
        let lookup_range_check_18_28_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_28);
        let lookup_range_check_18_29_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_29);
        let lookup_range_check_18_30_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_30);
        let lookup_range_check_18_31_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_31);
        let lookup_range_check_18_32_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_32);
        let lookup_range_check_18_33_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_33);
        let lookup_range_check_18_34_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_34);
        let lookup_range_check_18_35_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_35);
        let lookup_range_check_18_36_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_36);
        let lookup_range_check_18_37_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_37);
        let lookup_range_check_18_38_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_38);
        let lookup_range_check_18_39_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_39);
        let lookup_range_check_18_40_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_40);
        let lookup_range_check_18_41_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_41);
        let lookup_range_check_18_42_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_42);
        let lookup_range_check_18_43_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_43);
        let lookup_range_check_18_44_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_44);
        let lookup_range_check_18_45_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_45);
        let lookup_range_check_18_46_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_46);
        let lookup_range_check_18_47_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_47);
        let lookup_range_check_18_48_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_48);
        let lookup_range_check_18_49_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_49);
        let lookup_range_check_18_50_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_50);
        let lookup_range_check_18_51_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_51);
        let lookup_range_check_18_52_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_52);
        let lookup_range_check_18_53_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_53);
        let lookup_range_check_18_54_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_54);
        let lookup_range_check_18_55_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_55);
        let lookup_range_check_18_56_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_56);
        let lookup_range_check_18_57_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_57);
        let lookup_range_check_18_58_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_58);
        let lookup_range_check_18_59_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_59);
        let lookup_range_check_18_60_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_60);
        let lookup_range_check_18_61_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_61);

        // Collect lookup data pointers - 40 range_check_3_6_6_3
        let lookup_range_check_3_6_6_3_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_0);
        let lookup_range_check_3_6_6_3_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_1);
        let lookup_range_check_3_6_6_3_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_2);
        let lookup_range_check_3_6_6_3_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_3);
        let lookup_range_check_3_6_6_3_4_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_4);
        let lookup_range_check_3_6_6_3_5_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_5);
        let lookup_range_check_3_6_6_3_6_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_6);
        let lookup_range_check_3_6_6_3_7_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_7);
        let lookup_range_check_3_6_6_3_8_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_8);
        let lookup_range_check_3_6_6_3_9_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_9);
        let lookup_range_check_3_6_6_3_10_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_10);
        let lookup_range_check_3_6_6_3_11_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_11);
        let lookup_range_check_3_6_6_3_12_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_12);
        let lookup_range_check_3_6_6_3_13_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_13);
        let lookup_range_check_3_6_6_3_14_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_14);
        let lookup_range_check_3_6_6_3_15_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_15);
        let lookup_range_check_3_6_6_3_16_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_16);
        let lookup_range_check_3_6_6_3_17_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_17);
        let lookup_range_check_3_6_6_3_18_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_18);
        let lookup_range_check_3_6_6_3_19_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_19);
        let lookup_range_check_3_6_6_3_20_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_20);
        let lookup_range_check_3_6_6_3_21_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_21);
        let lookup_range_check_3_6_6_3_22_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_22);
        let lookup_range_check_3_6_6_3_23_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_23);
        let lookup_range_check_3_6_6_3_24_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_24);
        let lookup_range_check_3_6_6_3_25_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_25);
        let lookup_range_check_3_6_6_3_26_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_26);
        let lookup_range_check_3_6_6_3_27_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_27);
        let lookup_range_check_3_6_6_3_28_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_28);
        let lookup_range_check_3_6_6_3_29_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_29);
        let lookup_range_check_3_6_6_3_30_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_30);
        let lookup_range_check_3_6_6_3_31_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_31);
        let lookup_range_check_3_6_6_3_32_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_32);
        let lookup_range_check_3_6_6_3_33_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_33);
        let lookup_range_check_3_6_6_3_34_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_34);
        let lookup_range_check_3_6_6_3_35_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_35);
        let lookup_range_check_3_6_6_3_36_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_36);
        let lookup_range_check_3_6_6_3_37_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_37);
        let lookup_range_check_3_6_6_3_38_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_38);
        let lookup_range_check_3_6_6_3_39_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_39);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;
            let range_check_12_ptr = range_check_12 as *const _ as *mut std::os::raw::c_void;
            let range_check_18_ptr = range_check_18 as *const _ as *mut std::os::raw::c_void;
            let range_check_3_6_6_3_ptr = range_check_3_6_6_3 as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_mul_mod_builtin_interaction_traces(
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                range_check_12_ptr,
                range_check_18_ptr,
                range_check_3_6_6_3_ptr,

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

                // 32 range_check_12 lookups
                lookup_range_check_12_0_vec.as_ptr(),
                lookup_range_check_12_1_vec.as_ptr(),
                lookup_range_check_12_2_vec.as_ptr(),
                lookup_range_check_12_3_vec.as_ptr(),
                lookup_range_check_12_4_vec.as_ptr(),
                lookup_range_check_12_5_vec.as_ptr(),
                lookup_range_check_12_6_vec.as_ptr(),
                lookup_range_check_12_7_vec.as_ptr(),
                lookup_range_check_12_8_vec.as_ptr(),
                lookup_range_check_12_9_vec.as_ptr(),
                lookup_range_check_12_10_vec.as_ptr(),
                lookup_range_check_12_11_vec.as_ptr(),
                lookup_range_check_12_12_vec.as_ptr(),
                lookup_range_check_12_13_vec.as_ptr(),
                lookup_range_check_12_14_vec.as_ptr(),
                lookup_range_check_12_15_vec.as_ptr(),
                lookup_range_check_12_16_vec.as_ptr(),
                lookup_range_check_12_17_vec.as_ptr(),
                lookup_range_check_12_18_vec.as_ptr(),
                lookup_range_check_12_19_vec.as_ptr(),
                lookup_range_check_12_20_vec.as_ptr(),
                lookup_range_check_12_21_vec.as_ptr(),
                lookup_range_check_12_22_vec.as_ptr(),
                lookup_range_check_12_23_vec.as_ptr(),
                lookup_range_check_12_24_vec.as_ptr(),
                lookup_range_check_12_25_vec.as_ptr(),
                lookup_range_check_12_26_vec.as_ptr(),
                lookup_range_check_12_27_vec.as_ptr(),
                lookup_range_check_12_28_vec.as_ptr(),
                lookup_range_check_12_29_vec.as_ptr(),
                lookup_range_check_12_30_vec.as_ptr(),
                lookup_range_check_12_31_vec.as_ptr(),

                // 62 range_check_18 lookups
                lookup_range_check_18_0_vec.as_ptr(),
                lookup_range_check_18_1_vec.as_ptr(),
                lookup_range_check_18_2_vec.as_ptr(),
                lookup_range_check_18_3_vec.as_ptr(),
                lookup_range_check_18_4_vec.as_ptr(),
                lookup_range_check_18_5_vec.as_ptr(),
                lookup_range_check_18_6_vec.as_ptr(),
                lookup_range_check_18_7_vec.as_ptr(),
                lookup_range_check_18_8_vec.as_ptr(),
                lookup_range_check_18_9_vec.as_ptr(),
                lookup_range_check_18_10_vec.as_ptr(),
                lookup_range_check_18_11_vec.as_ptr(),
                lookup_range_check_18_12_vec.as_ptr(),
                lookup_range_check_18_13_vec.as_ptr(),
                lookup_range_check_18_14_vec.as_ptr(),
                lookup_range_check_18_15_vec.as_ptr(),
                lookup_range_check_18_16_vec.as_ptr(),
                lookup_range_check_18_17_vec.as_ptr(),
                lookup_range_check_18_18_vec.as_ptr(),
                lookup_range_check_18_19_vec.as_ptr(),
                lookup_range_check_18_20_vec.as_ptr(),
                lookup_range_check_18_21_vec.as_ptr(),
                lookup_range_check_18_22_vec.as_ptr(),
                lookup_range_check_18_23_vec.as_ptr(),
                lookup_range_check_18_24_vec.as_ptr(),
                lookup_range_check_18_25_vec.as_ptr(),
                lookup_range_check_18_26_vec.as_ptr(),
                lookup_range_check_18_27_vec.as_ptr(),
                lookup_range_check_18_28_vec.as_ptr(),
                lookup_range_check_18_29_vec.as_ptr(),
                lookup_range_check_18_30_vec.as_ptr(),
                lookup_range_check_18_31_vec.as_ptr(),
                lookup_range_check_18_32_vec.as_ptr(),
                lookup_range_check_18_33_vec.as_ptr(),
                lookup_range_check_18_34_vec.as_ptr(),
                lookup_range_check_18_35_vec.as_ptr(),
                lookup_range_check_18_36_vec.as_ptr(),
                lookup_range_check_18_37_vec.as_ptr(),
                lookup_range_check_18_38_vec.as_ptr(),
                lookup_range_check_18_39_vec.as_ptr(),
                lookup_range_check_18_40_vec.as_ptr(),
                lookup_range_check_18_41_vec.as_ptr(),
                lookup_range_check_18_42_vec.as_ptr(),
                lookup_range_check_18_43_vec.as_ptr(),
                lookup_range_check_18_44_vec.as_ptr(),
                lookup_range_check_18_45_vec.as_ptr(),
                lookup_range_check_18_46_vec.as_ptr(),
                lookup_range_check_18_47_vec.as_ptr(),
                lookup_range_check_18_48_vec.as_ptr(),
                lookup_range_check_18_49_vec.as_ptr(),
                lookup_range_check_18_50_vec.as_ptr(),
                lookup_range_check_18_51_vec.as_ptr(),
                lookup_range_check_18_52_vec.as_ptr(),
                lookup_range_check_18_53_vec.as_ptr(),
                lookup_range_check_18_54_vec.as_ptr(),
                lookup_range_check_18_55_vec.as_ptr(),
                lookup_range_check_18_56_vec.as_ptr(),
                lookup_range_check_18_57_vec.as_ptr(),
                lookup_range_check_18_58_vec.as_ptr(),
                lookup_range_check_18_59_vec.as_ptr(),
                lookup_range_check_18_60_vec.as_ptr(),
                lookup_range_check_18_61_vec.as_ptr(),

                // 40 range_check_3_6_6_3 lookups
                lookup_range_check_3_6_6_3_0_vec.as_ptr(),
                lookup_range_check_3_6_6_3_1_vec.as_ptr(),
                lookup_range_check_3_6_6_3_2_vec.as_ptr(),
                lookup_range_check_3_6_6_3_3_vec.as_ptr(),
                lookup_range_check_3_6_6_3_4_vec.as_ptr(),
                lookup_range_check_3_6_6_3_5_vec.as_ptr(),
                lookup_range_check_3_6_6_3_6_vec.as_ptr(),
                lookup_range_check_3_6_6_3_7_vec.as_ptr(),
                lookup_range_check_3_6_6_3_8_vec.as_ptr(),
                lookup_range_check_3_6_6_3_9_vec.as_ptr(),
                lookup_range_check_3_6_6_3_10_vec.as_ptr(),
                lookup_range_check_3_6_6_3_11_vec.as_ptr(),
                lookup_range_check_3_6_6_3_12_vec.as_ptr(),
                lookup_range_check_3_6_6_3_13_vec.as_ptr(),
                lookup_range_check_3_6_6_3_14_vec.as_ptr(),
                lookup_range_check_3_6_6_3_15_vec.as_ptr(),
                lookup_range_check_3_6_6_3_16_vec.as_ptr(),
                lookup_range_check_3_6_6_3_17_vec.as_ptr(),
                lookup_range_check_3_6_6_3_18_vec.as_ptr(),
                lookup_range_check_3_6_6_3_19_vec.as_ptr(),
                lookup_range_check_3_6_6_3_20_vec.as_ptr(),
                lookup_range_check_3_6_6_3_21_vec.as_ptr(),
                lookup_range_check_3_6_6_3_22_vec.as_ptr(),
                lookup_range_check_3_6_6_3_23_vec.as_ptr(),
                lookup_range_check_3_6_6_3_24_vec.as_ptr(),
                lookup_range_check_3_6_6_3_25_vec.as_ptr(),
                lookup_range_check_3_6_6_3_26_vec.as_ptr(),
                lookup_range_check_3_6_6_3_27_vec.as_ptr(),
                lookup_range_check_3_6_6_3_28_vec.as_ptr(),
                lookup_range_check_3_6_6_3_29_vec.as_ptr(),
                lookup_range_check_3_6_6_3_30_vec.as_ptr(),
                lookup_range_check_3_6_6_3_31_vec.as_ptr(),
                lookup_range_check_3_6_6_3_32_vec.as_ptr(),
                lookup_range_check_3_6_6_3_33_vec.as_ptr(),
                lookup_range_check_3_6_6_3_34_vec.as_ptr(),
                lookup_range_check_3_6_6_3_35_vec.as_ptr(),
                lookup_range_check_3_6_6_3_36_vec.as_ptr(),
                lookup_range_check_3_6_6_3_37_vec.as_ptr(),
                lookup_range_check_3_6_6_3_38_vec.as_ptr(),
                lookup_range_check_3_6_6_3_39_vec.as_ptr(),

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
        mul_mod_builtin,
        range_check_12, range_check_18, range_check_3_6_6_3,
    };
    use crate::witness::components_cuda::{
        memory_address_to_id_cuda, memory_id_to_big_cuda,
        mul_mod_builtin_cuda,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use crate::debug_tools::assert_constraints::assert_component;

    #[test]
    fn test_mul_mod_builtin_cpu_ref() {
        use cairo_air::relations;
        use cairo_air::components::mul_mod_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_mul_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for mul_mod builtin segment
        use stwo_cairo_adapter::builtins::MUL_MOD_MEMORY_CELLS;

        let mul_mod_segment = input.builtin_segments.mul_mod.as_ref()
            .expect("Expected mul_mod builtin segment");

        let segment_length = mul_mod_segment.stop_ptr - mul_mod_segment.begin_addr;
        let n_instances = segment_length / MUL_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = mul_mod_segment.begin_addr as u32;

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

        // Create mul_mod_builtin claim generator
        let mul_mod_claim_gen = mul_mod_builtin::ClaimGenerator::new(log_size, segment_start);

        // Initialize range check generators
        let range_check_12_state = range_check_12::ClaimGenerator::new();
        let range_check_18_state = range_check_18::ClaimGenerator::new();
        let range_check_3_6_6_3_state = range_check_3_6_6_3::ClaimGenerator::new();

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_12_relation = relations::RangeCheck_12::dummy();
        let range_check_18_relation = relations::RangeCheck_18::dummy();
        let range_check_3_6_6_3_relation = relations::RangeCheck_3_6_6_3::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(19);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (mul_mod_claim, mul_mod_interaction_gen) = mul_mod_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &range_check_12_state,
            &range_check_18_state,
            &range_check_3_6_6_3_state,
        );

        mock_tree_builder.finalize_interaction();

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_mod_interaction_claim = mul_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_12_relation,
            &range_check_3_6_6_3_relation,
            &range_check_18_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // Create component and verify with assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_mod_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_mod_builtin"),
                claim: mul_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_12_lookup_elements: relations::RangeCheck_12::dummy(),
                range_check_3_6_6_3_lookup_elements: relations::RangeCheck_3_6_6_3::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
            },
            mul_mod_interaction_claim.claimed_sum,
        );

        assert_component(&mul_mod_component, &trace)
    }

    #[test]
    fn test_mul_mod_builtin_trace_gen_by_cpu_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::mul_mod_builtin::{Component, Eval};
        use stwo::core::fields::m31::BaseField;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_mul_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for mul_mod builtin segment
        use stwo_cairo_adapter::builtins::MUL_MOD_MEMORY_CELLS;

        let mul_mod_segment = input.builtin_segments.mul_mod.as_ref()
            .expect("Expected mul_mod builtin segment");

        let segment_length = mul_mod_segment.stop_ptr - mul_mod_segment.begin_addr;
        let n_instances = segment_length / MUL_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = mul_mod_segment.begin_addr as u32;

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

        // Create mul_mod_builtin claim generator
        let mul_mod_claim_gen = mul_mod_builtin::ClaimGenerator::new(log_size, segment_start);

        // Initialize range check generators
        let range_check_12_state = range_check_12::ClaimGenerator::new();
        let range_check_18_state = range_check_18::ClaimGenerator::new();
        let range_check_3_6_6_3_state = range_check_3_6_6_3::ClaimGenerator::new();

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_12_relation = relations::RangeCheck_12::dummy();
        let range_check_18_relation = relations::RangeCheck_18::dummy();
        let range_check_3_6_6_3_relation = relations::RangeCheck_3_6_6_3::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(19);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (mul_mod_claim, mul_mod_interaction_gen) = mul_mod_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &range_check_12_state,
            &range_check_18_state,
            &range_check_3_6_6_3_state,
        );

        mock_tree_builder.finalize_interaction();

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_mod_interaction_claim = mul_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_12_relation,
            &range_check_3_6_6_3_relation,
            &range_check_18_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

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

        // Create component first to get actual constraint count
        let tree_span_provider_temp = &mut TraceLocationAllocator::default();
        let mul_mod_component_temp = Component::new(
            tree_span_provider_temp,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_mod_builtin"),
                claim: mul_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_12_lookup_elements: relations::RangeCheck_12::dummy(),
                range_check_3_6_6_3_lookup_elements: relations::RangeCheck_3_6_6_3::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
            },
            mul_mod_interaction_claim.claimed_sum,
        );

        // Create mock CUDA buffers with correct size
        let domain_size = 1 << mul_mod_claim.log_size;
        let n_constraints = mul_mod_component_temp.info.n_constraints.max(500); // Use actual constraint count

        let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        // Use the component we already created
        let mul_mod_component = mul_mod_component_temp;

        // Call CUDA evaluator
        let eval_ptr = &mul_mod_component.eval as *const _ as *mut std::os::raw::c_void;
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
                mul_mod_claim.log_size as u32,
                mul_mod_claim.log_size as u32,
                mul_mod_component.info.n_constraints as u32,
                mul_mod_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    mul_mod_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << mul_mod_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }
    }

    #[test]
    fn test_mul_mod_builtin_trace_gen_by_cuda_and_verify_by_cpu() {
        use cairo_air::relations;
        use cairo_air::components::mul_mod_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_mul_mod_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for mul_mod builtin segment
        use stwo_cairo_adapter::builtins::MUL_MOD_MEMORY_CELLS;

        let mul_mod_segment = input.builtin_segments.mul_mod.as_ref()
            .expect("Expected mul_mod builtin segment");

        let segment_length = mul_mod_segment.stop_ptr - mul_mod_segment.begin_addr;
        let n_instances = segment_length / MUL_MOD_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = mul_mod_segment.begin_addr as u32;

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
        let mul_mod_cuda_gen = mul_mod_builtin_cuda::CudaClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_12_relation = relations::RangeCheck_12::dummy();
        let range_check_18_relation = relations::RangeCheck_18::dummy();
        let range_check_3_6_6_3_relation = relations::RangeCheck_3_6_6_3::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (mul_mod_claim, mul_mod_interaction_gen) = mul_mod_cuda_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        // Interaction trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let mul_mod_interaction_claim = mul_mod_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_12_relation,
            &range_check_3_6_6_3_relation,
            &range_check_18_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // Verify with CPU
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let mul_mod_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("mul_mod_builtin"),
                claim: mul_mod_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_12_lookup_elements: relations::RangeCheck_12::dummy(),
                range_check_3_6_6_3_lookup_elements: relations::RangeCheck_3_6_6_3::dummy(),
                range_check_18_lookup_elements: relations::RangeCheck_18::dummy(),
            },
            mul_mod_interaction_claim.claimed_sum,
        );

        // Generate CPU trace for comparison
        let memory_address_to_id_cpu_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_cpu_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_cpu_state.get_id(addr);
            memory_address_to_id_cpu_state.add_input(&addr);
            memory_id_to_big_cpu_state.add_input(&id);
        }
        let mul_mod_cpu_gen = mul_mod_builtin::ClaimGenerator::new(log_size, segment_start);
        let range_check_12_state = range_check_12::ClaimGenerator::new();
        let range_check_18_state = range_check_18::ClaimGenerator::new();
        let range_check_3_6_6_3_state = range_check_3_6_6_3::ClaimGenerator::new();

        let mut cpu_mock_commitment_scheme = MockCommitmentScheme::default();
        // Preprocessed trace
        let mut cpu_tree_builder = cpu_mock_commitment_scheme.tree_builder();
        cpu_tree_builder.extend_evals(testing_preprocessed_tree(log_size).gen_trace());
        cpu_tree_builder.finalize_interaction();
        // CPU base trace
        let mut cpu_tree_builder = cpu_mock_commitment_scheme.tree_builder();
        let (cpu_mul_mod_claim, cpu_interaction_gen) = mul_mod_cpu_gen.write_trace(
            &mut cpu_tree_builder,
            &memory_address_to_id_cpu_state,
            &memory_id_to_big_cpu_state,
            &range_check_12_state,
            &range_check_18_state,
            &range_check_3_6_6_3_state,
        );
        cpu_tree_builder.finalize_interaction();
        // CPU interaction trace
        let mut cpu_tree_builder = cpu_mock_commitment_scheme.tree_builder();
        let cpu_interaction_claim = cpu_interaction_gen.write_interaction_trace(
            &mut cpu_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_12_relation,
            &range_check_3_6_6_3_relation,
            &range_check_18_relation,
        );
        cpu_tree_builder.finalize_interaction();
        let cpu_trace = cpu_mock_commitment_scheme.trace_domain_evaluations();

        // Verify claims match
        assert_eq!(mul_mod_claim.log_size, cpu_mul_mod_claim.log_size,
            "CUDA and CPU log_size should match");

        // Verify claimed_sum matches
        assert_eq!(mul_mod_interaction_claim.claimed_sum, cpu_interaction_claim.claimed_sum,
            "CUDA and CPU claimed_sum should match");

        // Compare base trace columns (tree index 1)
        let cuda_base_trace = &trace[1];
        let cpu_base_trace = &cpu_trace[1];
        assert_eq!(cuda_base_trace.len(), cpu_base_trace.len(),
            "CUDA and CPU base trace column count should match");

        // Verify all base trace columns match across all rows
        let n_rows = 1 << log_size;
        let mut total_base_mismatches = 0;
        for col in 0..cuda_base_trace.len() {
            for row in 0..n_rows {
                let cuda_val = cuda_base_trace[col][row];
                let cpu_val = cpu_base_trace[col][row];
                if cuda_val != cpu_val {
                    total_base_mismatches += 1;
                    if total_base_mismatches <= 5 {
                        println!("Base trace mismatch at col {}, row {}: CUDA={:?}, CPU={:?}",
                            col, row, cuda_val, cpu_val);
                    }
                }
            }
        }
        assert_eq!(total_base_mismatches, 0,
            "CUDA and CPU base traces should match exactly, found {} mismatches", total_base_mismatches);

        // Compare interaction trace columns (tree index 2)
        let cuda_int_trace = &trace[2];
        let cpu_int_trace = &cpu_trace[2];
        assert_eq!(cuda_int_trace.len(), cpu_int_trace.len(),
            "CUDA and CPU interaction trace column count should match");

        // Verify all interaction trace columns match across all rows
        let mut total_int_mismatches = 0;
        for col in 0..cuda_int_trace.len() {
            for row in 0..n_rows {
                let cuda_val = cuda_int_trace[col][row];
                let cpu_val = cpu_int_trace[col][row];
                if cuda_val != cpu_val {
                    total_int_mismatches += 1;
                    if total_int_mismatches <= 5 {
                        println!("Interaction trace mismatch at col {}, row {}: CUDA={:?}, CPU={:?}",
                            col, row, cuda_val, cpu_val);
                    }
                }
            }
        }
        assert_eq!(total_int_mismatches, 0,
            "CUDA and CPU interaction traces should match exactly, found {} mismatches", total_int_mismatches);

        // Verify constraint satisfaction
        assert_component(&mul_mod_component, &trace);
    }
}
