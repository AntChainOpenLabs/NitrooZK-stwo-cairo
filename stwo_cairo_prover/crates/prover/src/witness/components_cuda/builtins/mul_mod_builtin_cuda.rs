#![allow(unused_parens)]
use cairo_air::components::mul_mod_builtin::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use super::super::range_check::{rc_12, rc_18, rc_3_6_6_3};
use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::prelude::*;
pub const N_TRACE_COLUMNS: usize = 426;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 94;

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
        range_check_12_cuda_state: &rc_12::CudaClaimGenerator,
        range_check_18_cuda_state: &rc_18::CudaClaimGenerator,
        range_check_3_6_6_3_cuda_state: &rc_3_6_6_3::CudaClaimGenerator,
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

        // Add to CUDA generators - SIMD memory multiplicities are handled by the merge in
        // cairo_cuda.rs
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add range check multiplicities directly to CUDA generators
        macro_rules! add_rc_cuda {
            ($state:expr, [$($field:expr),+ $(,)?]) => {
                $(
                    $state.add_cuda_inputs(std::slice::from_ref($field));
                )+
            };
        }

        macro_rules! add_rc_cuda_for_relation {
            ($state:expr, $rel:expr, [$($field:expr),+ $(,)?]) => {
                $(
                    $state.add_cuda_inputs_for_relation(std::slice::from_ref($field), $rel);
                )+
            };
        }

        // 32 range_check_12 lookups (1 element each)
        add_rc_cuda!(
            range_check_12_cuda_state,
            [
                &lookup_data.range_check_12_0,
                &lookup_data.range_check_12_1,
                &lookup_data.range_check_12_2,
                &lookup_data.range_check_12_3,
                &lookup_data.range_check_12_4,
                &lookup_data.range_check_12_5,
                &lookup_data.range_check_12_6,
                &lookup_data.range_check_12_7,
                &lookup_data.range_check_12_8,
                &lookup_data.range_check_12_9,
                &lookup_data.range_check_12_10,
                &lookup_data.range_check_12_11,
                &lookup_data.range_check_12_12,
                &lookup_data.range_check_12_13,
                &lookup_data.range_check_12_14,
                &lookup_data.range_check_12_15,
                &lookup_data.range_check_12_16,
                &lookup_data.range_check_12_17,
                &lookup_data.range_check_12_18,
                &lookup_data.range_check_12_19,
                &lookup_data.range_check_12_20,
                &lookup_data.range_check_12_21,
                &lookup_data.range_check_12_22,
                &lookup_data.range_check_12_23,
                &lookup_data.range_check_12_24,
                &lookup_data.range_check_12_25,
                &lookup_data.range_check_12_26,
                &lookup_data.range_check_12_27,
                &lookup_data.range_check_12_28,
                &lookup_data.range_check_12_29,
                &lookup_data.range_check_12_30,
                &lookup_data.range_check_12_31
            ]
        );

        // 62 range_check_18 lookups (1 element each, all relation 0)
        add_rc_cuda_for_relation!(
            range_check_18_cuda_state,
            0,
            [
                &lookup_data.range_check_18_0,
                &lookup_data.range_check_18_1,
                &lookup_data.range_check_18_2,
                &lookup_data.range_check_18_3,
                &lookup_data.range_check_18_4,
                &lookup_data.range_check_18_5,
                &lookup_data.range_check_18_6,
                &lookup_data.range_check_18_7,
                &lookup_data.range_check_18_8,
                &lookup_data.range_check_18_9,
                &lookup_data.range_check_18_10,
                &lookup_data.range_check_18_11,
                &lookup_data.range_check_18_12,
                &lookup_data.range_check_18_13,
                &lookup_data.range_check_18_14,
                &lookup_data.range_check_18_15,
                &lookup_data.range_check_18_16,
                &lookup_data.range_check_18_17,
                &lookup_data.range_check_18_18,
                &lookup_data.range_check_18_19,
                &lookup_data.range_check_18_20,
                &lookup_data.range_check_18_21,
                &lookup_data.range_check_18_22,
                &lookup_data.range_check_18_23,
                &lookup_data.range_check_18_24,
                &lookup_data.range_check_18_25,
                &lookup_data.range_check_18_26,
                &lookup_data.range_check_18_27,
                &lookup_data.range_check_18_28,
                &lookup_data.range_check_18_29,
                &lookup_data.range_check_18_30,
                &lookup_data.range_check_18_31,
                &lookup_data.range_check_18_32,
                &lookup_data.range_check_18_33,
                &lookup_data.range_check_18_34,
                &lookup_data.range_check_18_35,
                &lookup_data.range_check_18_36,
                &lookup_data.range_check_18_37,
                &lookup_data.range_check_18_38,
                &lookup_data.range_check_18_39,
                &lookup_data.range_check_18_40,
                &lookup_data.range_check_18_41,
                &lookup_data.range_check_18_42,
                &lookup_data.range_check_18_43,
                &lookup_data.range_check_18_44,
                &lookup_data.range_check_18_45,
                &lookup_data.range_check_18_46,
                &lookup_data.range_check_18_47,
                &lookup_data.range_check_18_48,
                &lookup_data.range_check_18_49,
                &lookup_data.range_check_18_50,
                &lookup_data.range_check_18_51,
                &lookup_data.range_check_18_52,
                &lookup_data.range_check_18_53,
                &lookup_data.range_check_18_54,
                &lookup_data.range_check_18_55,
                &lookup_data.range_check_18_56,
                &lookup_data.range_check_18_57,
                &lookup_data.range_check_18_58,
                &lookup_data.range_check_18_59,
                &lookup_data.range_check_18_60,
                &lookup_data.range_check_18_61
            ]
        );

        // 40 range_check_3_6_6_3 lookups (4 elements each)
        add_rc_cuda!(
            range_check_3_6_6_3_cuda_state,
            [
                &lookup_data.range_check_3_6_6_3_0,
                &lookup_data.range_check_3_6_6_3_1,
                &lookup_data.range_check_3_6_6_3_2,
                &lookup_data.range_check_3_6_6_3_3,
                &lookup_data.range_check_3_6_6_3_4,
                &lookup_data.range_check_3_6_6_3_5,
                &lookup_data.range_check_3_6_6_3_6,
                &lookup_data.range_check_3_6_6_3_7,
                &lookup_data.range_check_3_6_6_3_8,
                &lookup_data.range_check_3_6_6_3_9,
                &lookup_data.range_check_3_6_6_3_10,
                &lookup_data.range_check_3_6_6_3_11,
                &lookup_data.range_check_3_6_6_3_12,
                &lookup_data.range_check_3_6_6_3_13,
                &lookup_data.range_check_3_6_6_3_14,
                &lookup_data.range_check_3_6_6_3_15,
                &lookup_data.range_check_3_6_6_3_16,
                &lookup_data.range_check_3_6_6_3_17,
                &lookup_data.range_check_3_6_6_3_18,
                &lookup_data.range_check_3_6_6_3_19,
                &lookup_data.range_check_3_6_6_3_20,
                &lookup_data.range_check_3_6_6_3_21,
                &lookup_data.range_check_3_6_6_3_22,
                &lookup_data.range_check_3_6_6_3_23,
                &lookup_data.range_check_3_6_6_3_24,
                &lookup_data.range_check_3_6_6_3_25,
                &lookup_data.range_check_3_6_6_3_26,
                &lookup_data.range_check_3_6_6_3_27,
                &lookup_data.range_check_3_6_6_3_28,
                &lookup_data.range_check_3_6_6_3_29,
                &lookup_data.range_check_3_6_6_3_30,
                &lookup_data.range_check_3_6_6_3_31,
                &lookup_data.range_check_3_6_6_3_32,
                &lookup_data.range_check_3_6_6_3_33,
                &lookup_data.range_check_3_6_6_3_34,
                &lookup_data.range_check_3_6_6_3_35,
                &lookup_data.range_check_3_6_6_3_36,
                &lookup_data.range_check_3_6_6_3_37,
                &lookup_data.range_check_3_6_6_3_38,
                &lookup_data.range_check_3_6_6_3_39
            ]
        );

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim {
                log_size,
                mul_mod_builtin_segment_start: self.segment_start,
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

    let sub_component_inputs_memory_address_to_id_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    // Range check sub-component inputs are not needed for mul_mod_builtin base trace generation
    // (range checks don't have sub-component inputs to pass to the kernel)
    let sub_component_inputs_range_check_12_vec: Vec<*const u32> = vec![];
    let sub_component_inputs_range_check_18_vec: Vec<*const u32> = vec![];
    let sub_component_inputs_range_check_3_6_6_3_vec: Vec<*const u32> = vec![];

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
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
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(1 << trace_log_size) })
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
        let lookup_range_check_12_10_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_10);
        let lookup_range_check_12_11_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_11);
        let lookup_range_check_12_12_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_12);
        let lookup_range_check_12_13_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_13);
        let lookup_range_check_12_14_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_14);
        let lookup_range_check_12_15_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_15);
        let lookup_range_check_12_16_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_16);
        let lookup_range_check_12_17_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_17);
        let lookup_range_check_12_18_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_18);
        let lookup_range_check_12_19_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_19);
        let lookup_range_check_12_20_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_20);
        let lookup_range_check_12_21_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_21);
        let lookup_range_check_12_22_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_22);
        let lookup_range_check_12_23_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_23);
        let lookup_range_check_12_24_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_24);
        let lookup_range_check_12_25_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_25);
        let lookup_range_check_12_26_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_26);
        let lookup_range_check_12_27_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_27);
        let lookup_range_check_12_28_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_28);
        let lookup_range_check_12_29_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_29);
        let lookup_range_check_12_30_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_30);
        let lookup_range_check_12_31_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_12_31);

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
        let lookup_range_check_18_10_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_10);
        let lookup_range_check_18_11_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_11);
        let lookup_range_check_18_12_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_12);
        let lookup_range_check_18_13_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_13);
        let lookup_range_check_18_14_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_14);
        let lookup_range_check_18_15_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_15);
        let lookup_range_check_18_16_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_16);
        let lookup_range_check_18_17_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_17);
        let lookup_range_check_18_18_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_18);
        let lookup_range_check_18_19_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_19);
        let lookup_range_check_18_20_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_20);
        let lookup_range_check_18_21_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_21);
        let lookup_range_check_18_22_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_22);
        let lookup_range_check_18_23_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_23);
        let lookup_range_check_18_24_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_24);
        let lookup_range_check_18_25_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_25);
        let lookup_range_check_18_26_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_26);
        let lookup_range_check_18_27_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_27);
        let lookup_range_check_18_28_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_28);
        let lookup_range_check_18_29_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_29);
        let lookup_range_check_18_30_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_30);
        let lookup_range_check_18_31_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_31);
        let lookup_range_check_18_32_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_32);
        let lookup_range_check_18_33_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_33);
        let lookup_range_check_18_34_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_34);
        let lookup_range_check_18_35_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_35);
        let lookup_range_check_18_36_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_36);
        let lookup_range_check_18_37_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_37);
        let lookup_range_check_18_38_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_38);
        let lookup_range_check_18_39_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_39);
        let lookup_range_check_18_40_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_40);
        let lookup_range_check_18_41_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_41);
        let lookup_range_check_18_42_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_42);
        let lookup_range_check_18_43_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_43);
        let lookup_range_check_18_44_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_44);
        let lookup_range_check_18_45_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_45);
        let lookup_range_check_18_46_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_46);
        let lookup_range_check_18_47_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_47);
        let lookup_range_check_18_48_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_48);
        let lookup_range_check_18_49_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_49);
        let lookup_range_check_18_50_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_50);
        let lookup_range_check_18_51_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_51);
        let lookup_range_check_18_52_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_52);
        let lookup_range_check_18_53_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_53);
        let lookup_range_check_18_54_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_54);
        let lookup_range_check_18_55_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_55);
        let lookup_range_check_18_56_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_56);
        let lookup_range_check_18_57_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_57);
        let lookup_range_check_18_58_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_58);
        let lookup_range_check_18_59_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_59);
        let lookup_range_check_18_60_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_60);
        let lookup_range_check_18_61_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_18_61);

        // Collect lookup data pointers - 40 range_check_3_6_6_3
        let lookup_range_check_3_6_6_3_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_0);
        let lookup_range_check_3_6_6_3_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_1);
        let lookup_range_check_3_6_6_3_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_2);
        let lookup_range_check_3_6_6_3_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_3);
        let lookup_range_check_3_6_6_3_4_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_4);
        let lookup_range_check_3_6_6_3_5_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_5);
        let lookup_range_check_3_6_6_3_6_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_6);
        let lookup_range_check_3_6_6_3_7_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_7);
        let lookup_range_check_3_6_6_3_8_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_8);
        let lookup_range_check_3_6_6_3_9_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_9);
        let lookup_range_check_3_6_6_3_10_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_10);
        let lookup_range_check_3_6_6_3_11_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_11);
        let lookup_range_check_3_6_6_3_12_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_12);
        let lookup_range_check_3_6_6_3_13_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_13);
        let lookup_range_check_3_6_6_3_14_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_14);
        let lookup_range_check_3_6_6_3_15_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_15);
        let lookup_range_check_3_6_6_3_16_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_16);
        let lookup_range_check_3_6_6_3_17_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_17);
        let lookup_range_check_3_6_6_3_18_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_18);
        let lookup_range_check_3_6_6_3_19_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_19);
        let lookup_range_check_3_6_6_3_20_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_20);
        let lookup_range_check_3_6_6_3_21_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_21);
        let lookup_range_check_3_6_6_3_22_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_22);
        let lookup_range_check_3_6_6_3_23_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_23);
        let lookup_range_check_3_6_6_3_24_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_24);
        let lookup_range_check_3_6_6_3_25_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_25);
        let lookup_range_check_3_6_6_3_26_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_26);
        let lookup_range_check_3_6_6_3_27_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_27);
        let lookup_range_check_3_6_6_3_28_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_28);
        let lookup_range_check_3_6_6_3_29_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_29);
        let lookup_range_check_3_6_6_3_30_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_30);
        let lookup_range_check_3_6_6_3_31_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_31);
        let lookup_range_check_3_6_6_3_32_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_32);
        let lookup_range_check_3_6_6_3_33_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_33);
        let lookup_range_check_3_6_6_3_34_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_34);
        let lookup_range_check_3_6_6_3_35_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_35);
        let lookup_range_check_3_6_6_3_36_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_36);
        let lookup_range_check_3_6_6_3_37_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_37);
        let lookup_range_check_3_6_6_3_38_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_38);
        let lookup_range_check_3_6_6_3_39_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_6_6_3_39);

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
            let mod_rc12 =
                create_modified_lookup_for_cuda(lookup_elements, RANGE_CHECK_12_RELATION_ID);
            let mod_rc18 = create_modified_lookup_for_cuda(lookup_elements, RC_18_RELATION_ID);
            let mod_rc3663 =
                create_modified_lookup_for_cuda(lookup_elements, RANGE_CHECK_3_6_6_3_RELATION_ID);

            bindings_airs::generate_mul_mod_builtin_interaction_traces(
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_rc12 as *const _ as *mut std::os::raw::c_void,
                &mod_rc18 as *const _ as *mut std::os::raw::c_void,
                &mod_rc3663 as *const _ as *mut std::os::raw::c_void,
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
