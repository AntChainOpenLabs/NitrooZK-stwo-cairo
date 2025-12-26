// CUDA witness generation for bitwise_builtin component
// This handles AND/OR/XOR bitwise operations on 252-bit integers

#![allow(unused_parens)]
use cairo_air::components::bitwise_builtin::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big};
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 89;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 19;

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
            Claim { log_size, bitwise_builtin_segment_start: self.segment_start },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

pub struct CudaSubComponentInputs {
    // 5 memory_address_to_id lookups
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 5],
    // 5 memory_id_to_big lookups
    pub memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 5],
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
                verify_bitwise_xor_9: std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << log_size))),
                verify_bitwise_xor_8: std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << log_size))),
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

    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_verify_bitwise_xor_9_vec: Vec<_> = sub_component_inputs.verify_bitwise_xor_9
        .iter()
        .flat_map(|row| row.iter().map(|x| x.device_ptr))
        .collect();
    let sub_component_inputs_verify_bitwise_xor_8_vec: Vec<_> = sub_component_inputs.verify_bitwise_xor_8
        .iter()
        .flat_map(|row| row.iter().map(|x| x.device_ptr))
        .collect();

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
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
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        verify_bitwise_xor_9: &relations::VerifyBitwiseXor_9,
        verify_bitwise_xor_8: &relations::VerifyBitwiseXor_8,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // 19 pairs × 4 m31 columns = 76 columns
        // Each pair stores: 4 m31 for accumulated QM31 value
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect lookup data pointers - 5 memory_address_to_id
        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_address_to_id_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_3);
        let lookup_memory_address_to_id_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_4);

        // Collect lookup data pointers - 5 memory_id_to_big
        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_memory_id_to_big_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_3);
        let lookup_memory_id_to_big_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_4);

        // Collect lookup data pointers - 27 verify_bitwise_xor_9
        let lookup_verify_bitwise_xor_9_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_0);
        let lookup_verify_bitwise_xor_9_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_1);
        let lookup_verify_bitwise_xor_9_2_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_2);
        let lookup_verify_bitwise_xor_9_3_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_3);
        let lookup_verify_bitwise_xor_9_4_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_4);
        let lookup_verify_bitwise_xor_9_5_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_5);
        let lookup_verify_bitwise_xor_9_6_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_6);
        let lookup_verify_bitwise_xor_9_7_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_7);
        let lookup_verify_bitwise_xor_9_8_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_8);
        let lookup_verify_bitwise_xor_9_9_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_9);
        let lookup_verify_bitwise_xor_9_10_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_10);
        let lookup_verify_bitwise_xor_9_11_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_11);
        let lookup_verify_bitwise_xor_9_12_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_12);
        let lookup_verify_bitwise_xor_9_13_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_13);
        let lookup_verify_bitwise_xor_9_14_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_14);
        let lookup_verify_bitwise_xor_9_15_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_15);
        let lookup_verify_bitwise_xor_9_16_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_16);
        let lookup_verify_bitwise_xor_9_17_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_17);
        let lookup_verify_bitwise_xor_9_18_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_18);
        let lookup_verify_bitwise_xor_9_19_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_19);
        let lookup_verify_bitwise_xor_9_20_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_20);
        let lookup_verify_bitwise_xor_9_21_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_21);
        let lookup_verify_bitwise_xor_9_22_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_22);
        let lookup_verify_bitwise_xor_9_23_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_23);
        let lookup_verify_bitwise_xor_9_24_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_24);
        let lookup_verify_bitwise_xor_9_25_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_25);
        let lookup_verify_bitwise_xor_9_26_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_26);

        // Collect lookup data pointers - 1 verify_bitwise_xor_8
        let lookup_verify_bitwise_xor_8_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_0);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_9_ptr = verify_bitwise_xor_9 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_8_ptr = verify_bitwise_xor_8 as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_bitwise_builtin_interaction_traces(
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                verify_bitwise_xor_9_ptr,
                verify_bitwise_xor_8_ptr,

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
        verify_bitwise_xor_8, verify_bitwise_xor_9,
        bitwise_builtin,
    };
    use crate::witness::components_cuda::{
        memory_address_to_id_cuda, memory_id_to_big_cuda,
        bitwise_builtin_cuda,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use crate::debug_tools::assert_constraints::assert_component;

    #[test]
    fn test_bitwise_builtin_cpu_ref() {
        use cairo_air::relations;
        use cairo_air::components::bitwise_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_bitwise_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for bitwise builtin segment
        use stwo_cairo_adapter::builtins::BITWISE_MEMORY_CELLS;

        let bitwise_segment = input.builtin_segments.bitwise.as_ref()
            .expect("Expected bitwise builtin segment");

        let segment_length = bitwise_segment.stop_ptr - bitwise_segment.begin_addr;
        let n_instances = segment_length / BITWISE_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = bitwise_segment.begin_addr as u32;

        println!("bitwise_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_bitwise_xor_8_state = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_9_state = verify_bitwise_xor_9::ClaimGenerator::new();

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

        // Create bitwise_builtin claim generator
        let bitwise_claim_gen = bitwise_builtin::ClaimGenerator::new(log_size, segment_start);

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (bitwise_claim, bitwise_interaction_gen) = bitwise_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &verify_bitwise_xor_8_state,
            &verify_bitwise_xor_9_state,
        );

        mock_tree_builder.finalize_interaction();

        println!("bitwise_builtin_claim log_size: {:?}", bitwise_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let bitwise_interaction_claim = bitwise_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &verify_bitwise_xor_9_relation,
            &verify_bitwise_xor_8_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("bitwise_builtin_interaction_claim.claimed_sum: {:?}", bitwise_interaction_claim.claimed_sum);

        // Create component and verify with assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let bitwise_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("bitwise_builtin"),
                claim: bitwise_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
            },
            bitwise_interaction_claim.claimed_sum,
        );

        assert_component(&bitwise_component, &trace)
    }

    #[test]
    fn test_bitwise_builtin_trace_gen_by_cpu_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::bitwise_builtin::{Component, Eval};
        use stwo::core::fields::m31::{M31, BaseField};
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::prover::backend::Column;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_bitwise_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for bitwise builtin segment
        use stwo_cairo_adapter::builtins::BITWISE_MEMORY_CELLS;

        let bitwise_segment = input.builtin_segments.bitwise.as_ref()
            .expect("Expected bitwise builtin segment");

        let segment_length = bitwise_segment.stop_ptr - bitwise_segment.begin_addr;
        let n_instances = segment_length / BITWISE_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = bitwise_segment.begin_addr as u32;

        println!("bitwise_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_bitwise_xor_8_state = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_9_state = verify_bitwise_xor_9::ClaimGenerator::new();

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

        // Create bitwise_builtin claim generator
        let bitwise_claim_gen = bitwise_builtin::ClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (bitwise_claim, bitwise_interaction_gen) = bitwise_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &verify_bitwise_xor_8_state,
            &verify_bitwise_xor_9_state,
        );

        mock_tree_builder.finalize_interaction();

        println!("bitwise_builtin_claim log_size: {:?}", bitwise_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let bitwise_interaction_claim = bitwise_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &verify_bitwise_xor_9_relation,
            &verify_bitwise_xor_8_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("bitwise_builtin_interaction_claim.claimed_sum: {:?}", bitwise_interaction_claim.claimed_sum);

        // Convert trace to CUDA format
        // trace[0] is preprocessed trace (Seq column)
        // trace[1] is base trace
        // trace[2] is interaction trace
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
        let bitwise_component_temp = Component::new(
            tree_span_provider_temp,
            Eval {
                eval_id: fnv1a_eval_id_gen("bitwise_builtin"),
                claim: bitwise_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
            },
            bitwise_interaction_claim.claimed_sum,
        );

        println!("bitwise_builtin n_constraints: {}", bitwise_component_temp.info.n_constraints);

        // Create mock CUDA buffers with correct size
        let domain_size = 1 << bitwise_claim.log_size;
        let n_constraints = bitwise_component_temp.info.n_constraints.max(500);

        let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        // Use the component we already created
        let bitwise_component = bitwise_component_temp;

        // Call CUDA evaluator
        let eval_ptr = &bitwise_component.eval as *const _ as *mut std::os::raw::c_void;
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
                bitwise_claim.log_size as u32,
                bitwise_claim.log_size as u32,
                bitwise_component.info.n_constraints as u32,
                bitwise_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    bitwise_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << bitwise_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }

        println!("bitwise_builtin CUDA evaluator test completed successfully!");
    }

    /// CUDA trace generation + CPU constraint verification.
    #[test]
    fn test_bitwise_builtin_trace_gen_by_cuda_and_verify_by_cpu() {
        use cairo_air::relations;
        use cairo_air::components::bitwise_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_bitwise_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for bitwise builtin segment
        use stwo_cairo_adapter::builtins::BITWISE_MEMORY_CELLS;

        let bitwise_segment = input.builtin_segments.bitwise.as_ref()
            .expect("Expected bitwise builtin segment");

        let segment_length = bitwise_segment.stop_ptr - bitwise_segment.begin_addr;
        let n_instances = segment_length / BITWISE_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = bitwise_segment.begin_addr as u32;

        println!("bitwise_builtin log_size: {}, segment_start: {}", log_size, segment_start);

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
        let bitwise_cuda_gen = bitwise_builtin_cuda::CudaClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (bitwise_claim, bitwise_interaction_gen) = bitwise_cuda_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        println!("bitwise_builtin_claim log_size: {:?}", bitwise_claim.log_size);

        // Interaction trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let bitwise_interaction_claim = bitwise_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &verify_bitwise_xor_9_relation,
            &verify_bitwise_xor_8_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("bitwise_builtin_interaction_claim.claimed_sum: {:?}", bitwise_interaction_claim.claimed_sum);

        // Verify with CPU
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let bitwise_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("bitwise_builtin"),
                claim: bitwise_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
            },
            bitwise_interaction_claim.claimed_sum,
        );

        assert_component(&bitwise_component, &trace);
        println!("bitwise_builtin CUDA trace gen + CPU verify test completed successfully!");
    }

    /// Compare CPU and CUDA traces column by column to identify discrepancies.
    #[test]
    fn test_bitwise_builtin_compare_cpu_vs_cuda_traces() {
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_bitwise_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for bitwise builtin segment
        use stwo_cairo_adapter::builtins::BITWISE_MEMORY_CELLS;

        let bitwise_segment = input.builtin_segments.bitwise.as_ref()
            .expect("Expected bitwise builtin segment");

        let segment_length = bitwise_segment.stop_ptr - bitwise_segment.begin_addr;
        let n_instances = segment_length / BITWISE_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = bitwise_segment.begin_addr as u32;

        println!("bitwise_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // ============ CPU trace generation ============
        let memory_address_to_id_state_cpu = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state_cpu = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let verify_bitwise_xor_8_state_cpu = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_9_state_cpu = verify_bitwise_xor_9::ClaimGenerator::new();

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_state_cpu.get_id(addr);
            memory_address_to_id_state_cpu.add_input(&addr);
            memory_id_to_big_state_cpu.add_input(&id);
        }

        let bitwise_claim_gen_cpu = bitwise_builtin::ClaimGenerator::new(log_size, segment_start);

        let mut mock_commitment_scheme_cpu = MockCommitmentScheme::default();
        let preprocessed_trace_cpu = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        mock_tree_builder_cpu.extend_evals(preprocessed_trace_cpu.gen_trace());
        mock_tree_builder_cpu.finalize_interaction();

        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        let (cpu_claim, _cpu_interaction_gen) = bitwise_claim_gen_cpu.write_trace(
            &mut mock_tree_builder_cpu,
            &memory_address_to_id_state_cpu,
            &memory_id_to_big_state_cpu,
            &verify_bitwise_xor_8_state_cpu,
            &verify_bitwise_xor_9_state_cpu,
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

        let bitwise_cuda_gen = bitwise_builtin_cuda::CudaClaimGenerator::new(log_size, segment_start);

        let mut mock_commitment_scheme_cuda = MockCommitmentScheme::default();
        let preprocessed_trace_cuda = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        mock_tree_builder_cuda.extend_evals(preprocessed_trace_cuda.gen_trace());
        mock_tree_builder_cuda.finalize_interaction();

        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        let (cuda_claim, _cuda_interaction_gen) = bitwise_cuda_gen.write_trace(
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
