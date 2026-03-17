// CUDA witness generation for poseidon_builtin component
// This handles Poseidon hash builtin operations
//
// This module uses CUDA for both base trace and interaction trace generation.
// The CUDA kernel generates lookup_data which is then used by the CUDA interaction kernel.

#![allow(unused_parens)]
use cairo_air::components::poseidon_builtin::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::prelude::*;

pub const N_TRACE_COLUMNS: usize = 341;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 68; // 17 logup columns × 4 M31

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

    /// Write trace using CUDA monolithic kernel.
    ///
    /// This function:
    /// 1. Generates base trace via CUDA kernel (gen_poseidon_builtin_trace)
    /// 2. Feeds memory_address_to_id from lookup_data
    /// 3. Commits only the first 6 columns (poseidon_builtin's own trace)
    /// 4. Returns 6 ID columns (downloaded from GPU) for feeding the SIMD aggregator
    ///
    /// Chain generators, memory_id_to_big, and range checks are NOT fed here —
    /// they are fed via the SIMD aggregator conversion path in cairo_cuda.rs step 5.5.
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator, PoseidonBuiltinIds) {
        let log_size = self.log_size;
        let n_rows = 1usize << log_size;
        let (trace, lookup_data, _sub_component_inputs) = write_trace_cuda(
            n_rows,
            log_size,
            self.segment_start,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
        );

        // Feed memory_address_to_id from lookup_data (6 addresses per row)
        let cuda_addr_inputs: [memory_address_to_id_cuda::CudaPackedInputType; 6] = [
            [lookup_data.memory_address_to_id_0[0].clone()],
            [lookup_data.memory_address_to_id_1[0].clone()],
            [lookup_data.memory_address_to_id_2[0].clone()],
            [lookup_data.memory_address_to_id_3[0].clone()],
            [lookup_data.memory_address_to_id_4[0].clone()],
            [lookup_data.memory_address_to_id_5[0].clone()],
        ];
        memory_address_to_id_cuda_state.add_cuda_inputs(&cuda_addr_inputs);

        // Download 6 ID columns from GPU for feeding the SIMD aggregator.
        // The aggregator needs (input_ids[3], output_ids[3]) per row.
        let ids = PoseidonBuiltinIds {
            input_state_ids: [
                lookup_data.memory_address_to_id_0[1].to_cpu(),
                lookup_data.memory_address_to_id_1[1].to_cpu(),
                lookup_data.memory_address_to_id_2[1].to_cpu(),
            ],
            output_state_ids: [
                lookup_data.memory_address_to_id_3[1].to_cpu(),
                lookup_data.memory_address_to_id_4[1].to_cpu(),
                lookup_data.memory_address_to_id_5[1].to_cpu(),
            ],
        };

        // The CUDA kernel generates a monolithic 341-column trace for the entire poseidon
        // pipeline, but the v1.1.0 AIR splits poseidon into poseidon_builtin (6 cols) +
        // separate chain sub-components. Only commit the first 6 columns.

        // Extract base trace pointers BEFORE consuming trace data.
        // The interaction kernel expects columns 120-283 (164 columns).
        let base_trace_ptrs: Vec<*const u32> = trace.data[120..284]
            .iter()
            .map(|c| c.device_ptr as *const u32)
            .collect();

        let trace_log = trace.log_size;
        let domain = CanonicCoset::new(trace_log).circle_domain();

        // Split: first 6 columns for tree commit, rest kept alive for base_trace_ptrs.
        let mut all_cols: Vec<BaseFieldVec> = trace.data.into_iter().collect();
        let remaining_trace_cols = all_cols.split_off(6); // cols 6-340

        let builtin_evals: Vec<_> = all_cols
            .into_iter()
            .map(|col| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, col))
            .collect();
        tree_builder.extend_evals(builtin_evals);

        (
            Claim {
                log_size,
                poseidon_builtin_segment_start: self.segment_start,
            },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
                base_trace_ptrs,
                _remaining_trace_cols: remaining_trace_cols,
            },
            ids,
        )
    }
}

/// The 6 memory IDs from poseidon_builtin's lookup_data, downloaded from GPU.
/// Used to feed the SIMD poseidon_aggregator.
pub struct PoseidonBuiltinIds {
    pub input_state_ids: [Vec<M31>; 3],
    pub output_state_ids: [Vec<M31>; 3],
}

pub struct CudaSubComponentInputs {
    // 6 memory_address_to_id lookups
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 6],
    // 6 memory_id_to_big lookups
    pub memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 6],
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
                // 6 memory_address_to_id lookups (2 elements each)
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                memory_address_to_id_3: init_lookup_array!(log_size),
                memory_address_to_id_4: init_lookup_array!(log_size),
                memory_address_to_id_5: init_lookup_array!(log_size),
                // 6 memory_id_to_big lookups (29 elements each)
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                memory_id_to_big_3: init_lookup_array!(log_size),
                memory_id_to_big_4: init_lookup_array!(log_size),
                memory_id_to_big_5: init_lookup_array!(log_size),
                // 2 range_check_3_3_3_3_3 lookups (5 elements each)
                range_check_3_3_3_3_3_0: init_lookup_array!(log_size),
                range_check_3_3_3_3_3_1: init_lookup_array!(log_size),
                // 6 range_check_4_4_4_4 lookups (4 elements each)
                range_check_4_4_4_4_0: init_lookup_array!(log_size),
                range_check_4_4_4_4_1: init_lookup_array!(log_size),
                range_check_4_4_4_4_2: init_lookup_array!(log_size),
                range_check_4_4_4_4_3: init_lookup_array!(log_size),
                range_check_4_4_4_4_4: init_lookup_array!(log_size),
                range_check_4_4_4_4_5: init_lookup_array!(log_size),
                // 3 range_check_4_4 lookups (2 elements each)
                range_check_4_4_0: init_lookup_array!(log_size),
                range_check_4_4_1: init_lookup_array!(log_size),
                range_check_4_4_2: init_lookup_array!(log_size),
                // 8 poseidon_full_round_chain lookups (32 elements each)
                poseidon_full_round_chain_0: init_lookup_array!(log_size),
                poseidon_full_round_chain_1: init_lookup_array!(log_size),
                poseidon_full_round_chain_2: init_lookup_array!(log_size),
                poseidon_full_round_chain_3: init_lookup_array!(log_size),
                poseidon_full_round_chain_4: init_lookup_array!(log_size),
                poseidon_full_round_chain_5: init_lookup_array!(log_size),
                poseidon_full_round_chain_6: init_lookup_array!(log_size),
                poseidon_full_round_chain_7: init_lookup_array!(log_size),
                // 27 poseidon_3_partial_rounds_chain lookups (42 elements each)
                poseidon_3_partial_rounds_chain_0: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_1: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_2: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_3: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_4: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_5: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_6: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_7: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_8: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_9: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_10: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_11: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_12: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_13: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_14: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_15: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_16: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_17: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_18: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_19: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_20: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_21: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_22: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_23: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_24: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_25: init_lookup_array!(log_size),
                poseidon_3_partial_rounds_chain_26: init_lookup_array!(log_size),
                // Base trace columns 120-283 for interaction kernels (164 columns)
                base_trace_cols: init_lookup_array!(log_size),
            }),
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers - 6 memory_address_to_id
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_address_to_id_3 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_3);
    let lookup_memory_address_to_id_4 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_4);
    let lookup_memory_address_to_id_5 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_5);

    // Collect lookup data pointers - 6 memory_id_to_big
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_memory_id_to_big_3 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_3);
    let lookup_memory_id_to_big_4 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_4);
    let lookup_memory_id_to_big_5 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_5);

    // Collect lookup data pointers - 2 range_check_3_3_3_3_3
    let lookup_range_check_3_3_3_3_3_0 = collect_lookup_ptrs!(lookup_data, range_check_3_3_3_3_3_0);
    let lookup_range_check_3_3_3_3_3_1 = collect_lookup_ptrs!(lookup_data, range_check_3_3_3_3_3_1);

    // Collect lookup data pointers - 6 range_check_4_4_4_4
    let lookup_range_check_4_4_4_4_0 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_0);
    let lookup_range_check_4_4_4_4_1 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_1);
    let lookup_range_check_4_4_4_4_2 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_2);
    let lookup_range_check_4_4_4_4_3 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_3);
    let lookup_range_check_4_4_4_4_4 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_4);
    let lookup_range_check_4_4_4_4_5 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_5);

    // Collect lookup data pointers - 3 range_check_4_4
    let lookup_range_check_4_4_0 = collect_lookup_ptrs!(lookup_data, range_check_4_4_0);
    let lookup_range_check_4_4_1 = collect_lookup_ptrs!(lookup_data, range_check_4_4_1);
    let lookup_range_check_4_4_2 = collect_lookup_ptrs!(lookup_data, range_check_4_4_2);

    // Collect lookup data pointers - 8 poseidon_full_round_chain
    let lookup_poseidon_full_round_chain_0 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_0);
    let lookup_poseidon_full_round_chain_1 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_1);
    let lookup_poseidon_full_round_chain_2 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_2);
    let lookup_poseidon_full_round_chain_3 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_3);
    let lookup_poseidon_full_round_chain_4 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_4);
    let lookup_poseidon_full_round_chain_5 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_5);
    let lookup_poseidon_full_round_chain_6 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_6);
    let lookup_poseidon_full_round_chain_7 =
        collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_7);

    // Collect lookup data pointers - 27 poseidon_3_partial_rounds_chain
    let lookup_poseidon_3_partial_rounds_chain_0 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_0);
    let lookup_poseidon_3_partial_rounds_chain_1 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_1);
    let lookup_poseidon_3_partial_rounds_chain_2 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_2);
    let lookup_poseidon_3_partial_rounds_chain_3 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_3);
    let lookup_poseidon_3_partial_rounds_chain_4 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_4);
    let lookup_poseidon_3_partial_rounds_chain_5 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_5);
    let lookup_poseidon_3_partial_rounds_chain_6 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_6);
    let lookup_poseidon_3_partial_rounds_chain_7 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_7);
    let lookup_poseidon_3_partial_rounds_chain_8 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_8);
    let lookup_poseidon_3_partial_rounds_chain_9 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_9);
    let lookup_poseidon_3_partial_rounds_chain_10 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_10);
    let lookup_poseidon_3_partial_rounds_chain_11 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_11);
    let lookup_poseidon_3_partial_rounds_chain_12 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_12);
    let lookup_poseidon_3_partial_rounds_chain_13 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_13);
    let lookup_poseidon_3_partial_rounds_chain_14 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_14);
    let lookup_poseidon_3_partial_rounds_chain_15 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_15);
    let lookup_poseidon_3_partial_rounds_chain_16 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_16);
    let lookup_poseidon_3_partial_rounds_chain_17 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_17);
    let lookup_poseidon_3_partial_rounds_chain_18 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_18);
    let lookup_poseidon_3_partial_rounds_chain_19 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_19);
    let lookup_poseidon_3_partial_rounds_chain_20 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_20);
    let lookup_poseidon_3_partial_rounds_chain_21 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_21);
    let lookup_poseidon_3_partial_rounds_chain_22 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_22);
    let lookup_poseidon_3_partial_rounds_chain_23 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_23);
    let lookup_poseidon_3_partial_rounds_chain_24 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_24);
    let lookup_poseidon_3_partial_rounds_chain_25 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_25);
    let lookup_poseidon_3_partial_rounds_chain_26 =
        collect_lookup_ptrs!(lookup_data, poseidon_3_partial_rounds_chain_26);

    // Collect lookup data pointers - base_trace_cols (cols 120-283)
    let lookup_base_trace_cols = collect_lookup_ptrs!(lookup_data, base_trace_cols);

    // Transpose memory_id_to_big big_values for GPU access
    let memory_id_to_big_transposed_big_values_vec: Vec<_> = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect();

    // Call CUDA kernel
    unsafe {
        bindings_airs::gen_poseidon_builtin_trace(
            traces_vec.as_ptr(),
            log_size,
            segment_start,
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,
            // 6 memory_address_to_id lookups
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            lookup_memory_address_to_id_3.as_ptr(),
            lookup_memory_address_to_id_4.as_ptr(),
            lookup_memory_address_to_id_5.as_ptr(),
            // 6 memory_id_to_big lookups
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            lookup_memory_id_to_big_3.as_ptr(),
            lookup_memory_id_to_big_4.as_ptr(),
            lookup_memory_id_to_big_5.as_ptr(),
            // 2 range_check_3_3_3_3_3 lookups
            lookup_range_check_3_3_3_3_3_0.as_ptr(),
            lookup_range_check_3_3_3_3_3_1.as_ptr(),
            // 6 range_check_4_4_4_4 lookups
            lookup_range_check_4_4_4_4_0.as_ptr(),
            lookup_range_check_4_4_4_4_1.as_ptr(),
            lookup_range_check_4_4_4_4_2.as_ptr(),
            lookup_range_check_4_4_4_4_3.as_ptr(),
            lookup_range_check_4_4_4_4_4.as_ptr(),
            lookup_range_check_4_4_4_4_5.as_ptr(),
            // 3 range_check_4_4 lookups
            lookup_range_check_4_4_0.as_ptr(),
            lookup_range_check_4_4_1.as_ptr(),
            lookup_range_check_4_4_2.as_ptr(),
            // 8 poseidon_full_round_chain lookups
            lookup_poseidon_full_round_chain_0.as_ptr(),
            lookup_poseidon_full_round_chain_1.as_ptr(),
            lookup_poseidon_full_round_chain_2.as_ptr(),
            lookup_poseidon_full_round_chain_3.as_ptr(),
            lookup_poseidon_full_round_chain_4.as_ptr(),
            lookup_poseidon_full_round_chain_5.as_ptr(),
            lookup_poseidon_full_round_chain_6.as_ptr(),
            lookup_poseidon_full_round_chain_7.as_ptr(),
            // 27 poseidon_3_partial_rounds_chain lookups
            lookup_poseidon_3_partial_rounds_chain_0.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_1.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_2.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_3.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_4.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_5.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_6.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_7.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_8.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_9.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_10.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_11.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_12.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_13.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_14.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_15.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_16.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_17.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_18.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_19.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_20.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_21.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_22.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_23.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_24.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_25.as_ptr(),
            lookup_poseidon_3_partial_rounds_chain_26.as_ptr(),
            // Base trace cols 120-283 (164 columns)
            lookup_base_trace_cols.as_ptr(),
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    // 6 memory_address_to_id lookups (2 elements each)
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    memory_address_to_id_3: [BaseFieldVec; 2],
    memory_address_to_id_4: [BaseFieldVec; 2],
    memory_address_to_id_5: [BaseFieldVec; 2],
    // 6 memory_id_to_big lookups (29 elements each)
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    memory_id_to_big_3: [BaseFieldVec; 29],
    memory_id_to_big_4: [BaseFieldVec; 29],
    memory_id_to_big_5: [BaseFieldVec; 29],
    // 2 range_check_3_3_3_3_3 lookups (5 elements each)
    range_check_3_3_3_3_3_0: [BaseFieldVec; 5],
    range_check_3_3_3_3_3_1: [BaseFieldVec; 5],
    // 6 range_check_4_4_4_4 lookups (4 elements each)
    range_check_4_4_4_4_0: [BaseFieldVec; 4],
    range_check_4_4_4_4_1: [BaseFieldVec; 4],
    range_check_4_4_4_4_2: [BaseFieldVec; 4],
    range_check_4_4_4_4_3: [BaseFieldVec; 4],
    range_check_4_4_4_4_4: [BaseFieldVec; 4],
    range_check_4_4_4_4_5: [BaseFieldVec; 4],
    // 3 range_check_4_4 lookups (2 elements each)
    range_check_4_4_0: [BaseFieldVec; 2],
    range_check_4_4_1: [BaseFieldVec; 2],
    range_check_4_4_2: [BaseFieldVec; 2],
    // 8 poseidon_full_round_chain lookups (32 elements each)
    // pfrc_0: input to round 0, pfrc_1: input to round 31
    // pfrc_2-4: intermediate states for rounds 1-3, pfrc_5-7: intermediate states for rounds 32-34
    poseidon_full_round_chain_0: [BaseFieldVec; 32],
    poseidon_full_round_chain_1: [BaseFieldVec; 32],
    poseidon_full_round_chain_2: [BaseFieldVec; 32],
    poseidon_full_round_chain_3: [BaseFieldVec; 32],
    poseidon_full_round_chain_4: [BaseFieldVec; 32],
    poseidon_full_round_chain_5: [BaseFieldVec; 32],
    poseidon_full_round_chain_6: [BaseFieldVec; 32],
    poseidon_full_round_chain_7: [BaseFieldVec; 32],
    // 27 poseidon_3_partial_rounds_chain lookups (42 elements each)
    poseidon_3_partial_rounds_chain_0: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_1: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_2: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_3: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_4: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_5: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_6: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_7: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_8: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_9: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_10: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_11: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_12: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_13: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_14: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_15: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_16: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_17: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_18: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_19: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_20: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_21: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_22: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_23: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_24: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_25: [BaseFieldVec; 42],
    poseidon_3_partial_rounds_chain_26: [BaseFieldVec; 42],
    // Base trace columns 120-283 for interaction kernels (164 columns)
    base_trace_cols: [BaseFieldVec; 164],
}

pub struct CudaInteractionClaimGenerator {
    #[allow(dead_code)]
    n_rows: usize,
    log_size: u32,
    lookup_data: Box<CudaLookupData>,
    /// Base trace column pointers - used for reconstructing lookup data in interaction kernel
    base_trace_ptrs: Vec<*const u32>,
    /// Keep GPU memory alive for base_trace_ptrs (columns 6-340 of the monolithic trace)
    _remaining_trace_cols: Vec<BaseFieldVec>,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        // Allocate GPU buffer for claimed_sum (4 M31 = 1 QM31)
        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Allocate GPU buffers for interaction trace columns (18 logup cols × 4 M31 = 72 columns)
        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..N_INTERACTION_TRACE_COLUMNS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(1 << trace_log_size) })
            .collect_vec();

        // Collect lookup data pointers - memory_address_to_id
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

        // Collect lookup data pointers - memory_id_to_big
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

        // Collect lookup data pointers - range_check_3_3_3_3_3
        let lookup_range_check_3_3_3_3_3_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_3_3_3_3_0);
        let lookup_range_check_3_3_3_3_3_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_3_3_3_3_3_1);

        // Collect lookup data pointers - range_check_4_4_4_4
        let lookup_range_check_4_4_4_4_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_0);
        let lookup_range_check_4_4_4_4_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_1);
        let lookup_range_check_4_4_4_4_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_2);
        let lookup_range_check_4_4_4_4_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_3);
        let lookup_range_check_4_4_4_4_4_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_4);
        let lookup_range_check_4_4_4_4_5_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_5);

        // Collect lookup data pointers - range_check_4_4
        let lookup_range_check_4_4_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_0);
        let lookup_range_check_4_4_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_1);
        let lookup_range_check_4_4_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_4_4_2);

        // Collect lookup data pointers - poseidon_full_round_chain
        let lookup_poseidon_full_round_chain_0_vec =
            collect_lookup_ptrs!(self.lookup_data, poseidon_full_round_chain_0);
        let lookup_poseidon_full_round_chain_1_vec =
            collect_lookup_ptrs!(self.lookup_data, poseidon_full_round_chain_1);

        // Collect lookup data pointers - base_trace_cols (cols 120-283)
        let lookup_base_trace_cols_vec = collect_lookup_ptrs!(self.lookup_data, base_trace_cols);

        let interaction_trace_vec: Vec<*const u32> = interaction_trace
            .iter()
            .map(|col| col.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_mem_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem_id_big =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
            let mod_pfrc = create_modified_lookup_for_cuda(
                lookup_elements,
                POSEIDON_FULL_ROUND_CHAIN_RELATION_ID,
            );
            let mod_rc_252 = create_modified_lookup_for_cuda(
                lookup_elements,
                RANGE_CHECK_252_WIDTH_27_RELATION_ID,
            );
            let mod_cube_252 =
                create_modified_lookup_for_cuda(lookup_elements, CUBE_252_RELATION_ID);
            let mod_rc_3_3_3_3_3 =
                create_modified_lookup_for_cuda(lookup_elements, RANGE_CHECK_3_3_3_3_3_RELATION_ID);
            let mod_rc_4_4_4_4 =
                create_modified_lookup_for_cuda(lookup_elements, RC_4_4_4_4_RELATION_ID);
            let mod_rc_4_4 =
                create_modified_lookup_for_cuda(lookup_elements, RANGE_CHECK_4_4_RELATION_ID);
            let mod_p3prc = create_modified_lookup_for_cuda(
                lookup_elements,
                POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID,
            );

            bindings_airs::gen_poseidon_builtin_interaction_trace(
                interaction_trace_vec.as_ptr(),
                trace_log_size,
                // Relation structs (each with relation-specific modified z)
                &mod_mem_addr as *const _ as *const u32,
                &mod_mem_id_big as *const _ as *const u32,
                &mod_pfrc as *const _ as *const u32,
                &mod_rc_252 as *const _ as *const u32,
                &mod_cube_252 as *const _ as *const u32,
                &mod_rc_3_3_3_3_3 as *const _ as *const u32,
                &mod_rc_4_4_4_4 as *const _ as *const u32,
                &mod_rc_4_4 as *const _ as *const u32,
                &mod_p3prc as *const _ as *const u32,
                // Base trace columns for reconstructing all lookup data
                self.base_trace_ptrs.as_ptr(),
                // memory_address_to_id lookups (6)
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_address_to_id_3_vec.as_ptr(),
                lookup_memory_address_to_id_4_vec.as_ptr(),
                lookup_memory_address_to_id_5_vec.as_ptr(),
                // memory_id_to_big lookups (6)
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_memory_id_to_big_3_vec.as_ptr(),
                lookup_memory_id_to_big_4_vec.as_ptr(),
                lookup_memory_id_to_big_5_vec.as_ptr(),
                // range_check_3_3_3_3_3 lookups (2)
                lookup_range_check_3_3_3_3_3_0_vec.as_ptr(),
                lookup_range_check_3_3_3_3_3_1_vec.as_ptr(),
                // range_check_4_4_4_4 lookups (6)
                lookup_range_check_4_4_4_4_0_vec.as_ptr(),
                lookup_range_check_4_4_4_4_1_vec.as_ptr(),
                lookup_range_check_4_4_4_4_2_vec.as_ptr(),
                lookup_range_check_4_4_4_4_3_vec.as_ptr(),
                lookup_range_check_4_4_4_4_4_vec.as_ptr(),
                lookup_range_check_4_4_4_4_5_vec.as_ptr(),
                // range_check_4_4 lookups (3)
                lookup_range_check_4_4_0_vec.as_ptr(),
                lookup_range_check_4_4_1_vec.as_ptr(),
                lookup_range_check_4_4_2_vec.as_ptr(),
                // poseidon_full_round_chain lookups (2)
                lookup_poseidon_full_round_chain_0_vec.as_ptr(),
                lookup_poseidon_full_round_chain_1_vec.as_ptr(),
                // base_trace_cols (cols 120-283, 164 columns)
                lookup_base_trace_cols_vec.as_ptr(),
                cuda_claimed_sum.device_ptr as *const u32,
            );
        }

        // The monolithic kernel generates 68 columns (17 logup × 4 M31) for the entire
        // poseidon pipeline. Only the first 16 columns (4 logup × 4 M31) belong to
        // poseidon_builtin's own 7 lookups (6 MemoryAddressToId + 1 PoseidonAggregator).
        //
        // The kernel's claimed_sum covers all 17 logup columns, but we only need
        // the partial sum for the first 4. Recompute from the last row of each
        // logup column (prefix-sum encoding: last row = column's total sum).
        let n_builtin_interaction_cols = 4 * 4; // 4 logup columns × 4 M31
        let n = 1usize << trace_log_size;
        let mut claimed_sum = SecureField::zero();
        for logup_col in 0..4 {
            let base = logup_col * 4;
            let m31s: [M31; 4] = std::array::from_fn(|i| interaction_trace[base + i].at(n - 1));
            claimed_sum += SecureField::from_m31_array(m31s);
        }

        // Extend tree builder with poseidon_builtin's own interaction trace (first 16 cols).
        let domain = CanonicCoset::new(trace_log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .take(n_builtin_interaction_cols)
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
