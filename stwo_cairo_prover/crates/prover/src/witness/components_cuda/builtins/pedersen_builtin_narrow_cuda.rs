// Native CUDA witness generation for pedersen_builtin_narrow_windows (3 columns).
//
// Narrow variant (window_bits_9). Identical structure to the wide variant
// (pedersen_builtin_cuda.rs) — only the aggregator type differs:
//   Wide:   pedersen_aggregator_window_bits_18 (relation_id = 520578465)
//   Narrow: pedersen_aggregator_window_bits_9  (relation_id = 194336987)
//
// Base trace: 3 columns (input_state_0_id, input_state_1_id, output_state_id)
// Lookup data: 3x memory_address_to_id (3 elems each) + 1x pedersen_agg (4 elems)
// Sub-component inputs: 3 memory_address_to_id feeds + 1 pedersen_aggregator feed
// Interaction trace: 2 logup columns (8 M31 columns)

use cairo_air::components::pedersen_builtin_narrow_windows::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

use super::super::memory_address_to_id_cuda;
use crate::witness::prelude::*;

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

    /// Write the pedersen_builtin_narrow_windows trace (3 columns) using native CUDA kernel.
    ///
    /// Routes sub-component inputs:
    /// - memory_address_to_id -> CUDA state (GPU-native multiplicity tracking)
    /// - pedersen_aggregator → returns 3 GPU ID arrays for direct GPU transfer
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator, [BaseFieldVec; 3]) {
        let log_size = self.log_size;
        let n_rows = 1u32 << log_size;
        let trace_size = n_rows as usize;

        // Allocate GPU columns for 3 trace columns.
        let trace_cols: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));

        // Allocate GPU arrays for lookup data (3x3 + 1x4 = 13 arrays).
        let lk_mem_0: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));
        let lk_mem_1: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));
        let lk_mem_2: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));
        let lk_agg_0: [BaseFieldVec; 4] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));

        // Allocate GPU arrays for sub-component inputs (3 + 3 = 6 arrays).
        let sub_mem: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));
        let sub_agg: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size));

        // Collect device pointers.
        let traces_ptrs: Vec<*const u32> = trace_cols.iter().map(|c| c.device_ptr).collect();
        let lk_mem_0_ptrs: Vec<*const u32> = lk_mem_0.iter().map(|c| c.device_ptr).collect();
        let lk_mem_1_ptrs: Vec<*const u32> = lk_mem_1.iter().map(|c| c.device_ptr).collect();
        let lk_mem_2_ptrs: Vec<*const u32> = lk_mem_2.iter().map(|c| c.device_ptr).collect();
        let lk_agg_0_ptrs: Vec<*const u32> = lk_agg_0.iter().map(|c| c.device_ptr).collect();
        let sub_mem_ptrs: Vec<*const u32> = sub_mem.iter().map(|c| c.device_ptr).collect();
        let sub_agg_ptrs: Vec<*const u32> = sub_agg.iter().map(|c| c.device_ptr).collect();

        // Call the CUDA kernel.
        unsafe {
            bindings_airs::gen_pedersen_builtin_narrow_trace(
                traces_ptrs.as_ptr(),
                lk_mem_0_ptrs.as_ptr(),
                lk_mem_1_ptrs.as_ptr(),
                lk_mem_2_ptrs.as_ptr(),
                lk_agg_0_ptrs.as_ptr(),
                sub_mem_ptrs.as_ptr(),
                sub_agg_ptrs.as_ptr(),
                memory_address_to_id_cuda_state.address_to_raw_id.device_ptr,
                self.segment_start,
                n_rows,
                log_size,
            );
        }

        // Route sub-component inputs: memory_address_to_id -> CUDA state.
        let mem_inputs: Vec<memory_address_to_id_cuda::CudaPackedInputType> =
            sub_mem.into_iter().map(|v| [v]).collect();
        memory_address_to_id_cuda_state.add_cuda_inputs(&mem_inputs);

        // Wrap trace columns as CircleEvaluations and extend tree builder.
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: Vec<CircleEvaluation<CudaBackend, M31, BitReversedOrder>> = trace_cols
            .into_iter()
            .map(|col| CircleEvaluation::new(domain, col))
            .collect();
        tree_builder.extend_evals(trace_evals);

        (
            Claim {
                log_size,
                pedersen_builtin_segment_start: self.segment_start,
            },
            CudaInteractionClaimGenerator {
                log_size,
                lookup_data: CudaLookupData {
                    lk_mem_0,
                    lk_mem_1,
                    lk_mem_2,
                    lk_agg_0,
                },
            },
            sub_agg,
        )
    }
}

struct CudaLookupData {
    lk_mem_0: [BaseFieldVec; 3],
    lk_mem_1: [BaseFieldVec; 3],
    lk_mem_2: [BaseFieldVec; 3],
    lk_agg_0: [BaseFieldVec; 4],
}

pub struct CudaInteractionClaimGenerator {
    log_size: u32,
    lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let log_size = self.log_size;
        let trace_size = 1usize << log_size;
        let n_interaction_columns = 4 * 2; // 2 logup columns x 4 M31 each

        // Allocate GPU columns for interaction trace.
        let interaction_trace: Vec<BaseFieldVec> = (0..n_interaction_columns)
            .map(|_| BaseFieldVec::new_zeroes(trace_size))
            .collect();

        // Allocate GPU buffer for claimed_sum (4 M31s = 1 QM31).
        let cuda_claimed_sum = BaseFieldVec::new_zeroes(4);

        // Collect pointers.
        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|c| c.device_ptr).collect();
        let lk_mem_0_ptrs: Vec<*const u32> = self
            .lookup_data
            .lk_mem_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lk_mem_1_ptrs: Vec<*const u32> = self
            .lookup_data
            .lk_mem_1
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lk_mem_2_ptrs: Vec<*const u32> = self
            .lookup_data
            .lk_mem_2
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lk_agg_0_ptrs: Vec<*const u32> = self
            .lookup_data
            .lk_agg_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();

        unsafe {
            bindings_airs::gen_pedersen_builtin_narrow_interaction_trace(
                lookup_elements as *const CommonLookupElements as *mut std::os::raw::c_void,
                lk_mem_0_ptrs.as_ptr(),
                lk_mem_1_ptrs.as_ptr(),
                lk_mem_2_ptrs.as_ptr(),
                lk_agg_0_ptrs.as_ptr(),
                log_size,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        // Read claimed_sum from GPU.
        let cs = cuda_claimed_sum.to_vec();
        let claimed_sum = SecureField::from_m31_array([cs[0], cs[1], cs[2], cs[3]]);

        // Wrap interaction trace columns as CircleEvaluations and extend tree builder.
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: Vec<CircleEvaluation<CudaBackend, M31, BitReversedOrder>> =
            interaction_trace
                .into_iter()
                .map(|col| CircleEvaluation::new(domain, col))
                .collect();
        tree_builder.extend_evals(trace_evals);

        InteractionClaim { claimed_sum }
    }
}
