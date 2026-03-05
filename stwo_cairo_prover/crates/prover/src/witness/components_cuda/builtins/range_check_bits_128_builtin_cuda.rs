// CUDA witness generation for range_check_builtin component (128-bit)
// This handles 128-bit range check operations

#![allow(unused_parens)]
use cairo_air::components::range_check_builtin::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::prelude::*;
pub const N_TRACE_COLUMNS: usize = 17;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 1;

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
                range_check_builtin_segment_start: self.segment_start,
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
    // 1 memory_address_to_id lookup
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 1],
    // 1 memory_id_to_big lookup
    pub memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 1],
}

pub struct CudaLookupData {
    // 1 memory_address_to_id lookup (2 elements)
    pub memory_address_to_id_0: [BaseFieldVec; 2],
    // 1 memory_id_to_big lookup (29 elements)
    pub memory_id_to_big_0: [BaseFieldVec; 29],
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
                // 1 memory_address_to_id lookup (2 elements)
                memory_address_to_id_0: init_lookup_array!(log_size),
                // 1 memory_id_to_big lookup (29 elements)
                memory_id_to_big_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);

    // Collect sub-component input pointers
    let sub_inputs_memory_address_to_id =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_inputs_memory_id_to_big =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);

    // Collect memory_id_to_big transposed_big_values pointers
    let memory_id_to_big_transposed_big_values_vec: Vec<_> = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect();

    unsafe {
        bindings_airs::generate_range_check_builtin_bits_128_traces(
            traces_vec.as_ptr(),
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_id_to_big_0.as_ptr(),
            sub_inputs_memory_address_to_id.as_ptr(),
            sub_inputs_memory_id_to_big.as_ptr(),
            segment_start,
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,
            n_rows as u32,
            log_size,
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

pub struct CudaInteractionClaimGenerator {
    pub n_rows: usize,
    pub log_size: u32,
    pub lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        // Allocate interaction trace columns (1 logical column = 4 M31 columns)
        let interaction_trace: Vec<BaseFieldVec> = (0..N_INTERACTION_TRACE_COLUMNS * 4)
            .map(|_| unsafe { BaseFieldVec::uninitialized(self.n_rows) })
            .collect();
        let interaction_trace_ptrs: Vec<_> =
            interaction_trace.iter().map(|c| c.device_ptr).collect();

        // Collect lookup data pointers
        let lookup_memory_address_to_id_0 =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);

        let mut claimed_sum = [0u32; 4];

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);

            bindings_airs::generate_range_check_builtin_bits_128_interaction_traces(
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                lookup_memory_address_to_id_0.as_ptr(),
                lookup_memory_id_to_big_0.as_ptr(),
                self.n_rows as u32,
                self.log_size,
                interaction_trace_ptrs.as_ptr(),
                claimed_sum.as_mut_ptr(),
            );
        }

        // Convert interaction trace to CircleEvaluations
        let domain = stwo::core::poly::circle::CanonicCoset::new(self.log_size).circle_domain();
        let evals: Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> =
            interaction_trace
                .into_iter()
                .map(|col| CircleEvaluation::new(domain, Col::<CudaBackend, BaseField>::from(col)))
                .collect();

        tree_builder.extend_evals(evals);

        // Convert claimed_sum from u32 array to SecureField
        let claimed_sum = SecureField::from_m31_array(std::array::from_fn(|i| {
            M31::from_u32_unchecked(claimed_sum[i])
        }));

        InteractionClaim { claimed_sum }
    }
}
