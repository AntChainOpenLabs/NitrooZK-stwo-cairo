#![allow(unused_parens)]

use cairo_air::components::range_check_252_width_27::{Claim, InteractionClaim, N_TRACE_COLUMNS};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use crate::witness::components_cuda::range_check::{rc_18, rc_9_9};
use crate::witness::prelude::*;

/// Number of interaction trace columns for range_check_252_width_27.
/// 8 logup columns * 4 (for SecureField) = 32
pub const N_INTERACTION_TRACE_COLUMNS: usize = 8;

/// Input format: Width27 (10 x 27-bit limbs representing a Felt252)
pub type PackedInputType = PackedFelt252Width27;

/// CUDA packed input type - 10 columns for Width27 format
pub type CudaPackedInputType = [BaseFieldVec; 10];

macro_rules! collect_ptrs {
    ($arr:expr) => {
        $arr.iter().map(|x| x.device_ptr).collect_vec()
    };
}

pub struct CudaClaimGenerator {
    /// Input data in Width27 format (10 columns)
    pub packed_inputs: [BaseFieldVec; 10],
    /// Number of rows
    pub size: u32,
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            packed_inputs: std::array::from_fn(|_| BaseFieldVec::new_zeroes(1)),
            size: 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size <= 1
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        // Range check dependencies
        range_check_18_state: &rc_18::CudaClaimGenerator,
        range_check_9_9_state: &rc_9_9::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let n_rows = self.size as usize;
        let log_size = n_rows.next_power_of_two().ilog2();
        let padded_size = 1usize << log_size;
        // Pad input buffers to power-of-two size (kernel expects padded_size elements)
        // IMPORTANT: Must pad with first PACKED input's values (16 rows) to match SIMD behavior
        // SIMD pads at packed input level: packed_inputs.resize(packed_size, *first_packed_input)
        // Each packed input contains 16 rows, so we need to replicate the first 16 rows
        // GPU in-place padding — no download/upload roundtrip.
        const N_LANES: usize = 16;
        if n_rows < padded_size {
            for input in self.packed_inputs.iter_mut() {
                input.pad_with_cycle(n_rows, padded_size, N_LANES);
            }
        }

        // Generate trace
        let trace = write_trace_cuda(&self.packed_inputs, n_rows, log_size);

        // Add to range check multiplicities.
        // NOTE: Use padded_size instead of n_rows because padding rows also contribute to
        // multiplicities (padding rows have zero values, which map to index 0 in range
        // check tables)
        add_to_range_check_multiplicities(
            &trace,
            padded_size as u32,
            range_check_18_state,
            range_check_9_9_state,
        );

        // Clone trace data for interaction generator before consuming for evals
        let trace_data_clone: [BaseFieldVec; N_TRACE_COLUMNS] = trace.data.clone();

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                trace_data: trace_data_clone,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {
        // Convert SIMD packed inputs to CUDA format
        // For each of the 10 limb positions, extract values from all lanes
        self.packed_inputs = (0..10)
            .map(|i| {
                let elements: Vec<M31> = packed_inputs
                    .iter()
                    .flat_map(|input| input.get_m31(i).to_array())
                    .collect();
                BaseFieldVec::from_vec(elements)
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Should have exactly 10 elements");

        self.size = self.packed_inputs[0].size as u32;
    }

    pub fn add_cuda_inputs(&mut self, inputs: &CudaPackedInputType) {
        if inputs[0].size == 0 {
            return; // Skip empty inputs
        }
        if self.size <= 1 {
            // First real data: replace initial placeholder zeros instead of extending.
            for (input_slice, new_slice) in self.packed_inputs.iter_mut().zip(inputs.iter()) {
                *input_slice = new_slice.clone();
            }
        } else {
            for (input_slice, new_slice) in self.packed_inputs.iter_mut().zip(inputs.iter()) {
                input_slice.extend(new_slice);
            }
        }
        self.size = self.packed_inputs[0].size as u32;
    }
}

fn write_trace_cuda(
    inputs: &[BaseFieldVec; 10],
    n_rows: usize,
    log_size: u32,
) -> CudaComponentTrace<N_TRACE_COLUMNS> {
    let trace = unsafe { CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size) };

    let input_ptrs = collect_ptrs!(inputs);
    let trace_ptrs = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    unsafe {
        bindings_airs::range_check_felt_252_width_27_generate_trace(
            input_ptrs.as_ptr(),
            1 << log_size, // padded size
            n_rows as u32, // actual data rows (for enabler column)
            trace_ptrs.as_ptr(),
        );
    }

    trace
}

fn add_to_range_check_multiplicities(
    trace: &CudaComponentTrace<N_TRACE_COLUMNS>,
    n_rows: u32,
    rc_18: &rc_18::CudaClaimGenerator,
    rc_9_9: &rc_9_9::CudaClaimGenerator,
) {
    let trace_ptrs = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    let rc_18_log_size = rc_18.log_size();
    let rc_9_9_log_size = rc_9_9.log_size();

    unsafe {
        bindings_airs::range_check_felt_252_width_27_add_to_multiplicities(
            trace_ptrs.as_ptr(),
            n_rows,
            // rc_18 multiplicities (relation_index 0)
            rc_18.multiplicities_ptr(0),
            rc_18_log_size,
            // rc_18_b multiplicities (relation_index 1)
            rc_18.multiplicities_ptr(1),
            rc_18_log_size,
            // rc_9_9 multiplicities (relation_index 0)
            rc_9_9.multiplicities_ptr(0),
            rc_9_9_log_size,
            // rc_9_9_b multiplicities (relation_index 1)
            rc_9_9.multiplicities_ptr(1),
            rc_9_9_log_size,
            // rc_9_9_c multiplicities (relation_index 2)
            rc_9_9.multiplicities_ptr(2),
            rc_9_9_log_size,
            // rc_9_9_d multiplicities (relation_index 3)
            rc_9_9.multiplicities_ptr(3),
            rc_9_9_log_size,
            // rc_9_9_e multiplicities (relation_index 4)
            rc_9_9.multiplicities_ptr(4),
            rc_9_9_log_size,
        );
    }
}

pub struct CudaInteractionClaimGenerator {
    pub log_size: u32,
    pub trace_data: [BaseFieldVec; N_TRACE_COLUMNS],
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;
        let trace_size = 1u32 << trace_log_size;

        // Allocate interaction trace columns (8 logup columns * 4 for qm31)
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(trace_size as usize) })
            .collect_vec();

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let trace_ptrs = self.trace_data.iter().map(|c| c.device_ptr).collect_vec();
        let interaction_trace_ptrs = interaction_trace.iter().map(|c| c.device_ptr).collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;

            let mod_rc_9_9 =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[0]);
            let mod_rc_18 = create_modified_lookup_for_cuda(lookup_elements, RC_18_RELATION_ID);
            let mod_rc_9_9_b =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[1]);
            let mod_rc_18_b = create_modified_lookup_for_cuda(lookup_elements, RC_18_B_RELATION_ID);
            let mod_rc_9_9_c =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[2]);
            let mod_rc_9_9_d =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[3]);
            let mod_rc_9_9_e =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[4]);
            let mod_rc_252 = create_modified_lookup_for_cuda(
                lookup_elements,
                RANGE_CHECK_252_WIDTH_27_RELATION_ID,
            );

            bindings_airs::range_check_felt_252_width_27_generate_interaction_trace(
                trace_ptrs.as_ptr(),
                trace_size,
                // Lookup elements (each with relation-specific modified z)
                &mod_rc_9_9 as *const _ as *mut std::os::raw::c_void,
                &mod_rc_18 as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_b as *const _ as *mut std::os::raw::c_void,
                &mod_rc_18_b as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_c as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_d as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_e as *const _ as *mut std::os::raw::c_void,
                &mod_rc_252 as *const _ as *mut std::os::raw::c_void,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
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

        println!(
            "range_check_252_width_27 cuda claimed_sum: {:?}",
            claimed_sum
        );
        InteractionClaim { claimed_sum }
    }
}
