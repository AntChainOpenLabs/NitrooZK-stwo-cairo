#![allow(unused_parens)]
#![allow(dead_code)]

use cairo_air::components::blake_round_sigma::{Claim, InteractionClaim, LOG_SIZE};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

use crate::witness::prelude::*;

pub type InputType = [M31; 1];
pub type PackedInputType = [PackedM31; 1];
pub type CudaPackedInputType = [BaseFieldVec; 1];

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::bindings_airs;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 1;

pub struct CudaClaimGenerator {
    pub mults: Vec<BaseFieldVec>,
}
impl CudaClaimGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            mults: (0..1)
                .map(|_| BaseFieldVec::new_zeroes(1 << LOG_SIZE))
                .collect_vec(),
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let mults = self.mults;

        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let trace = mults
            .clone()
            .into_iter()
            .map(|col| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, col))
            .collect();
        let lookup_data = CudaLookupData { mults };

        tree_builder.extend_evals(trace);

        (Claim {}, CudaInteractionClaimGenerator { lookup_data })
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {
        let cuda_inputs: [BaseFieldVec; 1] = (0..1)
            .map(|i| {
                let elements: Vec<_> = packed_inputs
                    .iter()
                    .flat_map(|input| input[i].to_array())
                    .collect();
                BaseFieldVec::from_vec(unsafe { std::mem::transmute(elements) })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Inputs should have exactly 3 elements");

        let cuda_input_vec = cuda_inputs.iter().map(|row| row.device_ptr).collect_vec();
        let mults_vec = self.mults.iter().map(|row| row.device_ptr).collect_vec();
        unsafe {
            bindings_airs::blake_round_sigma_mults_init(
                cuda_input_vec.as_ptr(),
                1,
                (1 << LOG_SIZE) as u32,
                mults_vec.as_ptr(),
                1,
                LOG_SIZE,
            );
        }
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        // println!("cuda_inputs verify_bitwise_xor_4_cuda inputs cols: {}, rows:{}, mults
        // log_size:{}", cuda_inputs.len(), cuda_inputs[0][0].size, LOG_SIZE);
        let inputs_vec = cuda_inputs
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        let mults_vec = self.mults.iter().map(|row| row.device_ptr).collect_vec();
        unsafe {
            bindings_airs::blake_round_sigma_mults_init(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                mults_vec.as_ptr(),
                1,
                LOG_SIZE,
            );
        }
    }

    /// Merges SIMD multiplicities into this CUDA generator.
    /// This should be called before write_trace() to ensure all multiplicities
    /// from SIMD components are included.
    ///
    /// # Arguments
    /// * `simd_multiplicities` - Flattened SIMD multiplicities from
    ///   blake_round_sigma::ClaimGenerator
    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
        let col_size = 1usize << LOG_SIZE;

        for col_idx in 0..self.mults.len() {
            // Get current CUDA multiplicities for this column
            let mut cuda_mults = self.mults[col_idx].to_vec();

            // Get corresponding SIMD multiplicities slice
            let simd_start = col_idx * col_size;
            let simd_end = std::cmp::min(simd_start + col_size, simd_multiplicities.len());

            if simd_start < simd_multiplicities.len() {
                let simd_slice = &simd_multiplicities[simd_start..simd_end];

                // Add SIMD multiplicities to CUDA multiplicities
                for (i, &simd_val) in simd_slice.iter().enumerate() {
                    // Convert M31 inner value to u32, add, then convert back to M31
                    cuda_mults[i] = M31(cuda_mults[i].0.wrapping_add(simd_val));
                }

                // Write back merged multiplicities to GPU
                self.mults[col_idx] = BaseFieldVec::from_vec(cuda_mults);
            }
        }
    }
}

struct CudaLookupData {
    mults: Vec<BaseFieldVec>,
}

pub struct CudaInteractionClaimGenerator {
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.lookup_data.mults[0].size.ilog2();

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_blake_round_sigma_vec = self
            .lookup_data
            .mults
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_blake_round_sigma =
                create_modified_lookup_for_cuda(lookup_elements, BLAKE_ROUND_SIGMA_RELATION_ID);

            bindings_airs::generate_blake_round_sigma_interaction_traces(
                &mod_blake_round_sigma as *const _ as *mut std::os::raw::c_void,
                lookup_blake_round_sigma_vec.as_ptr(),
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

        println!("cuda blake_round_sigma claimed sum: {:?}", claimed_sum);
        InteractionClaim { claimed_sum }
    }
}
