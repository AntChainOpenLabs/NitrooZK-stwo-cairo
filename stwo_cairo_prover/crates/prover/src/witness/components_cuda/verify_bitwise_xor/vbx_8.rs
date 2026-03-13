use std::ops::{Deref, DerefMut};

use cairo_air::components::verify_bitwise_xor_8::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;

use super::base::CudaVerifyBitwiseXorGenerator;
use super::vbx_8_b;
use crate::witness::utils::TreeBuilder;

pub const N_BITS: u32 = 8;

// Re-export CudaPackedInputType for external use
pub use super::base::CudaPackedInputType;

pub struct CudaClaimGenerator(CudaVerifyBitwiseXorGenerator);

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self(CudaVerifyBitwiseXorGenerator::new(N_BITS))
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        self.0.add_inputs(cuda_inputs);
    }

    /// Write the base trace directly to CUDA backend without CPU round-trip.
    ///
    /// Takes both VBX-8 (self) and VBX-8_B multiplicity buffers. Each becomes a
    /// base trace column via zero-copy reinterpret. Multiplicities stay on GPU
    /// for the interaction trace phase.
    pub fn write_trace_cuda(
        self,
        vbx_8_b: &vbx_8_b::CudaClaimGenerator,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.0.log_size;
        let domain = CanonicCoset::new(log_size).circle_domain();

        // Column 0: VBX-8 multiplicities (zero-copy)
        let m0_clone = self.0.multiplicities.clone();
        let col0 = BaseFieldVec::new(m0_clone.device_ptr, m0_clone.size);
        std::mem::forget(m0_clone);

        // Column 1: VBX-8_B multiplicities (zero-copy)
        let m1_clone = vbx_8_b.multiplicities.clone();
        let col1 = BaseFieldVec::new(m1_clone.device_ptr, m1_clone.size);
        std::mem::forget(m1_clone);

        tree_builder.extend_evals(vec![
            CircleEvaluation::new(domain, col0),
            CircleEvaluation::new(domain, col1),
        ]);

        (
            Claim {},
            CudaInteractionClaimGenerator {
                multiplicities_0: self.0.multiplicities,
                multiplicities_1: vbx_8_b.multiplicities.clone(),
                log_size,
            },
        )
    }
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// CUDA-based interaction claim generator for verify_bitwise_xor_8 (paired).
///
/// Holds both VBX-8 and VBX-8_B multiplicities on GPU. The paired CUDA kernel
/// computes the cross-multiplied logup fraction matching the SIMD formula:
///   numerator = -(denom0 * mults_1 + denom1 * mults_0)
///   denominator = denom0 * denom1
pub struct CudaInteractionClaimGenerator {
    multiplicities_0: Uint32Vec,
    multiplicities_1: Uint32Vec,
    log_size: u32,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_size = 1usize << self.log_size;

        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..4)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size))
            .collect();

        let cuda_claimed_sum: Col<CudaBackend, BaseField> = Col::<CudaBackend, BaseField>::zeros(4);

        let trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        unsafe {
            bindings_airs::verify_bitwise_xor_8_paired_interaction_trace(
                lookup_elements as *const _ as *mut std::os::raw::c_void,
                self.multiplicities_0.device_ptr,
                self.multiplicities_1.device_ptr,
                self.log_size,
                trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        let cs = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([cs[0], cs[1], cs[2], cs[3]]);

        let domain = CanonicCoset::new(self.log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
            })
            .collect();
        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}

impl Deref for CudaClaimGenerator {
    type Target = CudaVerifyBitwiseXorGenerator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CudaClaimGenerator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
