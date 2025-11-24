#![allow(unused_parens)]
#![allow(dead_code)]

use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

pub type CudaPackedInputType = [BaseFieldVec; 1];

pub struct CudaClaimGenerator {}

impl CudaClaimGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {}
    }

    pub fn add_cuda_inputs(&self, _cuda_inputs: &[CudaPackedInputType]) {
        // Stub implementation - range check handled by CPU or inline
    }
}
