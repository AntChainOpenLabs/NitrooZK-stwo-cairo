use std::ops::{Deref, DerefMut};

use cairo_air::components::verify_bitwise_xor_9::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::prover::backend::cuda::CudaBackend;

use super::base::{
    CudaInteractionClaimGeneratorCuda, CudaPackedInputType, CudaVerifyBitwiseXorGenerator,
};
use crate::witness::utils::TreeBuilder;

pub const N_BITS: u32 = 9;

pub struct CudaClaimGenerator(CudaVerifyBitwiseXorGenerator);

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self(CudaVerifyBitwiseXorGenerator::new(N_BITS))
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        self.0.add_inputs(cuda_inputs);
    }

    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, InteractionClaimGeneratorCuda) {
        let interaction_gen = self.0.write_trace_cuda(tree_builder);
        (Claim {}, InteractionClaimGeneratorCuda(interaction_gen))
    }
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct InteractionClaimGeneratorCuda(CudaInteractionClaimGeneratorCuda);

impl InteractionClaimGeneratorCuda {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let claimed_sum = self
            .0
            .write_interaction_trace(tree_builder, lookup_elements);
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
