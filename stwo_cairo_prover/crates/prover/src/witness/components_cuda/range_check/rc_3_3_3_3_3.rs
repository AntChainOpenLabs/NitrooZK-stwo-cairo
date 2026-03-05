use std::ops::{Deref, DerefMut};

use cairo_air::components::range_check_3_3_3_3_3::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

use super::base::{CudaInteractionClaimGeneratorCuda, CudaRangeCheckGenerator};
use crate::witness::utils::TreeBuilder;

pub type CudaPackedInputType = [BaseFieldVec; 5];
pub const RANGES: [u32; 5] = [3, 3, 3, 3, 3];

const RELATION_CONSTANT: M31 = M31(502259093);

pub struct CudaClaimGenerator(CudaRangeCheckGenerator<5>);

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self(CudaRangeCheckGenerator::new(RANGES))
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        for input in cuda_inputs {
            let n_rows = input[0].size;
            self.0.add_inputs_internal(input, n_rows);
        }
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

pub struct InteractionClaimGeneratorCuda(CudaInteractionClaimGeneratorCuda<5>);

impl InteractionClaimGeneratorCuda {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let inner = unsafe {
            &*(lookup_elements as *const CommonLookupElements
                as *const stwo_constraint_framework::logup::LookupElements<128>)
        };
        let claimed_sum = self
            .0
            .write_interaction_trace(tree_builder, inner, RELATION_CONSTANT);
        InteractionClaim { claimed_sum }
    }
}

impl Deref for CudaClaimGenerator {
    type Target = CudaRangeCheckGenerator<5>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CudaClaimGenerator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
