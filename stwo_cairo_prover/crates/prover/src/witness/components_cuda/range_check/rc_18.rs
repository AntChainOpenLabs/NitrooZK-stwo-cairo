use cairo_air::components::range_check_18::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

use super::base::{CudaMultiRelationInteractionGen, CudaMultiRelationRangeCheckGenerator};
use crate::witness::utils::TreeBuilder;

pub type CudaPackedInputType = [BaseFieldVec; 1];
pub const RANGES: [u32; 1] = [18];
pub const N_RELATIONS: usize = 2;

/// Relation constants from the AIR definition (range_check_18.rs).
const RELATION_CONSTANTS: [M31; N_RELATIONS] = [M31(1109051422), M31(1424798916)];

pub struct CudaClaimGenerator(CudaMultiRelationRangeCheckGenerator<1, N_RELATIONS>);

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self(CudaMultiRelationRangeCheckGenerator::new(RANGES))
    }

    /// Adds inputs for a specific relation index.
    pub fn add_cuda_inputs_for_relation(
        &self,
        cuda_inputs: &[CudaPackedInputType],
        relation_index: usize,
    ) {
        for input in cuda_inputs {
            let n_rows = input[0].size;
            self.0
                .add_inputs_for_relation(input, n_rows, relation_index);
        }
    }

    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, InteractionClaimGeneratorCuda) {
        let interaction_gen = self.0.write_trace_cuda(tree_builder, RELATION_CONSTANTS);
        (Claim {}, InteractionClaimGeneratorCuda(interaction_gen))
    }

    pub fn log_size(&self) -> u32 {
        self.0.log_size()
    }

    pub fn merge_simd_multiplicities(&mut self, simd_mults: &[Vec<u32>]) {
        self.0.merge_simd_multiplicities(simd_mults);
    }

    /// Returns the device pointer for the multiplicity vector of the given relation.
    pub fn multiplicities_ptr(&self, relation_index: usize) -> *const u32 {
        self.0.multiplicities[relation_index].device_ptr
    }
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct InteractionClaimGeneratorCuda(CudaMultiRelationInteractionGen<1, N_RELATIONS>);

impl InteractionClaimGeneratorCuda {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        // CommonLookupElements is a newtype around LookupElements<128>.
        // Transmute to access the inner LookupElements<128>.
        let inner = unsafe {
            &*(lookup_elements as *const CommonLookupElements
                as *const stwo_constraint_framework::logup::LookupElements<128>)
        };
        let claimed_sum = self.0.write_interaction_trace(tree_builder, inner);
        InteractionClaim { claimed_sum }
    }
}
