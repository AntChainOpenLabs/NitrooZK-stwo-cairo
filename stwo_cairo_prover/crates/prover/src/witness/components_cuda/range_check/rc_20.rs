use cairo_air::components::range_check_20::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Column;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

use super::base::{CudaMultiRelationInteractionGen, CudaMultiRelationRangeCheckGenerator};
use crate::witness::utils::TreeBuilder;

pub type CudaPackedInputType = [BaseFieldVec; 1];
pub const RANGES: [u32; 1] = [20];
pub const N_RELATIONS: usize = 8;

/// Relation constants from the AIR definition (range_check_20.rs).
const RELATION_CONSTANTS: [M31; N_RELATIONS] = [
    M31(1410849886),
    M31(514232941),
    M31(531010560),
    M31(480677703),
    M31(497455322),
    M31(447122465),
    M31(463900084),
    M31(682009131),
];

pub struct CudaClaimGenerator(pub CudaMultiRelationRangeCheckGenerator<1, N_RELATIONS>);

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

    /// Adds inputs for a specific relation with an offset correction applied on CPU.
    ///
    /// Downloads GPU data, applies `offset` to each value, re-uploads, then calls
    /// `add_inputs_for_relation`. Used by mul_opcode and generic_opcode which store
    /// range_check_19 values with a different base offset than range_check_20 expects.
    pub fn add_cuda_inputs_for_relation_with_offset(
        &self,
        cuda_inputs: &[[BaseFieldVec; 1]],
        relation_index: usize,
        offset: M31,
    ) {
        for input in cuda_inputs {
            let cpu_vals: Vec<M31> = input[0].to_cpu();
            let corrected: Vec<M31> = cpu_vals.into_iter().map(|v| v + offset).collect();
            let corrected_gpu = BaseFieldVec::from_vec(corrected);
            let n_rows = corrected_gpu.size;
            self.0
                .add_inputs_for_relation(&[corrected_gpu], n_rows, relation_index);
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
        let inner = unsafe {
            &*(lookup_elements as *const CommonLookupElements
                as *const stwo_constraint_framework::logup::LookupElements<128>)
        };
        let claimed_sum = self.0.write_interaction_trace(tree_builder, inner);
        InteractionClaim { claimed_sum }
    }
}
