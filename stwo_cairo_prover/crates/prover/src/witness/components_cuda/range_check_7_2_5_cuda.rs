//! CUDA implementation for range_check_7_2_5 component.
//!
//! This module provides a CUDA-accelerated claim generator for 7-2-5 bit range checks.
//! This is a multi-segment range check with 3 segments totaling 14 bits.

use std::ops::{Deref, DerefMut};

use cairo_air::components::range_check_7_2_5::{Claim, InteractionClaim};
use cairo_air::relations;
use stwo::prover::backend::simd::SimdBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

use super::range_check_cuda::{CudaInteractionClaimGenerator, CudaRangeCheckGenerator};
use crate::witness::utils::TreeBuilder;

/// Input type for CUDA range check: three values (7-bit, 2-bit, 5-bit)
pub type CudaPackedInputType = [BaseFieldVec; 3];

/// Range configuration: 7-bit, 2-bit, 5-bit segments
pub const RANGES: [u32; 3] = [7, 2, 5];

/// CUDA claim generator for rc_7_2_5 (newtype wrapper)
pub struct CudaClaimGenerator(CudaRangeCheckGenerator<3>);

impl CudaClaimGenerator {
    /// Create a new CUDA claim generator for rc_7_2_5
    pub fn new_rc_7_2_5() -> Self {
        Self(CudaRangeCheckGenerator::new(RANGES))
    }

    /// Add inputs from CUDA vectors
    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        if cuda_inputs.is_empty() {
            return;
        }
        // Process each input batch
        for input in cuda_inputs {
            let n_rows = input[0].size;
            self.0.add_inputs_internal(input, n_rows);
        }
    }

    /// Write the trace for this range check component.
    pub fn write_trace(
        &self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
    ) -> (Claim, InteractionClaimGenerator) {
        let interaction_gen = self.0.write_trace(tree_builder);
        (Claim {}, InteractionClaimGenerator(interaction_gen))
    }
}

/// Interaction claim generator for rc_7_2_5
pub struct InteractionClaimGenerator(CudaInteractionClaimGenerator<3>);

impl InteractionClaimGenerator {
    pub fn write_interaction_trace(
        &self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        lookup_elements: &relations::RangeCheck_7_2_5,
    ) -> InteractionClaim {
        let claimed_sum = self.0.write_interaction_trace(tree_builder, lookup_elements);
        InteractionClaim { claimed_sum }
    }
}

impl Deref for CudaClaimGenerator {
    type Target = CudaRangeCheckGenerator<3>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CudaClaimGenerator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
