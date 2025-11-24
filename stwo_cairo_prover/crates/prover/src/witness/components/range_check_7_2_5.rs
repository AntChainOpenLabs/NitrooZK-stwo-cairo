#![allow(dead_code)]

//! Minimal stub for range_check_7_2_5 witness component

use cairo_air::components::range_check_7_2_5::{Claim, InteractionClaim};
use stwo::core::fields::qm31::QM31;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::SimdBackend;

use crate::witness::utils::TreeBuilder;

pub type PackedInputType = [PackedM31; 3];

pub struct ClaimGenerator {}

impl ClaimGenerator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn add_packed_inputs(&self, _inputs: &[PackedInputType]) {
        // Minimal stub - only used by blake_round for collecting inputs
    }

    pub fn add_packed_m31(&self, _inputs: &[PackedM31; 3]) {
        // Minimal stub - used by range_check_vector components
    }

    pub fn write_trace(
        self,
        _tree_builder: &mut impl TreeBuilder<SimdBackend>,
    ) -> (Claim, InteractionClaimGenerator) {
        // Stub: Returns minimal valid claim
        (
            Claim { log_size: 14 },
            InteractionClaimGenerator {},
        )
    }
}

// Stub for InteractionClaimGenerator - not actually used in blake_round flow
pub struct InteractionClaimGenerator {}

impl InteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        _tree_builder: &mut impl TreeBuilder<SimdBackend>,
        _lookup_elements: &cairo_air::relations::RangeCheck_7_2_5,
    ) -> InteractionClaim {
        // Stub: Returns zero claimed sum
        InteractionClaim {
            claimed_sum: QM31::from_u32_unchecked(0, 0, 0, 0),
        }
    }
}
