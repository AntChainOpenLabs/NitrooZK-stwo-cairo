use cairo_air::components::{
    range_check_11, range_check_12, range_check_18, range_check_20, range_check_3_3_3_3_3,
    range_check_3_6_6_3, range_check_4_3, range_check_4_4, range_check_4_4_4_4, range_check_6,
    range_check_7_2_5, range_check_8, range_check_9_9,
};
use cairo_air::relations::CommonLookupElements;
use stwo::prover::backend::cuda::CudaBackend;

use super::components_cuda::range_check::{
    rc_11, rc_12, rc_18, rc_20, rc_3_3_3_3_3, rc_3_6_6_3, rc_4_3, rc_4_4, rc_4_4_4_4, rc_6,
    rc_7_2_5, rc_8, rc_9_9,
};
use crate::witness::utils::TreeBuilder;

/// Local grouped claim type (mirrors legacy `cairo_air::range_checks_air::RangeChecksClaim`).
pub struct RangeChecksClaim {
    pub rc_6: range_check_6::Claim,
    pub rc_8: range_check_8::Claim,
    pub rc_11: range_check_11::Claim,
    pub rc_12: range_check_12::Claim,
    pub rc_18: range_check_18::Claim,
    pub rc_20: range_check_20::Claim,
    pub rc_4_3: range_check_4_3::Claim,
    pub rc_4_4: range_check_4_4::Claim,
    pub rc_9_9: range_check_9_9::Claim,
    pub rc_7_2_5: range_check_7_2_5::Claim,
    pub rc_3_6_6_3: range_check_3_6_6_3::Claim,
    pub rc_4_4_4_4: range_check_4_4_4_4::Claim,
    pub rc_3_3_3_3_3: range_check_3_3_3_3_3::Claim,
}

pub struct RangeChecksInteractionClaim {
    pub rc_6: range_check_6::InteractionClaim,
    pub rc_8: range_check_8::InteractionClaim,
    pub rc_11: range_check_11::InteractionClaim,
    pub rc_12: range_check_12::InteractionClaim,
    pub rc_18: range_check_18::InteractionClaim,
    pub rc_20: range_check_20::InteractionClaim,
    pub rc_4_3: range_check_4_3::InteractionClaim,
    pub rc_4_4: range_check_4_4::InteractionClaim,
    pub rc_9_9: range_check_9_9::InteractionClaim,
    pub rc_7_2_5: range_check_7_2_5::InteractionClaim,
    pub rc_3_6_6_3: range_check_3_6_6_3::InteractionClaim,
    pub rc_4_4_4_4: range_check_4_4_4_4::InteractionClaim,
    pub rc_3_3_3_3_3: range_check_3_3_3_3_3::InteractionClaim,
}

/// CUDA range checks claim generator.
///
/// Holds 13 CUDA range check generators (rc_5_4 removed in v1.1.0).
pub struct RangeChecksCudaClaimGenerator {
    pub rc_6_trace_generator: rc_6::CudaClaimGenerator,
    pub rc_8_trace_generator: rc_8::CudaClaimGenerator,
    pub rc_11_trace_generator: rc_11::CudaClaimGenerator,
    pub rc_12_trace_generator: rc_12::CudaClaimGenerator,
    pub rc_18_trace_generator: rc_18::CudaClaimGenerator,
    pub rc_20_trace_generator: rc_20::CudaClaimGenerator,
    pub rc_4_3_trace_generator: rc_4_3::CudaClaimGenerator,
    pub rc_4_4_trace_generator: rc_4_4::CudaClaimGenerator,
    pub rc_9_9_trace_generator: rc_9_9::CudaClaimGenerator,
    pub rc_7_2_5_trace_generator: rc_7_2_5::CudaClaimGenerator,
    pub rc_3_6_6_3_trace_generator: rc_3_6_6_3::CudaClaimGenerator,
    pub rc_4_4_4_4_trace_generator: rc_4_4_4_4::CudaClaimGenerator,
    pub rc_3_3_3_3_3_trace_generator: rc_3_3_3_3_3::CudaClaimGenerator,
}

impl RangeChecksCudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            rc_6_trace_generator: rc_6::CudaClaimGenerator::new(),
            rc_8_trace_generator: rc_8::CudaClaimGenerator::new(),
            rc_11_trace_generator: rc_11::CudaClaimGenerator::new(),
            rc_12_trace_generator: rc_12::CudaClaimGenerator::new(),
            rc_18_trace_generator: rc_18::CudaClaimGenerator::new(),
            rc_20_trace_generator: rc_20::CudaClaimGenerator::new(),
            rc_4_3_trace_generator: rc_4_3::CudaClaimGenerator::new(),
            rc_4_4_trace_generator: rc_4_4::CudaClaimGenerator::new(),
            rc_9_9_trace_generator: rc_9_9::CudaClaimGenerator::new(),
            rc_7_2_5_trace_generator: rc_7_2_5::CudaClaimGenerator::new(),
            rc_3_6_6_3_trace_generator: rc_3_6_6_3::CudaClaimGenerator::new(),
            rc_4_4_4_4_trace_generator: rc_4_4_4_4::CudaClaimGenerator::new(),
            rc_3_3_3_3_3_trace_generator: rc_3_3_3_3_3::CudaClaimGenerator::new(),
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (RangeChecksClaim, RangeChecksCudaInteractionClaimGenerator) {
        let (rc_6_claim, rc_6_interaction_gen) =
            self.rc_6_trace_generator.write_trace_cuda(tree_builder);
        let (rc_8_claim, rc_8_interaction_gen) =
            self.rc_8_trace_generator.write_trace_cuda(tree_builder);
        let (rc_11_claim, rc_11_interaction_gen) =
            self.rc_11_trace_generator.write_trace_cuda(tree_builder);
        let (rc_12_claim, rc_12_interaction_gen) =
            self.rc_12_trace_generator.write_trace_cuda(tree_builder);
        let (rc_18_claim, rc_18_interaction_gen) =
            self.rc_18_trace_generator.write_trace_cuda(tree_builder);
        let (rc_20_claim, rc_20_interaction_gen) =
            self.rc_20_trace_generator.write_trace_cuda(tree_builder);
        let (rc_4_3_claim, rc_4_3_interaction_gen) =
            self.rc_4_3_trace_generator.write_trace_cuda(tree_builder);
        let (rc_4_4_claim, rc_4_4_interaction_gen) =
            self.rc_4_4_trace_generator.write_trace_cuda(tree_builder);
        let (rc_9_9_claim, rc_9_9_interaction_gen) =
            self.rc_9_9_trace_generator.write_trace_cuda(tree_builder);
        let (rc_7_2_5_claim, rc_7_2_5_interaction_gen) =
            self.rc_7_2_5_trace_generator.write_trace_cuda(tree_builder);
        let (rc_3_6_6_3_claim, rc_3_6_6_3_interaction_gen) = self
            .rc_3_6_6_3_trace_generator
            .write_trace_cuda(tree_builder);
        let (rc_4_4_4_4_claim, rc_4_4_4_4_interaction_gen) = self
            .rc_4_4_4_4_trace_generator
            .write_trace_cuda(tree_builder);
        let (rc_3_3_3_3_3_claim, rc_3_3_3_3_3_interaction_gen) = self
            .rc_3_3_3_3_3_trace_generator
            .write_trace_cuda(tree_builder);

        (
            RangeChecksClaim {
                rc_6: rc_6_claim,
                rc_8: rc_8_claim,
                rc_11: rc_11_claim,
                rc_12: rc_12_claim,
                rc_18: rc_18_claim,
                rc_20: rc_20_claim,
                rc_4_3: rc_4_3_claim,
                rc_4_4: rc_4_4_claim,
                rc_9_9: rc_9_9_claim,
                rc_7_2_5: rc_7_2_5_claim,
                rc_3_6_6_3: rc_3_6_6_3_claim,
                rc_4_4_4_4: rc_4_4_4_4_claim,
                rc_3_3_3_3_3: rc_3_3_3_3_3_claim,
            },
            RangeChecksCudaInteractionClaimGenerator {
                rc_6_interaction_gen,
                rc_8_interaction_gen,
                rc_11_interaction_gen,
                rc_12_interaction_gen,
                rc_18_interaction_gen,
                rc_20_interaction_gen,
                rc_4_3_interaction_gen,
                rc_4_4_interaction_gen,
                rc_9_9_interaction_gen,
                rc_7_2_5_interaction_gen,
                rc_3_6_6_3_interaction_gen,
                rc_4_4_4_4_interaction_gen,
                rc_3_3_3_3_3_interaction_gen,
            },
        )
    }
}

pub struct RangeChecksCudaInteractionClaimGenerator {
    rc_6_interaction_gen: rc_6::InteractionClaimGeneratorCuda,
    rc_8_interaction_gen: rc_8::InteractionClaimGeneratorCuda,
    rc_11_interaction_gen: rc_11::InteractionClaimGeneratorCuda,
    rc_12_interaction_gen: rc_12::InteractionClaimGeneratorCuda,
    rc_18_interaction_gen: rc_18::InteractionClaimGeneratorCuda,
    rc_20_interaction_gen: rc_20::InteractionClaimGeneratorCuda,
    rc_4_3_interaction_gen: rc_4_3::InteractionClaimGeneratorCuda,
    rc_4_4_interaction_gen: rc_4_4::InteractionClaimGeneratorCuda,
    rc_9_9_interaction_gen: rc_9_9::InteractionClaimGeneratorCuda,
    rc_7_2_5_interaction_gen: rc_7_2_5::InteractionClaimGeneratorCuda,
    rc_3_6_6_3_interaction_gen: rc_3_6_6_3::InteractionClaimGeneratorCuda,
    rc_4_4_4_4_interaction_gen: rc_4_4_4_4::InteractionClaimGeneratorCuda,
    rc_3_3_3_3_3_interaction_gen: rc_3_3_3_3_3::InteractionClaimGeneratorCuda,
}

impl RangeChecksCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        common_lookup_elements: &CommonLookupElements,
    ) -> RangeChecksInteractionClaim {
        let rc_6_interaction_claim = self
            .rc_6_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_8_interaction_claim = self
            .rc_8_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_11_interaction_claim = self
            .rc_11_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_12_interaction_claim = self
            .rc_12_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_18_interaction_claim = self
            .rc_18_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_20_interaction_claim = self
            .rc_20_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_4_3_interaction_claim = self
            .rc_4_3_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_4_4_interaction_claim = self
            .rc_4_4_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_9_9_interaction_claim = self
            .rc_9_9_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_7_2_5_interaction_claim = self
            .rc_7_2_5_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_3_6_6_3_interaction_claim = self
            .rc_3_6_6_3_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_4_4_4_4_interaction_claim = self
            .rc_4_4_4_4_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);
        let rc_3_3_3_3_3_interaction_claim = self
            .rc_3_3_3_3_3_interaction_gen
            .write_interaction_trace(tree_builder, common_lookup_elements);

        RangeChecksInteractionClaim {
            rc_6: rc_6_interaction_claim,
            rc_8: rc_8_interaction_claim,
            rc_11: rc_11_interaction_claim,
            rc_12: rc_12_interaction_claim,
            rc_18: rc_18_interaction_claim,
            rc_20: rc_20_interaction_claim,
            rc_4_3: rc_4_3_interaction_claim,
            rc_4_4: rc_4_4_interaction_claim,
            rc_9_9: rc_9_9_interaction_claim,
            rc_7_2_5: rc_7_2_5_interaction_claim,
            rc_3_6_6_3: rc_3_6_6_3_interaction_claim,
            rc_4_4_4_4: rc_4_4_4_4_interaction_claim,
            rc_3_3_3_3_3: rc_3_3_3_3_3_interaction_claim,
        }
    }
}
