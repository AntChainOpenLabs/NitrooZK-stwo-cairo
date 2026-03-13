//! CUDA version of pedersen context (window_bits_9) claim generator.
//!
//! Architecture:
//! - pedersen_aggregator_wb9 runs via native CUDA kernel (GPU-resident small pedersen table)
//! - partial_ec_mul_wb9 runs via CUDA directly (311-col trace)
//! - pedersen_points_table_wb9 runs via CUDA directly (1-col trace)
//!
//! The aggregator CUDA kernel writes PEM sub-component inputs directly to GPU arrays,
//! which are then fed to the PEM CUDA generator via set_cuda_inputs().
//! No CPU round-trip for PEM inputs.

use cairo_air::components::{
    partial_ec_mul_window_bits_9, pedersen_aggregator_window_bits_9 as pedersen_aggregator_wb9_air,
    pedersen_aggregator_window_bits_9, pedersen_points_table_window_bits_9,
};
use cairo_air::relations::CommonLookupElements;

/// Local grouped interaction claim type.
pub struct InteractionClaim {
    pub pedersen_aggregator: pedersen_aggregator_window_bits_9::InteractionClaim,
    pub partial_ec_mul: partial_ec_mul_window_bits_9::InteractionClaim,
    pub pedersen_points_table: pedersen_points_table_window_bits_9::InteractionClaim,
}

pub struct PedersenContextWb9InteractionClaim {
    pub claim: Option<InteractionClaim>,
}

use stwo::prover::backend::cuda::CudaBackend;
use tracing::{span, Level};

use crate::witness::components_cuda::range_check::{rc_20, rc_8, rc_9_9};
use crate::witness::components_cuda::{
    memory_id_to_big_cuda, partial_ec_mul_wb9_cuda, pedersen_aggregator_wb9_cuda,
    pedersen_points_table_wb9_cuda,
};
use crate::witness::utils::TreeBuilder;

pub struct PedersenContextWb9CudaClaimGenerator {
    /// Native CUDA generator for pedersen_aggregator_wb9 (receives inputs from pedersen_builtin).
    pub pedersen_aggregator_wb9_cuda: pedersen_aggregator_wb9_cuda::CudaClaimGenerator,
    /// CUDA generator for partial_ec_mul_wb9 trace and interaction trace.
    pub partial_ec_mul_wb9_cuda: partial_ec_mul_wb9_cuda::CudaClaimGenerator,
    /// CUDA generator for pedersen_points_table_wb9 trace and interaction trace.
    pub pedersen_points_table_wb9_cuda: pedersen_points_table_wb9_cuda::CudaClaimGenerator,
}

/// Result of running the aggregator.
pub struct AggregatorResult {
    pub pedersen_aggregator_claim: pedersen_aggregator_wb9_air::Claim,
    pub pedersen_aggregator_interaction_gen:
        pedersen_aggregator_wb9_cuda::CudaInteractionClaimGenerator,
}

impl PedersenContextWb9CudaClaimGenerator {
    pub fn new(
        _preprocessed_trace: std::sync::Arc<
            stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace,
        >,
    ) -> Self {
        Self {
            pedersen_aggregator_wb9_cuda: pedersen_aggregator_wb9_cuda::CudaClaimGenerator::new(),
            partial_ec_mul_wb9_cuda: partial_ec_mul_wb9_cuda::CudaClaimGenerator::new(),
            pedersen_points_table_wb9_cuda: pedersen_points_table_wb9_cuda::CudaClaimGenerator::new(
            ),
        }
    }

    /// Returns true if there are no pedersen inputs.
    pub fn is_empty(&self) -> bool {
        self.pedersen_aggregator_wb9_cuda.is_empty()
    }

    /// Run aggregator via native CUDA kernel, feeding PEM inputs directly to GPU.
    pub fn write_trace_aggregator(
        self,
        cuda_tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        rc_8_state: &rc_8::CudaClaimGenerator,
    ) -> (
        Option<AggregatorResult>,
        partial_ec_mul_wb9_cuda::CudaClaimGenerator,
        pedersen_points_table_wb9_cuda::CudaClaimGenerator,
    ) {
        let span = span!(
            Level::INFO,
            "write pedersen aggregator wb9 trace (native CUDA)"
        )
        .entered();

        if self.pedersen_aggregator_wb9_cuda.is_empty() {
            span.exit();
            return (
                None,
                self.partial_ec_mul_wb9_cuda,
                self.pedersen_points_table_wb9_cuda,
            );
        }

        // Run aggregator trace via native CUDA kernel.
        let _ped_t = std::time::Instant::now();
        let mut pem_cuda = self.partial_ec_mul_wb9_cuda;
        let (aggregator_claim, aggregator_interaction_gen) =
            self.pedersen_aggregator_wb9_cuda.write_trace(
                cuda_tree_builder,
                memory_id_to_big_state,
                rc_8_state,
                &mut pem_cuda,
            );
        eprintln!(
            "[PED-WB9-PROFILE]  aggregator (234 cols, CUDA): {}ms",
            _ped_t.elapsed().as_millis()
        );

        span.exit();

        (
            Some(AggregatorResult {
                pedersen_aggregator_claim: aggregator_claim,
                pedersen_aggregator_interaction_gen: aggregator_interaction_gen,
            }),
            pem_cuda,
            self.pedersen_points_table_wb9_cuda,
        )
    }
}

/// Run CUDA components: partial_ec_mul_wb9 + pedersen_points_table_wb9.
#[allow(clippy::too_many_arguments)]
pub fn write_cuda_components(
    pem_cuda: partial_ec_mul_wb9_cuda::CudaClaimGenerator,
    ppt_cuda: pedersen_points_table_wb9_cuda::CudaClaimGenerator,
    tree_builder: &mut impl TreeBuilder<CudaBackend>,
    rc_9_9_cuda_state: &rc_9_9::CudaClaimGenerator,
    rc_20_cuda_state: &rc_20::CudaClaimGenerator,
) -> CudaComponentsResult {
    let span = span!(Level::INFO, "write pedersen wb9 PEM + PPT trace (CUDA)").entered();

    // Run CUDA PEM kernel (311-col trace + feeds PPT CUDA, rc_9_9, rc_20)
    let _ped_t = std::time::Instant::now();
    let (pem_claim, pem_interaction_gen) =
        pem_cuda.write_trace(tree_builder, &ppt_cuda, rc_9_9_cuda_state, rc_20_cuda_state);
    eprintln!(
        "[PED-WB9-PROFILE]  PEM (311 cols, CUDA):     {}ms",
        _ped_t.elapsed().as_millis()
    );

    // Run CUDA PPT kernel (1-col trace)
    let _ped_t = std::time::Instant::now();
    let (ppt_claim, ppt_interaction_gen) = ppt_cuda.write_trace_cuda(tree_builder);
    eprintln!(
        "[PED-WB9-PROFILE]  PPT (1 col, CUDA):        {}ms",
        _ped_t.elapsed().as_millis()
    );

    span.exit();

    CudaComponentsResult {
        partial_ec_mul_claim: pem_claim,
        partial_ec_mul_interaction_gen: pem_interaction_gen,
        ppt_claim,
        ppt_interaction_gen,
    }
}

/// Result of running PEM + PPT CUDA kernels.
pub struct CudaComponentsResult {
    pub partial_ec_mul_claim: cairo_air::components::partial_ec_mul_window_bits_9::Claim,
    pub partial_ec_mul_interaction_gen: partial_ec_mul_wb9_cuda::CudaInteractionClaimGenerator,
    pub ppt_claim: cairo_air::components::pedersen_points_table_window_bits_9::Claim,
    pub ppt_interaction_gen: pedersen_points_table_wb9_cuda::CudaInteractionClaimGenerator,
}

/// Holds interaction claim generators for the pedersen wb9 context.
pub struct PedersenContextWb9CudaInteractionClaimGenerator {
    pub pedersen_aggregator_interaction_gen:
        Option<pedersen_aggregator_wb9_cuda::CudaInteractionClaimGenerator>,
    pub partial_ec_mul_cuda_interaction_gen:
        Option<partial_ec_mul_wb9_cuda::CudaInteractionClaimGenerator>,
    pub ppt_cuda_interaction_gen:
        Option<pedersen_points_table_wb9_cuda::CudaInteractionClaimGenerator>,
}

impl PedersenContextWb9CudaInteractionClaimGenerator {
    pub fn new_empty() -> Self {
        Self {
            pedersen_aggregator_interaction_gen: None,
            partial_ec_mul_cuda_interaction_gen: None,
            ppt_cuda_interaction_gen: None,
        }
    }

    pub fn new(
        pedersen_aggregator_interaction_gen: pedersen_aggregator_wb9_cuda::CudaInteractionClaimGenerator,
        partial_ec_mul_cuda_interaction_gen: partial_ec_mul_wb9_cuda::CudaInteractionClaimGenerator,
        ppt_cuda_interaction_gen: pedersen_points_table_wb9_cuda::CudaInteractionClaimGenerator,
    ) -> Self {
        Self {
            pedersen_aggregator_interaction_gen: Some(pedersen_aggregator_interaction_gen),
            partial_ec_mul_cuda_interaction_gen: Some(partial_ec_mul_cuda_interaction_gen),
            ppt_cuda_interaction_gen: Some(ppt_cuda_interaction_gen),
        }
    }

    /// Returns true if there are no interaction generators.
    pub fn is_empty(&self) -> bool {
        self.pedersen_aggregator_interaction_gen.is_none()
    }
}

/// Enum to hold pedersen wb9 interaction generators.
pub enum PedersenWb9InteractionMode {
    Cuda(PedersenContextWb9CudaInteractionClaimGenerator),
    Empty,
}

impl PedersenWb9InteractionMode {
    pub fn is_empty(&self) -> bool {
        match self {
            PedersenWb9InteractionMode::Cuda(g) => g.is_empty(),
            PedersenWb9InteractionMode::Empty => true,
        }
    }

    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        common_lookup_elements: &CommonLookupElements,
    ) -> PedersenContextWb9InteractionClaim {
        match self {
            PedersenWb9InteractionMode::Cuda(gen) => {
                let _pit_t = std::time::Instant::now();
                let agg_interaction = gen
                    .pedersen_aggregator_interaction_gen
                    .unwrap()
                    .write_interaction_trace(tree_builder, common_lookup_elements);
                eprintln!(
                    "[PED-WB9-IT-PROF]  aggregator interaction:   {}ms",
                    _pit_t.elapsed().as_millis()
                );

                let _pit_t = std::time::Instant::now();
                let pem_interaction = gen
                    .partial_ec_mul_cuda_interaction_gen
                    .unwrap()
                    .write_interaction_trace(tree_builder, common_lookup_elements);
                eprintln!(
                    "[PED-WB9-IT-PROF]  PEM interaction:          {}ms",
                    _pit_t.elapsed().as_millis()
                );

                let _pit_t = std::time::Instant::now();
                let ppt_interaction = gen
                    .ppt_cuda_interaction_gen
                    .unwrap()
                    .write_interaction_trace(tree_builder, common_lookup_elements);
                eprintln!(
                    "[PED-WB9-IT-PROF]  PPT interaction:          {}ms",
                    _pit_t.elapsed().as_millis()
                );

                PedersenContextWb9InteractionClaim {
                    claim: Some(InteractionClaim {
                        pedersen_aggregator: agg_interaction,
                        partial_ec_mul: pem_interaction,
                        pedersen_points_table: ppt_interaction,
                    }),
                }
            }
            PedersenWb9InteractionMode::Empty => PedersenContextWb9InteractionClaim { claim: None },
        }
    }
}
