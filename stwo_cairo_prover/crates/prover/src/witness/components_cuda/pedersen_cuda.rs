//! CUDA version of pedersen context claim generator.
//!
//! Architecture:
//! - pedersen_aggregator runs via native CUDA kernel (GPU-resident pedersen table)
//! - partial_ec_mul runs via CUDA directly (297-col trace)
//! - pedersen_points_table runs via CUDA directly (1-col trace)
//!
//! The aggregator CUDA kernel writes PEM sub-component inputs directly to GPU arrays,
//! which are then fed to the PEM CUDA generator via set_cuda_inputs().
//! No CPU round-trip for PEM inputs.

use cairo_air::components::pedersen_aggregator_window_bits_18 as pedersen_aggregator_air;
use cairo_air::components::{
    partial_ec_mul_window_bits_18, pedersen_aggregator_window_bits_18,
    pedersen_points_table_window_bits_18,
};
use cairo_air::relations::CommonLookupElements;

/// Local grouped interaction claim type (mirrors legacy `cairo_air::pedersen::air::InteractionClaim`).
pub struct InteractionClaim {
    pub pedersen_aggregator: pedersen_aggregator_window_bits_18::InteractionClaim,
    pub partial_ec_mul: partial_ec_mul_window_bits_18::InteractionClaim,
    pub pedersen_points_table: pedersen_points_table_window_bits_18::InteractionClaim,
}

pub struct PedersenContextInteractionClaim {
    pub claim: Option<InteractionClaim>,
}
use stwo::prover::backend::cuda::CudaBackend;
use tracing::{span, Level};

use crate::witness::components_cuda::range_check::{rc_20, rc_8, rc_9_9};
use crate::witness::components_cuda::{
    memory_id_to_big_cuda, partial_ec_mul_cuda, pedersen_aggregator_cuda,
    pedersen_points_table_cuda,
};
use crate::witness::utils::TreeBuilder;

pub struct PedersenContextCudaClaimGenerator {
    /// Native CUDA generator for pedersen_aggregator (receives inputs from pedersen_builtin).
    pub pedersen_aggregator_cuda: pedersen_aggregator_cuda::CudaClaimGenerator,
    /// CUDA generator for partial_ec_mul trace and interaction trace.
    pub partial_ec_mul_cuda: partial_ec_mul_cuda::CudaClaimGenerator,
    /// CUDA generator for pedersen_points_table trace and interaction trace.
    pub pedersen_points_table_cuda: pedersen_points_table_cuda::CudaClaimGenerator,
}

/// Result of running the aggregator.
/// Contains the aggregator's claim and CUDA interaction generator.
pub struct AggregatorResult {
    pub pedersen_aggregator_claim: pedersen_aggregator_air::Claim,
    pub pedersen_aggregator_interaction_gen:
        pedersen_aggregator_cuda::CudaInteractionClaimGenerator,
}

impl PedersenContextCudaClaimGenerator {
    pub fn new(
        _preprocessed_trace: std::sync::Arc<
            stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace,
        >,
    ) -> Self {
        Self {
            pedersen_aggregator_cuda: pedersen_aggregator_cuda::CudaClaimGenerator::new(),
            partial_ec_mul_cuda: partial_ec_mul_cuda::CudaClaimGenerator::new(),
            pedersen_points_table_cuda: pedersen_points_table_cuda::CudaClaimGenerator::new(),
        }
    }

    /// Returns true if there are no pedersen inputs.
    pub fn is_empty(&self) -> bool {
        self.pedersen_aggregator_cuda.is_empty()
    }

    /// Run aggregator via native CUDA kernel, feeding PEM inputs directly to GPU.
    ///
    /// The aggregator CUDA kernel generates 206 trace columns and writes all
    /// sub-component inputs (memory_id_to_big, range_check_8, partial_ec_mul)
    /// directly to GPU arrays. PEM inputs are fed to the PEM CUDA generator
    /// via set_cuda_inputs() — no CPU round-trip.
    ///
    /// After this call, the caller should call `write_cuda_components()`.
    pub fn write_trace_aggregator(
        self,
        cuda_tree_builder: &mut impl crate::witness::utils::TreeBuilder<CudaBackend>,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        rc_8_state: &rc_8::CudaClaimGenerator,
    ) -> (
        Option<AggregatorResult>,
        partial_ec_mul_cuda::CudaClaimGenerator,
        pedersen_points_table_cuda::CudaClaimGenerator,
    ) {
        let span = span!(Level::INFO, "write pedersen aggregator trace (native CUDA)").entered();

        if self.pedersen_aggregator_cuda.is_empty() {
            span.exit();
            return (
                None,
                self.partial_ec_mul_cuda,
                self.pedersen_points_table_cuda,
            );
        }

        // Run aggregator trace via native CUDA kernel.
        // PEM inputs are fed directly from GPU arrays to the PEM CUDA generator.
        let _ped_t = std::time::Instant::now();
        let mut pem_cuda = self.partial_ec_mul_cuda;
        let (aggregator_claim, aggregator_interaction_gen) =
            self.pedersen_aggregator_cuda.write_trace(
                cuda_tree_builder,
                memory_id_to_big_state,
                rc_8_state,
                &mut pem_cuda,
            );
        eprintln!(
            "[PED-PROFILE]  aggregator (206 cols, CUDA): {}ms",
            _ped_t.elapsed().as_millis()
        );

        span.exit();

        (
            Some(AggregatorResult {
                pedersen_aggregator_claim: aggregator_claim,
                pedersen_aggregator_interaction_gen: aggregator_interaction_gen,
            }),
            pem_cuda,
            self.pedersen_points_table_cuda,
        )
    }
}

/// Run CUDA components: partial_ec_mul + pedersen_points_table.
///
/// PEM inputs were already fed by the aggregator kernel. This just runs the
/// PEM and PPT CUDA kernels.
///
/// Returns claims and interaction generators for both components.
#[allow(clippy::too_many_arguments)]
pub fn write_cuda_components(
    pem_cuda: partial_ec_mul_cuda::CudaClaimGenerator,
    ppt_cuda: pedersen_points_table_cuda::CudaClaimGenerator,
    tree_builder: &mut impl TreeBuilder<CudaBackend>,
    rc_9_9_cuda_state: &rc_9_9::CudaClaimGenerator,
    rc_20_cuda_state: &rc_20::CudaClaimGenerator,
) -> CudaComponentsResult {
    let span = span!(Level::INFO, "write pedersen PEM + PPT trace (CUDA)").entered();

    // Run CUDA PEM kernel (297-col trace + feeds PPT CUDA, rc_9_9, rc_20)
    let _ped_t = std::time::Instant::now();
    let (pem_claim, pem_interaction_gen) =
        pem_cuda.write_trace(tree_builder, &ppt_cuda, rc_9_9_cuda_state, rc_20_cuda_state);
    eprintln!(
        "[PED-PROFILE]  PEM (297 cols, CUDA):     {}ms",
        _ped_t.elapsed().as_millis()
    );

    // Run CUDA PPT kernel (1-col trace)
    let _ped_t = std::time::Instant::now();
    let (ppt_claim, ppt_interaction_gen) = ppt_cuda.write_trace_cuda(tree_builder);
    eprintln!(
        "[PED-PROFILE]  PPT (1 col, CUDA):        {}ms",
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
    pub partial_ec_mul_claim: cairo_air::components::partial_ec_mul_window_bits_18::Claim,
    pub partial_ec_mul_interaction_gen: partial_ec_mul_cuda::CudaInteractionClaimGenerator,
    pub ppt_claim: cairo_air::components::pedersen_points_table_window_bits_18::Claim,
    pub ppt_interaction_gen: pedersen_points_table_cuda::CudaInteractionClaimGenerator,
}

/// Holds interaction claim generators for the pedersen context.
///
/// All three sub-components use CUDA generators.
pub struct PedersenContextCudaInteractionClaimGenerator {
    pub pedersen_aggregator_interaction_gen:
        Option<pedersen_aggregator_cuda::CudaInteractionClaimGenerator>,
    pub partial_ec_mul_cuda_interaction_gen:
        Option<partial_ec_mul_cuda::CudaInteractionClaimGenerator>,
    pub ppt_cuda_interaction_gen: Option<pedersen_points_table_cuda::CudaInteractionClaimGenerator>,
}

impl PedersenContextCudaInteractionClaimGenerator {
    pub fn new_empty() -> Self {
        Self {
            pedersen_aggregator_interaction_gen: None,
            partial_ec_mul_cuda_interaction_gen: None,
            ppt_cuda_interaction_gen: None,
        }
    }

    pub fn new(
        pedersen_aggregator_interaction_gen: pedersen_aggregator_cuda::CudaInteractionClaimGenerator,
        partial_ec_mul_cuda_interaction_gen: partial_ec_mul_cuda::CudaInteractionClaimGenerator,
        ppt_cuda_interaction_gen: pedersen_points_table_cuda::CudaInteractionClaimGenerator,
    ) -> Self {
        Self {
            pedersen_aggregator_interaction_gen: Some(pedersen_aggregator_interaction_gen),
            partial_ec_mul_cuda_interaction_gen: Some(partial_ec_mul_cuda_interaction_gen),
            ppt_cuda_interaction_gen: Some(ppt_cuda_interaction_gen),
        }
    }

    /// Returns true if there are no interaction generators (empty pedersen context).
    pub fn is_empty(&self) -> bool {
        self.pedersen_aggregator_interaction_gen.is_none()
    }
}

/// Enum to hold pedersen interaction generators.
pub enum PedersenInteractionMode {
    Cuda(PedersenContextCudaInteractionClaimGenerator),
    Empty,
}

impl PedersenInteractionMode {
    pub fn is_empty(&self) -> bool {
        match self {
            PedersenInteractionMode::Cuda(g) => g.is_empty(),
            PedersenInteractionMode::Empty => true,
        }
    }

    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        common_lookup_elements: &CommonLookupElements,
    ) -> PedersenContextInteractionClaim {
        match self {
            PedersenInteractionMode::Cuda(gen) => {
                // All three use CUDA interaction generators
                let _pit_t = std::time::Instant::now();
                let agg_interaction = gen
                    .pedersen_aggregator_interaction_gen
                    .unwrap()
                    .write_interaction_trace(tree_builder, common_lookup_elements);
                eprintln!(
                    "[PED-IT-PROF]  aggregator interaction:   {}ms",
                    _pit_t.elapsed().as_millis()
                );

                let _pit_t = std::time::Instant::now();
                let pem_interaction = gen
                    .partial_ec_mul_cuda_interaction_gen
                    .unwrap()
                    .write_interaction_trace(tree_builder, common_lookup_elements);
                eprintln!(
                    "[PED-IT-PROF]  PEM interaction:          {}ms",
                    _pit_t.elapsed().as_millis()
                );

                let _pit_t = std::time::Instant::now();
                let ppt_interaction = gen
                    .ppt_cuda_interaction_gen
                    .unwrap()
                    .write_interaction_trace(tree_builder, common_lookup_elements);
                eprintln!(
                    "[PED-IT-PROF]  PPT interaction:          {}ms",
                    _pit_t.elapsed().as_millis()
                );

                PedersenContextInteractionClaim {
                    claim: Some(InteractionClaim {
                        pedersen_aggregator: agg_interaction,
                        partial_ec_mul: pem_interaction,
                        pedersen_points_table: ppt_interaction,
                    }),
                }
            }
            PedersenInteractionMode::Empty => PedersenContextInteractionClaim { claim: None },
        }
    }
}
