//! SIMD/CUDA trace conversion utilities for the V0 CUDA proving path.
//!
//! The V0 path generates traces using SIMD (via the standard `CairoClaimGenerator`),
//! then converts them to CUDA format for GPU-accelerated commitment and proving.

use stwo::core::fields::m31::M31;
use stwo::core::pcs::TreeSubspan;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use super::utils::TreeBuilder;

/// Convert a SimdBackend CircleEvaluation into a CudaBackend version.
pub fn convert_simd_to_cuda_evaluation(
    eval: CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
) -> CircleEvaluation<CudaBackend, M31, BitReversedOrder> {
    CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(
        eval.domain,
        <CudaBackend as ColumnOps<M31>>::Column::from_iter(eval.values.to_cpu()),
    )
}

/// Convert a CudaBackend CircleEvaluation into a SimdBackend version.
pub fn convert_cuda_to_simd_evaluation(
    eval: CircleEvaluation<CudaBackend, M31, BitReversedOrder>,
) -> CircleEvaluation<SimdBackend, M31, BitReversedOrder> {
    use stwo::prover::backend::simd::column::BaseColumn;
    CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(
        eval.domain,
        BaseColumn::from_iter(eval.values.to_cpu()),
    )
}

/// A trace collector for SIMD traces, allowing later conversion to CUDA format.
///
/// Implements `TreeBuilder<SimdBackend>` so it can be passed to `write_trace()` and
/// `write_interaction_trace()` as a drop-in replacement for stwo's TreeBuilder.
/// The collected traces can then be converted to CUDA format and committed on the GPU.
pub struct SimdTraceCollector {
    traces: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    tree_index: usize,
}

impl SimdTraceCollector {
    pub fn new(tree_index: usize) -> Self {
        Self {
            traces: Vec::new(),
            tree_index,
        }
    }

    /// Convert collected traces to CudaBackend format and extend the CUDA tree builder.
    pub fn extend_cuda_tree_builder(self, tree_builder: &mut impl TreeBuilder<CudaBackend>) {
        tree_builder.extend_evals(
            self.traces
                .into_iter()
                .map(convert_simd_to_cuda_evaluation)
                .collect(),
        );
    }
}

impl TreeBuilder<SimdBackend> for SimdTraceCollector {
    fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ) -> TreeSubspan {
        let col_start = self.traces.len();
        self.traces.extend(columns);
        let col_end = self.traces.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }
}
