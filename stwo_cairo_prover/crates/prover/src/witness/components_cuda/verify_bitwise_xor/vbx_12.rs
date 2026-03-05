use cairo_air::components::verify_bitwise_xor_12::{
    Claim, InteractionClaim, LOG_SIZE, N_MULT_COLUMNS,
};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;

pub use super::base::CudaPackedInputType;
use crate::witness::utils::TreeBuilder;

/// CUDA multiplicity accumulator for verify_bitwise_xor_12.
///
/// VBX 12 uses a multi-column layout: N_MULT_COLUMNS (16) multiplicity columns,
/// each of size 2^LOG_SIZE (2^20). This differs from VBX 4/7/8/9 which use a
/// single column.
pub struct CudaClaimGenerator {
    pub multiplicities: [Uint32Vec; N_MULT_COLUMNS],
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            multiplicities: std::array::from_fn(|_| Uint32Vec::new_zeroes(1 << LOG_SIZE)),
        }
    }

    /// Feed sub-component inputs from CUDA kernels (e.g., blake_g_cuda).
    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        if cuda_inputs.is_empty() {
            return;
        }

        for batch in cuda_inputs {
            let inputs_vec: Vec<*const u32> = batch.iter().map(|x| x.device_ptr).collect_vec();
            let n_rows = batch[0].size;

            let mults_vec: Vec<*const u32> = self
                .multiplicities
                .iter()
                .map(|m| m.device_ptr)
                .collect_vec();

            unsafe {
                bindings_airs::verify_bitwise_xor_12_mults_init(
                    inputs_vec.as_ptr(),
                    1,
                    n_rows as u32,
                    mults_vec.as_ptr() as *const *const u32,
                    N_MULT_COLUMNS as u32,
                    LOG_SIZE,
                );
            }
        }
    }

    /// Write the base trace directly to CUDA backend without CPU round-trip.
    ///
    /// Each of the 16 GPU multiplicity buffers becomes a base trace column via
    /// zero-copy reinterpret (clone Uint32Vec → wrap as BaseFieldVec → forget clone).
    /// Multiplicities stay on GPU for the interaction trace phase.
    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();

        let trace: Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> = self
            .multiplicities
            .iter()
            .map(|m| {
                let m_clone = m.clone();
                let col = BaseFieldVec::new(m_clone.device_ptr, m_clone.size);
                std::mem::forget(m_clone);
                CircleEvaluation::new(domain, col)
            })
            .collect();

        tree_builder.extend_evals(trace);

        (
            Claim {},
            CudaInteractionClaimGenerator {
                multiplicities: std::array::from_fn(|i| self.multiplicities[i].clone()),
            },
        )
    }
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// CUDA-based interaction claim generator for verify_bitwise_xor_12.
///
/// Keeps multiplicities on GPU. Interaction trace is computed entirely on GPU
/// via the native CUDA kernel that processes 8 column pairs iteratively,
/// producing 32 interaction trace columns (8 pairs × 4 QM31 coords).
pub struct CudaInteractionClaimGenerator {
    multiplicities: [Uint32Vec; N_MULT_COLUMNS],
}

impl CudaInteractionClaimGenerator {
    /// Write the interaction trace using native CUDA kernel.
    ///
    /// The kernel processes 8 column pairs, each generating logup fractions,
    /// batch-inverting, and accumulating running sums. The relation constant
    /// `VERIFY_BITWISE_XOR_12_RELATION_ID` is hardcoded in the CUDA kernel,
    /// so `CommonLookupElements` is passed directly (same pattern as VBX 4/7/9).
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_size = 1usize << LOG_SIZE;

        // 8 column pairs × 4 QM31 coords = 32 interaction trace columns
        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..32)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size))
            .collect();

        let cuda_claimed_sum: Col<CudaBackend, BaseField> = Col::<CudaBackend, BaseField>::zeros(4);

        let mults_ptrs: Vec<*const u32> =
            self.multiplicities.iter().map(|m| m.device_ptr).collect();
        let trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        unsafe {
            bindings_airs::verify_bitwise_xor_12_interaction_trace(
                lookup_elements as *const _ as *mut std::os::raw::c_void,
                mults_ptrs.as_ptr(),
                LOG_SIZE,
                trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        let cs = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([cs[0], cs[1], cs[2], cs[3]]);

        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
            })
            .collect();
        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
