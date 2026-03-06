//! CUDA version of pedersen_points_table_window_bits_9 for GPU multiplicity tracking.
//!
//! The small pedersen_points_table is a static lookup table (~32K rows, LOG_SIZE=15)
//! used by partial_ec_mul_window_bits_9. Adapted from pedersen_points_table_cuda.rs (w18).

use cairo_air::components::pedersen_points_table_window_bits_9::{
    Claim, InteractionClaim, LOG_SIZE,
};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;

use crate::witness::utils::TreeBuilder;

// Total unpadded rows for WINDOW_BITS=9: 28672
const PEDERSEN_TABLE_SMALL_N_ROWS: usize = 28672;

/// Ensure the GPU small pedersen table is initialized.
pub fn ensure_pedersen_table_small() {
    unsafe {
        if bindings_airs::is_pedersen_table_small_initialized() {
            return;
        }
        bindings_airs::initialize_pedersen_table_small();
    }
}

/// CUDA packed input type (1 column: the table index)
pub type CudaPackedInputType = [BaseFieldVec; 1];

pub struct CudaClaimGenerator {
    pub multiplicities: Uint32Vec,
    pub log_size: u32,
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        let log_size = PEDERSEN_TABLE_SMALL_N_ROWS.next_power_of_two().ilog2();
        let size = 1usize << log_size;

        Self {
            multiplicities: Uint32Vec::new_zeroes(size),
            log_size,
        }
    }

    pub fn multiplicities_ptr(&self) -> *const u32 {
        self.multiplicities.device_ptr
    }

    pub fn get_multiplicities_cpu(&self) -> Vec<u32> {
        self.multiplicities.to_vec()
    }

    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
        let mut cuda_mults = self.multiplicities.to_vec();
        let min_len = std::cmp::min(cuda_mults.len(), simd_multiplicities.len());
        for i in 0..min_len {
            cuda_mults[i] += simd_multiplicities[i];
        }
        self.multiplicities = Uint32Vec::from_vec(cuda_mults);
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        for input in cuda_inputs {
            let n_rows = input[0].size;
            if n_rows == 0 {
                continue;
            }
            unsafe {
                bindings_airs::pedersen_points_table_small_add_inputs(
                    input[0].device_ptr,
                    n_rows as u32,
                    self.multiplicities.device_ptr,
                    self.log_size,
                );
            }
        }
    }

    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        assert_eq!(
            log_size, LOG_SIZE,
            "pedersen_points_table_wb9 log_size mismatch"
        );

        let n = 1usize << log_size;
        let mults_cpu = self.multiplicities.to_vec();

        let mults_bf: Vec<BaseField> = (0..n)
            .map(|i| BaseField::from_u32_unchecked(mults_cpu[i]))
            .collect();

        use stwo::prover::backend::simd::SimdBackend;
        let simd_col =
            <SimdBackend as stwo::prover::backend::ColumnOps<BaseField>>::Column::from_iter(
                mults_bf,
            );
        let domain = CanonicCoset::new(log_size).circle_domain();
        let simd_eval =
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, simd_col);
        let cuda_eval = crate::witness::cairo_cuda::convert_simd_to_cuda_evaluation(simd_eval);

        tree_builder.extend_evals(vec![cuda_eval]);

        (
            Claim {},
            CudaInteractionClaimGenerator {
                multiplicities: self.multiplicities,
                log_size,
            },
        )
    }
}

pub struct CudaInteractionClaimGenerator {
    pub multiplicities: Uint32Vec,
    pub log_size: u32,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let log_size = self.log_size;

        ensure_pedersen_table_small();

        let interaction_trace: Vec<BaseFieldVec> = (0..4)
            .map(|_| BaseFieldVec::new_zeroes(1 << log_size))
            .collect();

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        let cuda_claimed_sum = BaseFieldVec::new_zeroes(4);

        unsafe {
            bindings_airs::pedersen_points_table_wb9_interaction_trace(
                lookup_elements as *const CommonLookupElements as *mut std::os::raw::c_void,
                self.multiplicities.device_ptr,
                log_size,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        let claimed_sum_vec = cuda_claimed_sum.to_vec();
        let claimed_sum = SecureField::from_m31_array([
            claimed_sum_vec[0],
            claimed_sum_vec[1],
            claimed_sum_vec[2],
            claimed_sum_vec[3],
        ]);

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
