//! CUDA version of pedersen_points_table for GPU multiplicity tracking and trace generation.
//!
//! The pedersen_points_table is a static lookup table (~8M rows, LOG_SIZE=23)
//! used by partial_ec_mul. This module:
//! - Tracks multiplicities on GPU using atomicAdd
//! - Generates base trace directly from GPU multiplicities (no CPU round-trip)
//! - Generates interaction trace via native CUDA kernel (no CPU round-trip)

use cairo_air::components::pedersen_points_table_window_bits_18::{
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

// Calculate n_rows for WINDOW_BITS=18: (2 * 252 / 18) << 18
const PEDERSEN_TABLE_N_ROWS: usize = 7340032;

/// Ensure the GPU pedersen table is initialized.
///
/// Generates the table directly on GPU via `initialize_pedersen_table()`.
/// No CPU→GPU data transfer needed.
///
/// Safe to call multiple times; subsequent calls are no-ops.
pub fn ensure_pedersen_table() {
    unsafe {
        if bindings_airs::is_pedersen_table_initialized() {
            return;
        }
        bindings_airs::initialize_pedersen_table();
    }
}

/// CUDA packed input type for pedersen_points_table (1 column: the table index)
pub type CudaPackedInputType = [BaseFieldVec; 1];

/// CUDA generator for pedersen_points_table multiplicities and trace generation.
pub struct CudaClaimGenerator {
    /// GPU multiplicities (atomically updated by CUDA kernels)
    pub multiplicities: Uint32Vec,
    /// Log size of the table
    pub log_size: u32,
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaClaimGenerator {
    /// Create a new CUDA pedersen_points_table generator.
    ///
    /// Allocates GPU memory for multiplicities matching the table size.
    pub fn new() -> Self {
        // PEDERSEN_TABLE_N_ROWS is the total number of rows in the pedersen points table
        let log_size = PEDERSEN_TABLE_N_ROWS.next_power_of_two().ilog2();
        let size = 1usize << log_size;

        Self {
            multiplicities: Uint32Vec::new_zeroes(size),
            log_size,
        }
    }

    /// Get the device pointer for multiplicities (for passing to CUDA kernels).
    pub fn multiplicities_ptr(&self) -> *const u32 {
        self.multiplicities.device_ptr
    }

    /// Get multiplicities from GPU as a CPU Vec<u32>.
    /// Used for merging CUDA multiplicities into SIMD generators.
    pub fn get_multiplicities_cpu(&self) -> Vec<u32> {
        self.multiplicities.to_vec()
    }

    /// Merges SIMD multiplicities into this CUDA generator.
    /// This should be called to add multiplicities from SIMD trace generation.
    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
        // Get current CUDA multiplicities
        let mut cuda_mults = self.multiplicities.to_vec();

        // Add SIMD multiplicities (handle size difference)
        let min_len = std::cmp::min(cuda_mults.len(), simd_multiplicities.len());
        for i in 0..min_len {
            cuda_mults[i] += simd_multiplicities[i];
        }

        // Replace multiplicities with merged data
        self.multiplicities = Uint32Vec::from_vec(cuda_mults);
    }

    /// Add CUDA inputs to update multiplicities.
    ///
    /// Takes arrays of table indices (raw lookup values) and atomically
    /// adds 1 to the multiplicity at each index.
    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        for input in cuda_inputs {
            let n_rows = input[0].size;
            if n_rows == 0 {
                continue;
            }
            unsafe {
                bindings_airs::pedersen_points_table_add_inputs(
                    input[0].device_ptr,
                    n_rows as u32,
                    self.multiplicities.device_ptr,
                    self.log_size,
                );
            }
        }
    }

    /// Write base trace and return interaction claim generator.
    ///
    /// Generates the base trace (1 multiplicity column) in natural order,
    /// matching the SIMD path exactly. The result is converted to CUDA
    /// and committed.
    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        assert_eq!(
            log_size, LOG_SIZE,
            "pedersen_points_table log_size mismatch"
        );

        let n = 1usize << log_size;
        let mults_cpu = self.multiplicities.to_vec();

        // Build the base trace column in natural order, matching the SIMD path.
        // The preprocessed Seq column at position i stores value i (natural order),
        // so the multiplicity at position i must be mults[i].
        let mults_bf: Vec<BaseField> = (0..n)
            .map(|i| BaseField::from_u32_unchecked(mults_cpu[i]))
            .collect();

        // Upload to GPU via SIMD → CUDA conversion path
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

        // Pass original (natural order) multiplicities to interaction trace generator.
        (
            Claim {},
            CudaInteractionClaimGenerator {
                multiplicities: self.multiplicities,
                log_size,
            },
        )
    }
}

/// CUDA interaction claim generator for pedersen_points_table.
///
/// Keeps multiplicities on GPU for direct CUDA interaction trace generation.
pub struct CudaInteractionClaimGenerator {
    pub multiplicities: Uint32Vec,
    pub log_size: u32,
}

impl CudaInteractionClaimGenerator {
    /// Write the interaction trace using the native CUDA kernel.
    ///
    /// The pedersen_points_table CUDA kernel computes the logup interaction trace
    /// entirely on GPU using the GPU-resident pedersen table columns and multiplicities.
    /// This avoids downloading multiplicities to CPU, loading 56 pedersen table columns,
    /// running SIMD LogupTraceGenerator, and uploading back.
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let log_size = self.log_size;

        // Ensure GPU pedersen table is initialized (no-op if already done).
        ensure_pedersen_table();

        // The CUDA kernel uses LookupElementsBasic<58> (including relation_id as values[0]),
        // matching the SIMD convention. Pass CommonLookupElements directly — the kernel copies
        // only the first 58 alpha_powers it needs from the 128-element struct.

        // Allocate 4 output columns on GPU (1 QM31 logup column = 4 M31 components).
        let interaction_trace: Vec<BaseFieldVec> = (0..4)
            .map(|_| BaseFieldVec::new_zeroes(1 << log_size))
            .collect();

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        // Allocate claimed_sum output on GPU (4 x M31 for QM31).
        let cuda_claimed_sum = BaseFieldVec::new_zeroes(4);

        unsafe {
            bindings_airs::pedersen_points_table_interaction_trace(
                lookup_elements as *const CommonLookupElements as *mut std::os::raw::c_void,
                self.multiplicities.device_ptr,
                log_size,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        // Download claimed_sum from GPU.
        let claimed_sum_vec = cuda_claimed_sum.to_vec();
        let claimed_sum = SecureField::from_m31_array([
            claimed_sum_vec[0],
            claimed_sum_vec[1],
            claimed_sum_vec[2],
            claimed_sum_vec[3],
        ]);

        // Wrap output columns as CircleEvaluations and extend the tree builder.
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
