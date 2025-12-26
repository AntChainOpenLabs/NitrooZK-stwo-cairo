//! Generic CUDA implementation for range check components.
//!
//! This module provides a generic `CudaRangeCheckGenerator` that can be used for
//! any range check configuration (single or multi-segment).
//!
//! The implementation uses CUDA for multiplicity accumulation and trace generation.

use itertools::Itertools;
use std::simd::Simd;
use stwo::core::fields::m31::M31;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_cairo_common::preprocessed_columns::preprocessed_utils::SIMD_ENUMERATION_0;

use crate::witness::utils::TreeBuilder;

/// Generic CUDA range check generator for N_RANGES segments.
///
/// This generator accumulates multiplicities on the GPU using atomicAdd.
/// The multiplicities can later be merged with SIMD generators for trace generation.
pub struct CudaRangeCheckGenerator<const N_RANGES: usize> {
    pub multiplicities: Uint32Vec,
    pub ranges: [u32; N_RANGES],
    pub log_size: u32,
}

impl<const N_RANGES: usize> CudaRangeCheckGenerator<N_RANGES> {
    /// Create a new CUDA range check generator with the specified ranges.
    ///
    /// # Arguments
    /// * `ranges` - Array of bit sizes for each segment (e.g., [7, 2, 5] for rc_7_2_5)
    pub fn new(ranges: [u32; N_RANGES]) -> Self {
        let log_size: u32 = ranges.iter().sum();
        let size = 1usize << log_size;
        Self {
            multiplicities: Uint32Vec::new_zeroes(size),
            ranges,
            log_size,
        }
    }

    /// Add inputs from CUDA vectors to the multiplicity table using atomicAdd.
    ///
    /// # Arguments
    /// * `inputs` - Array of BaseFieldVec containing the input values for each segment
    /// * `n_rows` - Number of rows in the input vectors
    ///
    /// # Safety
    /// This function calls CUDA kernels through FFI bindings.
    pub fn add_inputs_internal(&self, inputs: &[BaseFieldVec; N_RANGES], n_rows: usize) {
        unsafe {
            let input_ptrs: Vec<*const u32> = inputs.iter().map(|v| v.device_ptr).collect();
            // CUDA kernel expects:
            // - input_col_sizes: number of batches (columns of input) = 1
            // - input_row_sizes: number of rows = n_rows
            // - n_range: number of range segments = N_RANGES
            // - inputs array size: input_col_sizes * n_range = N_RANGES
            bindings_airs::range_check_vector_add_inputs(
                input_ptrs.as_ptr(),
                1u32,           // One batch at a time
                n_rows as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                self.multiplicities.device_ptr,
                self.log_size,
            );
        }
    }

    /// Add inputs from multiple CUDA input batches.
    ///
    /// # Arguments
    /// * `cuda_inputs` - Slice of input batches, where each batch contains N_RANGES BaseFieldVecs
    pub fn add_inputs_batch_internal(&self, cuda_inputs: &[[BaseFieldVec; N_RANGES]]) {
        if cuda_inputs.is_empty() {
            return;
        }

        let inputs_vec: Vec<*const u32> = cuda_inputs
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        let n_rows = cuda_inputs[0][0].size;

        unsafe {
            bindings_airs::range_check_vector_add_inputs(
                inputs_vec.as_ptr(),
                (cuda_inputs.len() * N_RANGES) as u32,
                n_rows as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                self.multiplicities.device_ptr,
                self.log_size,
            );
        }
    }

    /// Get the multiplicities as a CPU vector.
    pub fn get_multiplicities(&self) -> Vec<u32> {
        self.multiplicities.to_vec()
    }

    /// Get the log size of the range check table.
    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// Write the trace for this range check component.
    ///
    /// Converts multiplicities from GPU to CPU and generates the trace.
    pub fn write_trace(
        &self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
    ) -> CudaInteractionClaimGenerator<N_RANGES> {
        let log_size = self.log_size;

        // Get multiplicities from GPU
        let multiplicities_u32 = self.multiplicities.to_vec();

        // Convert to PackedM31 for SIMD processing
        let n_packed = multiplicities_u32.len() / 16; // N_LANES = 16
        let mut multiplicity_data = Vec::with_capacity(n_packed);
        for chunk in multiplicities_u32.chunks(16) {
            let arr: [u32; 16] = chunk.try_into().unwrap_or([0; 16]);
            let simd = std::simd::Simd::from_array(arr);
            multiplicity_data.push(unsafe { PackedM31::from_simd_unchecked(simd) });
        }

        let multiplicity_column = BaseColumn::from_simd(multiplicity_data.clone());

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace = [multiplicity_column]
            .map(|col| CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(domain, col));

        tree_builder.extend_evals(trace);

        CudaInteractionClaimGenerator {
            multiplicities: multiplicity_data,
            ranges: self.ranges,
            log_size,
        }
    }
}

/// Interaction claim generator for CUDA range checks.
#[derive(Debug)]
pub struct CudaInteractionClaimGenerator<const N_RANGES: usize> {
    pub multiplicities: Vec<PackedM31>,
    pub ranges: [u32; N_RANGES],
    pub log_size: u32,
}

impl<const N_RANGES: usize> CudaInteractionClaimGenerator<N_RANGES> {
    /// Write the interaction trace for this range check component.
    ///
    /// Uses the accumulated multiplicities to generate the logup trace.
    pub fn write_interaction_trace<L>(
        &self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        lookup_elements: &L,
    ) -> stwo::core::fields::qm31::SecureField
    where
        L: stwo_constraint_framework::Relation<PackedM31, stwo::prover::backend::simd::qm31::PackedSecureField>,
    {
        use stwo::prover::backend::simd::m31::LOG_N_LANES;
        use stwo::prover::backend::simd::qm31::PackedSecureField;
        use stwo_constraint_framework::LogupTraceGenerator;

        let log_size = self.log_size;
        let mut logup_gen = LogupTraceGenerator::new(log_size);
        let mut col_gen = logup_gen.new_col();

        // Lookup values columns.
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let numerator = PackedSecureField::from(-self.multiplicities[vec_row]);
            let partitions = stwo_cairo_common::preprocessed_columns::preprocessed_trace::partition_into_bit_segments(
                SIMD_ENUMERATION_0 + Simd::splat((vec_row * 16) as u32),
                self.ranges,
            );
            let partitions: [_; N_RANGES] = std::array::from_fn(|i| unsafe {
                PackedM31::from_simd_unchecked(partitions[i])
            });
            let denom = lookup_elements.combine(&partitions);
            col_gen.write_frac(vec_row, numerator, denom);
        }
        col_gen.finalize_col();

        let (trace, claimed_sum) = logup_gen.finalize_last();
        tree_builder.extend_evals(trace);

        claimed_sum
    }
}
