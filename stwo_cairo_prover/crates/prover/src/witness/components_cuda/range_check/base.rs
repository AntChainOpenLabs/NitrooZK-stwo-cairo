use itertools::Itertools;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_constraint_framework::logup::LookupElements;

use crate::witness::utils::TreeBuilder;

pub struct CudaRangeCheckGenerator<const N_RANGES: usize> {
    pub multiplicities: Uint32Vec,
    pub ranges: [u32; N_RANGES],
    pub log_size: u32,
}

impl<const N_RANGES: usize> CudaRangeCheckGenerator<N_RANGES> {
    pub fn new(ranges: [u32; N_RANGES]) -> Self {
        let log_size: u32 = ranges.iter().sum();
        let size = 1usize << log_size;
        Self {
            multiplicities: Uint32Vec::new_zeroes(size),
            ranges,
            log_size,
        }
    }

    pub fn add_inputs_internal(&self, inputs: &[BaseFieldVec; N_RANGES], n_rows: usize) {
        unsafe {
            let input_ptrs: Vec<*const u32> = inputs.iter().map(|v| v.device_ptr).collect();
            bindings_airs::range_check_vector_add_inputs(
                input_ptrs.as_ptr(),
                1u32,
                n_rows as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                self.multiplicities.device_ptr,
                self.log_size,
            );
        }
    }

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
                cuda_inputs.len() as u32,
                n_rows as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                self.multiplicities.device_ptr,
                self.log_size,
            );
        }
    }

    pub fn get_multiplicities(&self) -> Vec<u32> {
        self.multiplicities.to_vec()
    }

    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
        if std::env::var("MULT_DIAG").unwrap_or_default() == "1" {
            let cuda_mults = self.multiplicities.to_vec();
            let cuda_total: u64 = cuda_mults.iter().map(|&x| x as u64).sum();
            let simd_total: u64 = simd_multiplicities.iter().map(|&x| x as u64).sum();
            println!(
                "[MULT_DIAG] Single ranges={:?}: cuda_total={}, simd_total={}, combined={}",
                self.ranges,
                cuda_total,
                simd_total,
                cuda_total + simd_total
            );
        }

        // Upload SIMD mults to GPU and add in-place — no download/upload roundtrip.
        let gpu_simd = Uint32Vec::from_vec(simd_multiplicities.to_vec());
        self.multiplicities.add_from(&gpu_simd);
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> CudaInteractionClaimGeneratorCuda<N_RANGES> {
        let log_size = self.log_size;

        let multiplicities_for_trace = self.multiplicities.clone();
        let multiplicity_column = BaseFieldVec::new(
            multiplicities_for_trace.device_ptr,
            multiplicities_for_trace.size,
        );
        std::mem::forget(multiplicities_for_trace);

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
            domain,
            multiplicity_column,
        );

        tree_builder.extend_evals(vec![trace]);

        CudaInteractionClaimGeneratorCuda {
            multiplicities: self.multiplicities,
            ranges: self.ranges,
            log_size,
        }
    }
}

#[derive(Debug)]
pub struct CudaInteractionClaimGeneratorCuda<const N_RANGES: usize> {
    pub multiplicities: Uint32Vec,
    pub ranges: [u32; N_RANGES],
    pub log_size: u32,
}

impl<const N_RANGES: usize> CudaInteractionClaimGeneratorCuda<N_RANGES> {
    /// Writes the interaction trace for a single-relation range check component.
    ///
    /// The `relation_constant` is the M31 constant that identifies this relation in the AIR
    /// eval (the first element in the `RelationEntry` values array). It gets baked into
    /// modified lookup elements so the CUDA kernel computes the correct denominator.
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &LookupElements<128>,
        relation_constant: M31,
    ) -> SecureField {
        let modified_lookup =
            create_per_relation_lookup_elements::<N_RANGES>(lookup_elements, relation_constant);

        let log_size = self.log_size;
        let trace_size = 1usize << log_size;

        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..4)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size))
            .collect();

        let cuda_claimed_sum: Col<CudaBackend, BaseField> = Col::<CudaBackend, BaseField>::zeros(4);

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        unsafe {
            let lookup_element_ptr =
                &modified_lookup as *const LookupElements<N_RANGES> as *mut std::os::raw::c_void;

            bindings_airs::range_check_vector_generate_interaction_trace(
                lookup_element_ptr,
                self.multiplicities.device_ptr,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                log_size,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([
            claimed_sum_vec[0],
            claimed_sum_vec[1],
            claimed_sum_vec[2],
            claimed_sum_vec[3],
        ]);

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
            })
            .collect();

        tree_builder.extend_evals(trace);

        claimed_sum
    }
}

// ---------------------------------------------------------------------------
// Multi-relation range check generator
// ---------------------------------------------------------------------------

/// Creates a `LookupElements<N_RANGES>` that bakes a relation constant into `z` so the
/// existing CUDA kernel (which doesn't know about relation constants) computes the correct
/// denominator:
///
///   kernel computes: alpha_powers[0]*val0 + alpha_powers[1]*val1 + ... - z_modified
///
/// where z_modified = z - alpha_powers[0]*REL_CONST, so the result equals:
///
///   alpha_powers[0]*REL_CONST + alpha_powers[1]*val0 + ... - z   (correct)
///
/// The alpha_powers are shifted by 1 from the common lookup elements.
fn create_per_relation_lookup_elements<const N_RANGES: usize>(
    common: &LookupElements<128>,
    relation_constant: M31,
) -> LookupElements<N_RANGES> {
    LookupElements {
        z: common.z - common.alpha_powers[0] * SecureField::from(relation_constant),
        alpha: common.alpha,
        alpha_powers: std::array::from_fn(|i| common.alpha_powers[i + 1]),
    }
}

/// Multi-relation range check generator. Stores `N_RELATIONS` separate multiplicity vectors
/// (one per relation) over the same domain of size `2^sum(ranges)`.
pub struct CudaMultiRelationRangeCheckGenerator<const N_RANGES: usize, const N_RELATIONS: usize> {
    pub multiplicities: [Uint32Vec; N_RELATIONS],
    pub ranges: [u32; N_RANGES],
    pub log_size: u32,
}

impl<const N_RANGES: usize, const N_RELATIONS: usize>
    CudaMultiRelationRangeCheckGenerator<N_RANGES, N_RELATIONS>
{
    pub fn new(ranges: [u32; N_RANGES]) -> Self {
        let log_size: u32 = ranges.iter().sum();
        let size = 1usize << log_size;
        Self {
            multiplicities: std::array::from_fn(|_| Uint32Vec::new_zeroes(size)),
            ranges,
            log_size,
        }
    }

    /// Adds inputs to the multiplicity vector for a specific relation index.
    pub fn add_inputs_for_relation(
        &self,
        inputs: &[BaseFieldVec; N_RANGES],
        n_rows: usize,
        relation_index: usize,
    ) {
        unsafe {
            let input_ptrs: Vec<*const u32> = inputs.iter().map(|v| v.device_ptr).collect();
            bindings_airs::range_check_vector_add_inputs(
                input_ptrs.as_ptr(),
                1u32,
                n_rows as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                self.multiplicities[relation_index].device_ptr,
                self.log_size,
            );
        }
    }

    /// Batch-adds inputs to the multiplicity vector for a specific relation index.
    pub fn add_inputs_batch_for_relation(
        &self,
        cuda_inputs: &[[BaseFieldVec; N_RANGES]],
        relation_index: usize,
    ) {
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
                cuda_inputs.len() as u32,
                n_rows as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                self.multiplicities[relation_index].device_ptr,
                self.log_size,
            );
        }
    }

    /// Merges SIMD multiplicities (one `Vec<u32>` per relation) into the CUDA vectors.
    pub fn merge_simd_multiplicities(&mut self, simd_mults: &[Vec<u32>]) {
        assert_eq!(simd_mults.len(), N_RELATIONS);
        for (relation_idx, simd_mult) in simd_mults.iter().enumerate() {
            if std::env::var("MULT_DIAG").unwrap_or_default() == "1" {
                let cuda_mults = self.multiplicities[relation_idx].to_vec();
                let cuda_total: u64 = cuda_mults.iter().map(|&x| x as u64).sum();
                let simd_total: u64 = simd_mult.iter().map(|&x| x as u64).sum();
                println!(
                    "[MULT_DIAG] Multi ranges={:?} rel={}: cuda_total={}, simd_total={}, combined={}",
                    self.ranges, relation_idx, cuda_total, simd_total, cuda_total + simd_total
                );
            }

            // Upload SIMD mults to GPU and add in-place — no download/upload roundtrip.
            let gpu_simd = Uint32Vec::from_vec(simd_mult.clone());
            self.multiplicities[relation_idx].add_from(&gpu_simd);
        }
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// Writes N_RELATIONS multiplicity columns to the base trace and returns the
    /// interaction generator.
    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        relation_constants: [M31; N_RELATIONS],
    ) -> CudaMultiRelationInteractionGen<N_RANGES, N_RELATIONS> {
        let log_size = self.log_size;
        let domain = CanonicCoset::new(log_size).circle_domain();

        // Output one column per relation (the multiplicity column for that relation).
        let mut trace_evals = Vec::with_capacity(N_RELATIONS);
        for i in 0..N_RELATIONS {
            let mults_clone = self.multiplicities[i].clone();
            let column = BaseFieldVec::new(mults_clone.device_ptr, mults_clone.size);
            std::mem::forget(mults_clone);
            trace_evals.push(
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, column),
            );
        }
        tree_builder.extend_evals(trace_evals);

        CudaMultiRelationInteractionGen {
            multiplicities: self.multiplicities,
            relation_constants,
            ranges: self.ranges,
            log_size,
        }
    }
}

/// Interaction generator for multi-relation range checks.
/// Uses paired logup matching the SIMD LogupTraceGenerator column format:
/// - Intermediate columns (0..N_PAIRS-2): per-row cumulative fractions across pairs
/// - Last column (N_PAIRS-1): prefix sum over rows (with shift) of total fraction
///
/// The CUDA kernel handles all pairs entirely on GPU, eliminating CPU downloads
/// and the expensive fraction recovery + re-upload round-trips.
pub struct CudaMultiRelationInteractionGen<const N_RANGES: usize, const N_RELATIONS: usize> {
    pub multiplicities: [Uint32Vec; N_RELATIONS],
    pub relation_constants: [M31; N_RELATIONS],
    pub ranges: [u32; N_RANGES],
    pub log_size: u32,
}

impl<const N_RANGES: usize, const N_RELATIONS: usize>
    CudaMultiRelationInteractionGen<N_RANGES, N_RELATIONS>
{
    /// Writes the interaction trace using paired logup entirely on GPU.
    ///
    /// Calls a single CUDA kernel that processes all N_PAIRS relation pairs,
    /// computing per-pair logup fractions, accumulating across pairs via
    /// the finalize_col pattern, and applying cumsum_shift + prefix_sum
    /// on the last column.
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &LookupElements<128>,
    ) -> SecureField {
        let log_size = self.log_size;
        let trace_size = 1usize << log_size;
        let n_pairs = N_RELATIONS / 2;
        let domain = CanonicCoset::new(log_size).circle_domain();

        // 1. Create modified lookup elements per relation (bake relation constant into z, shift
        //    alpha_powers by 1).
        let modified_lookups: Vec<LookupElements<N_RANGES>> = (0..N_RELATIONS)
            .map(|i| {
                create_per_relation_lookup_elements::<N_RANGES>(
                    lookup_elements,
                    self.relation_constants[i],
                )
            })
            .collect();

        // Host array of pointers to the lookup elements (cast to *mut c_void).
        let lookup_ptrs: Vec<*mut std::os::raw::c_void> = modified_lookups
            .iter()
            .map(|le| le as *const LookupElements<N_RANGES> as *mut std::os::raw::c_void)
            .collect();

        // 2. Collect multiplicity device pointers.
        let mult_ptrs: Vec<*const u32> = self
            .multiplicities
            .iter()
            .map(|m| m.device_ptr as *const u32)
            .collect();

        // 3. Allocate interaction trace columns on GPU (4 × n_pairs).
        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..4 * n_pairs)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size))
            .collect();
        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        // 4. Allocate claimed_sum on GPU (4 M31s for QM31).
        let cuda_claimed_sum: Col<CudaBackend, BaseField> = Col::<CudaBackend, BaseField>::zeros(4);

        // 5. Call the multi-relation CUDA kernel.
        unsafe {
            bindings_airs::range_check_multi_relation_interaction_trace(
                n_pairs as u32,
                N_RANGES as u32,
                self.ranges.as_ptr(),
                lookup_ptrs.as_ptr(),
                mult_ptrs.as_ptr(),
                log_size,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        // 6. Read claimed_sum from GPU.
        let cs = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([cs[0], cs[1], cs[2], cs[3]]);

        // 7. Wrap columns as CircleEvaluations and extend tree builder.
        let trace: Vec<CircleEvaluation<CudaBackend, _, BitReversedOrder>> = interaction_trace
            .into_iter()
            .map(|col| CircleEvaluation::new(domain, col))
            .collect();

        tree_builder.extend_evals(trace);

        claimed_sum
    }
}
