
use cairo_air::components::memory_address_to_id::{
    Claim, InteractionClaim, MEMORY_ADDRESS_TO_ID_SPLIT, N_TRACE_COLUMNS,
};
use cairo_air::relations;
use itertools::Itertools;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::QM31;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_cairo_adapter::memory::Memory;

use crate::witness::utils::TreeBuilder;

pub type InputType = M31;
pub type PackedInputType = PackedM31;
pub type CudaPackedInputType = [BaseFieldVec; 1];

/// A struct that represents a mapping from Address to ID. Zero address is not allowed.
#[derive(Debug)]
pub struct CudaAddressToId {
    /// Since zero address is reserved, the vector holding the data is offset by 1, i.e. the ID of
    /// address 1 is stored at index 0, and so on.
    data: Uint32Vec,
}
impl CudaAddressToId {
    pub fn new(data: Uint32Vec) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A struct to generate the memory address to ID trace.
pub struct CudaClaimGenerator {
    pub address_to_raw_id: Uint32Vec,
    pub multiplicities: Uint32Vec,
}

impl CudaClaimGenerator {
    pub fn new(memory: &Memory) -> Self {
        // Note that while `memory.address_to_id` starts from address 0, the memory component can
        // only yield addresses starting from 1.
        let mem_vec = (1..memory.address_to_id.len())
            .map(|addr| memory.get_raw_id(addr as u32))
            .collect_vec();
        let address_to_raw_id = Uint32Vec::from_vec(mem_vec);
        let multiplicities = Uint32Vec::new_zeroes(address_to_raw_id.size);

        Self {
            address_to_raw_id,
            multiplicities,
        }
    }

    /// Merges SIMD multiplicities into this CUDA generator.
    /// This should be called before write_trace() to ensure all multiplicities
    /// from SIMD opcodes are included.
    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
        // Get current CUDA multiplicities
        let mut cuda_mults = self.multiplicities.to_vec();

        // Add SIMD multiplicities (handle size difference - SIMD may have more entries)
        let min_len = std::cmp::min(cuda_mults.len(), simd_multiplicities.len());
        for i in 0..min_len {
            cuda_mults[i] += simd_multiplicities[i];
        }

        // Replace multiplicities with merged data
        self.multiplicities = Uint32Vec::from_vec(cuda_mults);
    }

    pub fn add_cuda_input(&mut self, addr: &BaseField) {
        self.multiplicities.increase_at(addr.0 - 1);
    }

    pub fn add_cuda_inputs(&mut self, cuda_inputs: &[CudaPackedInputType]) {
        let inputs_vec = cuda_inputs
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        unsafe {
            bindings_airs::memory_address_to_id_add_inputs(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                self.multiplicities.device_ptr,
                self.multiplicities.size.ilog2(),
            );
        }
    }

    pub fn get_id(&self, input: BaseField) -> M31 {
        let id = self.address_to_raw_id.get_data(input.0 as usize);
        M31(id)
    }

    pub fn get_id_vec(&self, input_vec: Vec<BaseField>) -> BaseFieldVec {
        let id_vec = input_vec
            .iter()
            .map(|x| M31(self.address_to_raw_id.get_data(x.0 as usize)))
            .collect::<Vec<_>>();
        BaseFieldVec::from_vec(id_vec)
    }

    /// Generate trace on CUDA and return evaluations (without adding to tree_builder).
    /// This allows the caller to add traces in the correct order.
    pub fn generate_trace(
        self,
    ) -> (
        Claim,
        Vec<stwo::prover::poly::circle::CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
        CudaInteractionClaimGenerator,
    ) {
        // Use address_to_raw_id.size to match SIMD behavior
        let address_to_id_len = self.address_to_raw_id.size;

        // Calculate size (padded to power of 2), matching SIMD logic
        let size = std::cmp::max(
            (address_to_id_len.div_ceil(MEMORY_ADDRESS_TO_ID_SPLIT)).next_power_of_two(),
            16,
        );
        let log_size = size.checked_ilog2().unwrap();
        // Use next_power_of_two to ensure proper bounds checking in the kernel
        let total_size_log = self
            .address_to_raw_id
            .size
            .next_power_of_two()
            .checked_ilog2()
            .unwrap();

        // Allocate trace columns on GPU
        let traces: Vec<BaseFieldVec> =
            (0..N_TRACE_COLUMNS).map(|_| BaseFieldVec::new_uninitialized(size)).collect();

        let trace_ptrs: Vec<*const u32> = traces.iter().map(|t| t.device_ptr).collect();

        // Call CUDA kernel to generate trace
        unsafe {
            bindings_airs::generate_memory_address_to_id_traces(
                trace_ptrs.as_ptr(),
                std::ptr::null(), // No interaction trace yet
                self.address_to_raw_id.device_ptr,
                self.multiplicities.device_ptr,
                total_size_log,
                log_size,
                std::ptr::null_mut(), // No lookup elements yet
            );
        }

        // Convert to CircleEvaluations
        let domain = CanonicCoset::new(log_size).circle_domain();
        let evals = traces
            .into_iter()
            .map(|trace| {
                stwo::prover::poly::circle::CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                    domain,
                    trace,
                )
            })
            .collect::<Vec<_>>();

        (
            Claim { log_size },
            evals,
            CudaInteractionClaimGenerator {
                address_to_raw_id: self.address_to_raw_id,
                multiplicities: self.multiplicities,
                total_size_log,
                log_size,
            },
        )
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let (claim, evals, interaction_gen) = self.generate_trace();
        tree_builder.extend_evals(evals);
        (claim, interaction_gen)
    }
}

pub struct CudaInteractionClaimGenerator {
    address_to_raw_id: Uint32Vec,
    multiplicities: Uint32Vec,
    total_size_log: u32,
    log_size: u32,
}

impl CudaInteractionClaimGenerator {
    /// Generate interaction trace on CUDA and return evaluations (without adding to tree_builder).
    pub fn generate_interaction_trace(
        self,
        lookup_elements: &relations::MemoryAddressToId,
    ) -> (
        InteractionClaim,
        Vec<stwo::prover::poly::circle::CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    ) {
        let size = 1 << self.log_size;

        // Number of interaction columns: 8 M31 values per pair (4 for numerator, 4 for denominator)
        // We have 8 pairs (16 chunks / 2)
        let num_interaction_cols = 8 * (MEMORY_ADDRESS_TO_ID_SPLIT / 2);

        // Allocate interaction trace columns on GPU
        let interaction_traces: Vec<BaseFieldVec> =
            (0..num_interaction_cols).map(|_| BaseFieldVec::new_uninitialized(size)).collect();

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_traces.iter().map(|t| t.device_ptr).collect();

        // Allocate trace columns on GPU (needed for reading IDs and multiplicities)
        let traces: Vec<BaseFieldVec> =
            (0..N_TRACE_COLUMNS).map(|_| BaseFieldVec::new_uninitialized(size)).collect();

        let trace_ptrs: Vec<*const u32> = traces.iter().map(|t| t.device_ptr).collect();

        // Regenerate trace for interaction calculation
        unsafe {
            bindings_airs::generate_memory_address_to_id_traces(
                trace_ptrs.as_ptr(),
                interaction_trace_ptrs.as_ptr(),
                self.address_to_raw_id.device_ptr,
                self.multiplicities.device_ptr,
                self.total_size_log,
                self.log_size,
                lookup_elements as *const _ as *mut std::ffi::c_void,
            );
        }

        // Convert to CircleEvaluations
        let domain = CanonicCoset::new(self.log_size).circle_domain();
        let evals = interaction_traces
            .iter()
            .map(|trace| {
                stwo::prover::poly::circle::CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                    domain,
                    trace.clone(),
                )
            })
            .collect::<Vec<_>>();

        // Calculate claimed sum from interaction trace
        // Sum the last row of all interaction columns
        let mut claimed_sum = QM31::default();
        for col_idx in (0..num_interaction_cols).step_by(8) {
            // Each logup column is represented as 8 M31 values (numerator and denominator, each 4 M31)
            // We need to sum the fractions
            let last_row_idx = (size - 1) as usize;

            let num_a = interaction_traces[col_idx].to_vec()[last_row_idx];
            let num_b = interaction_traces[col_idx + 1].to_vec()[last_row_idx];
            let num_c = interaction_traces[col_idx + 2].to_vec()[last_row_idx];
            let num_d = interaction_traces[col_idx + 3].to_vec()[last_row_idx];

            let denom_a = interaction_traces[col_idx + 4].to_vec()[last_row_idx];
            let denom_b = interaction_traces[col_idx + 5].to_vec()[last_row_idx];
            let denom_c = interaction_traces[col_idx + 6].to_vec()[last_row_idx];
            let denom_d = interaction_traces[col_idx + 7].to_vec()[last_row_idx];

            let numerator = QM31::from_m31_array([num_a, num_b, num_c, num_d]);
            let denominator = QM31::from_m31_array([denom_a, denom_b, denom_c, denom_d]);

            // Add numerator / denominator to claimed_sum
            claimed_sum += numerator / denominator;
        }

        (InteractionClaim { claimed_sum }, evals)
    }

    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &relations::MemoryAddressToId,
    ) -> InteractionClaim {
        let (claim, evals) = self.generate_interaction_trace(lookup_elements);
        tree_builder.extend_evals(evals);
        claim
    }
}


