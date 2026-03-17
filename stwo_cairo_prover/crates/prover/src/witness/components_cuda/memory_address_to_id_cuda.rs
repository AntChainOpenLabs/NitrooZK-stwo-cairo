use cairo_air::components::memory_address_to_id::{
    Claim, InteractionClaim, MEMORY_ADDRESS_TO_ID_SPLIT, N_TRACE_COLUMNS,
};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_cairo_adapter::memory::Memory;

use crate::witness::utils::TreeBuilder;

const N_LOGUP_COLS: usize = MEMORY_ADDRESS_TO_ID_SPLIT / 2;
const N_INTERACTION_COLS: usize = N_LOGUP_COLS * 4;

pub type InputType = M31;
pub type PackedInputType = PackedM31;
pub type CudaPackedInputType = [BaseFieldVec; 1];

#[derive(Debug)]
pub struct CudaAddressToId {
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

pub struct CudaClaimGenerator {
    pub address_to_raw_id: Uint32Vec,
    pub multiplicities: Uint32Vec,
}

impl CudaClaimGenerator {
    pub fn new(memory: &Memory) -> Self {
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

    pub fn get_multiplicities_sum(&self) -> u64 {
        let mults = self.multiplicities.to_vec();
        mults.iter().map(|&x| x as u64).sum()
    }

    pub fn get_multiplicities(&self) -> Vec<u32> {
        self.multiplicities.to_vec()
    }

    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
        // Upload SIMD mults to GPU and add in-place — no download/upload roundtrip.
        let gpu_simd = Uint32Vec::from_vec(simd_multiplicities.to_vec());
        self.multiplicities.add_from(&gpu_simd);
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
            let mults_log_size = self.multiplicities.size.next_power_of_two().ilog2();
            bindings_airs::memory_address_to_id_add_inputs(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                self.multiplicities.device_ptr,
                mults_log_size,
            );
        }
    }

    pub fn get_id(&self, input: BaseField) -> M31 {
        // address_to_raw_id is offset by 1: index 0 holds the ID for address 1.
        let id = self.address_to_raw_id.get_data(input.0 as usize - 1);
        M31(id)
    }

    pub fn get_id_vec(&self, input_vec: Vec<BaseField>) -> BaseFieldVec {
        let id_vec = input_vec
            .iter()
            .map(|x| M31(self.address_to_raw_id.get_data(x.0 as usize - 1)))
            .collect::<Vec<_>>();
        BaseFieldVec::from_vec(id_vec)
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let total_size = self.address_to_raw_id.size;

        let size = std::cmp::max(
            total_size
                .div_ceil(MEMORY_ADDRESS_TO_ID_SPLIT)
                .next_power_of_two(),
            N_LANES,
        );
        let log_size = size.checked_ilog2().unwrap();
        let trace_size = 1 << log_size;

        let padded_size = total_size.next_power_of_two();
        let total_size_log = padded_size.checked_ilog2().unwrap();

        let mut padded_ids = Uint32Vec::new_zeroes(padded_size);
        let mut padded_mults = Uint32Vec::new_zeroes(padded_size);

        padded_ids.copy_from(&self.address_to_raw_id);
        padded_mults.copy_from(&self.multiplicities);

        let trace_columns: [BaseFieldVec; N_TRACE_COLUMNS] =
            std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size));

        let trace_ptrs: Vec<*const u32> = trace_columns.iter().map(|col| col.device_ptr).collect();

        unsafe {
            bindings_airs::generate_memory_address_to_id_traces(
                trace_ptrs.as_ptr(),
                std::ptr::null(),
                padded_ids.device_ptr,
                padded_mults.device_ptr,
                total_size_log,
                log_size,
                std::ptr::null_mut(),
            );
        }

        let domain = CanonicCoset::new(log_size).circle_domain();
        let cuda_evals: Vec<_> = trace_columns
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
            })
            .collect();

        tree_builder.extend_evals(cuda_evals);

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                padded_ids,
                padded_mults,
                total_size_log,
                log_size,
            },
        )
    }
}

pub struct CudaInteractionClaimGenerator {
    pub padded_ids: Uint32Vec,
    pub padded_mults: Uint32Vec,
    pub total_size_log: u32,
    pub log_size: u32,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let log_size = self.log_size;
        let trace_size = 1usize << log_size;

        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..N_INTERACTION_COLS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(trace_size) })
            .collect();

        let cuda_claimed_sum: Col<CudaBackend, BaseField> = unsafe { Col::<CudaBackend, BaseField>::uninitialized(4) };

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        unsafe {
            let modified_lookup = super::cuda_lookup_helper::create_modified_lookup_for_cuda(
                lookup_elements,
                super::cuda_lookup_helper::MEMORY_ADDRESS_TO_ID_RELATION_ID,
            );
            let lookup_element_ptr = &modified_lookup as *const _ as *mut std::os::raw::c_void;

            bindings_airs::memory_address_to_id_generate_interaction_trace(
                lookup_element_ptr,
                self.padded_ids.device_ptr,
                self.padded_mults.device_ptr,
                self.total_size_log,
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

        InteractionClaim { claimed_sum }
    }
}
