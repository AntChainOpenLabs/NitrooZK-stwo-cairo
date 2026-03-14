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

use crate::witness::utils::TreeBuilder;

pub type CudaPackedInputType = [BaseFieldVec; 3];

pub struct CudaVerifyBitwiseXorGenerator {
    pub multiplicities: Uint32Vec,
    pub n_bits: u32,
    pub log_size: u32,
}

impl CudaVerifyBitwiseXorGenerator {
    pub fn new(n_bits: u32) -> Self {
        let log_size = 2 * n_bits;
        let size = 1usize << log_size;
        Self {
            multiplicities: Uint32Vec::new_zeroes(size),
            n_bits,
            log_size,
        }
    }

    pub fn add_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        if cuda_inputs.is_empty() {
            return;
        }

        for batch in cuda_inputs {
            let inputs_vec: Vec<*const u32> = batch.iter().map(|x| x.device_ptr).collect_vec();
            let n_rows = batch[0].size;

            unsafe {
                match self.n_bits {
                    4 => bindings_airs::verify_bitwise_xor_4_mults_init(
                        inputs_vec.as_ptr(),
                        1,
                        n_rows as u32,
                        self.multiplicities.device_ptr,
                        self.log_size,
                    ),
                    7 => bindings_airs::verify_bitwise_xor_7_mults_init(
                        inputs_vec.as_ptr(),
                        1,
                        n_rows as u32,
                        self.multiplicities.device_ptr,
                        self.log_size,
                    ),
                    8 => bindings_airs::verify_bitwise_xor_8_mults_init(
                        inputs_vec.as_ptr(),
                        1,
                        n_rows as u32,
                        self.multiplicities.device_ptr,
                        self.log_size,
                    ),
                    9 => bindings_airs::verify_bitwise_xor_9_mults_init(
                        inputs_vec.as_ptr(),
                        1,
                        n_rows as u32,
                        self.multiplicities.device_ptr,
                        self.log_size,
                    ),
                    _ => panic!("Unsupported n_bits: {}", self.n_bits),
                }
            }
        }
    }

    pub fn get_multiplicities(&self) -> Vec<u32> {
        self.multiplicities.to_vec()
    }

    pub fn merge_simd_multiplicities(&mut self, simd_multiplicities: &[u32]) {
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
    ) -> CudaInteractionClaimGeneratorCuda {
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
            n_bits: self.n_bits,
            log_size,
        }
    }
}

#[derive(Debug)]
pub struct CudaInteractionClaimGeneratorCuda {
    pub multiplicities: Uint32Vec,
    pub n_bits: u32,
    pub log_size: u32,
}

impl CudaInteractionClaimGeneratorCuda {
    pub fn write_interaction_trace<L>(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &L,
    ) -> SecureField
    where
        L: stwo_constraint_framework::Relation<
            stwo::prover::backend::simd::m31::PackedM31,
            stwo::prover::backend::simd::qm31::PackedSecureField,
        >,
    {
        let log_size = self.log_size;
        let trace_size = 1usize << log_size;

        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..4)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size))
            .collect();

        let cuda_claimed_sum: Col<CudaBackend, BaseField> = Col::<CudaBackend, BaseField>::zeros(4);

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|col| col.device_ptr).collect();

        unsafe {
            let lookup_element_ptr = lookup_elements as *const L as *mut std::os::raw::c_void;

            match self.n_bits {
                4 => bindings_airs::verify_bitwise_xor_4_interaction_trace(
                    lookup_element_ptr,
                    self.multiplicities.device_ptr,
                    log_size,
                    interaction_trace_ptrs.as_ptr(),
                    cuda_claimed_sum.device_ptr as *mut u32,
                ),
                7 => bindings_airs::verify_bitwise_xor_7_interaction_trace(
                    lookup_element_ptr,
                    self.multiplicities.device_ptr,
                    log_size,
                    interaction_trace_ptrs.as_ptr(),
                    cuda_claimed_sum.device_ptr as *mut u32,
                ),
                8 => bindings_airs::verify_bitwise_xor_8_interaction_trace(
                    lookup_element_ptr,
                    self.multiplicities.device_ptr,
                    log_size,
                    interaction_trace_ptrs.as_ptr(),
                    cuda_claimed_sum.device_ptr as *mut u32,
                ),
                9 => bindings_airs::verify_bitwise_xor_9_interaction_trace(
                    lookup_element_ptr,
                    self.multiplicities.device_ptr,
                    log_size,
                    interaction_trace_ptrs.as_ptr(),
                    cuda_claimed_sum.device_ptr as *mut u32,
                ),
                _ => panic!("Unsupported n_bits: {}", self.n_bits),
            }
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
