#![allow(unused_parens)]
#![allow(dead_code)]
use cairo_air::components::verify_bitwise_xor_9::{
    InteractionClaim, LOG_SIZE,
};

use crate::witness::prelude::*;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
pub type InputType = [M31; 3];
pub type PackedInputType = [PackedM31; 3];
pub type CudaPackedInputType = [BaseFieldVec; 3];

use itertools::Itertools;
use stwo::stwo_cuda::bindings_airs;
use stwo::stwo_cuda::base_field_vec::Uint32Vec;

pub struct CudaClaimGenerator {
    pub mults: Uint32Vec,
}
impl CudaClaimGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            mults: Uint32Vec::new_zeroes(1 << LOG_SIZE),
        }
    }
    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        // println!("cuda_inputs verify_bitwise_xor_9_cuda inputs cols: {}, rows:{}, mults log_size:{}", cuda_inputs.len(), cuda_inputs[0][0].size, LOG_SIZE);
        let inputs_vec = cuda_inputs.iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        unsafe {
            bindings_airs::verify_bitwise_xor_9_mults_init(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                self.mults.device_ptr,
                LOG_SIZE,
            );
        }
    }
}

#[derive(Uninitialized, IterMut, ParIterMut)]
struct CudaLookupData {
    verify_bitwise_xor_9_0: Vec<[PackedM31; 3]>,
    mults: Vec<PackedM31>,
}

pub struct CudaInteractionClaimGenerator {
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        verify_bitwise_xor_9: &relations::VerifyBitwiseXor_9,
    ) -> InteractionClaim {
        let mut logup_gen = LogupTraceGenerator::new(LOG_SIZE);

        // Sum last logup term.
        let mut col_gen = logup_gen.new_col();
        (
            col_gen.par_iter_mut(),
            &self.lookup_data.verify_bitwise_xor_9_0,
            self.lookup_data.mults,
        )
            .into_par_iter()
            .for_each(|(writer, values, mults)| {
                let denom = verify_bitwise_xor_9.combine(values);
                writer.write_frac(-PackedQM31::one() * mults, denom);
            });
        col_gen.finalize_col();

        let (trace, claimed_sum) = logup_gen.finalize_last();
        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
