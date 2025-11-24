#![allow(unused_parens)]
#![allow(dead_code)]

use cairo_air::components::verify_bitwise_xor_12::{
    LOG_SIZE, N_MULT_COLUMNS,
};
use itertools::Itertools;

use crate::witness::prelude::*;
// use crate::witness::utils_cuda::AtomicMultiplicityColumnCuda;

pub type InputType = [M31; 3];
pub type PackedInputType = [PackedM31; 3];

use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
pub type CudaPackedInputType = [BaseFieldVec; 3];


use stwo::stwo_cuda::bindings_airs;
use stwo::stwo_cuda::base_field_vec::Uint32Vec;

pub struct CudaClaimGenerator {
    pub mults: Vec<Uint32Vec>,
}
impl CudaClaimGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            mults:  (0..N_MULT_COLUMNS)
                .map(|_| Uint32Vec::new_zeroes(1 << LOG_SIZE))
                .collect_vec(),
            }
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        // println!("cuda_inputs verify_bitwise_xor_8_cuda inputs cols: {}, rows:{}, mults log_size:{}", cuda_inputs.len(), cuda_inputs[0][0].size, LOG_SIZE);
        let inputs_vec = cuda_inputs.iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        let mults_vec = self.mults
            .iter()
            .map(|row| row.device_ptr)
            .collect_vec();
        unsafe {
            bindings_airs::verify_bitwise_xor_12_mults_init(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                mults_vec.as_ptr(),
                N_MULT_COLUMNS as u32,
                1 << LOG_SIZE as u32,
            );
        }
    }
}

#[derive(Uninitialized, IterMut, ParIterMut)]
struct CudaLookupDataCuda {
    mults: Vec<BaseFieldVec>,
}

pub struct CudaInteractionClaimGenerator {
    lookup_data: CudaLookupDataCuda,
}
