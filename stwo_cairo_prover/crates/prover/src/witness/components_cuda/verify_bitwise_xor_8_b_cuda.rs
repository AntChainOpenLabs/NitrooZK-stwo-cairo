#![allow(unused_parens)]
#![allow(dead_code)]
use cairo_air::components::verify_bitwise_xor_8_b::{
    LOG_SIZE, N_TRACE_COLUMNS,
};
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::witness::prelude::*;
pub type InputType = [M31; 3];
pub type PackedInputType = [PackedM31; 3];

pub type CudaPackedInputType = [BaseFieldVec; 3];
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}

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

    pub fn add_cuda_inputs(&self, _cuda_inputs: &[CudaPackedInputType]) {
        // TODO: Implement CUDA kernel for verify_bitwise_xor_8_b_mults_init
        // let inputs_vec = cuda_inputs.iter()
        //     .flat_map(|row| row.iter().map(|x| x.device_ptr))
        //     .collect_vec();

        // unsafe {
        //     bindings_airs::verify_bitwise_xor_8_b_mults_init(
        //         inputs_vec.as_ptr(),
        //         cuda_inputs.len() as u32,
        //         cuda_inputs[0][0].size as u32,
        //         self.mults.device_ptr,
        //         LOG_SIZE,
        //     );
        // }
    }
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(mults: Uint32Vec) -> (CudaComponentTrace<N_TRACE_COLUMNS>, CudaLookupData) {
    let log_size = LOG_SIZE;
    let (trace, lookup_data) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(LOG_SIZE),
            CudaLookupData{
                verify_bitwise_xor_8_0: init_lookup_array!(LOG_SIZE),
                mults: BaseFieldVec::uninitialized(1 << LOG_SIZE),
            },
        )
    };

    let bitwisexor_8_0 = BitwiseXor::new(8, 0);
    let bitwisexor_8_1 = BitwiseXor::new(8, 1);
    let bitwisexor_8_2 = BitwiseXor::new(8, 2);

    (trace, lookup_data)
}

struct CudaLookupData {
    verify_bitwise_xor_8_0: [BaseFieldVec; 3],
    mults: BaseFieldVec,
}

pub struct InteractionClaimGenerator {
    lookup_data: CudaLookupData,
}