#![allow(unused_parens)]
// use stwo_cairo_adapter::decode::deconstruct_instruction;
use stwo_cairo_adapter::HashMap;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint128Vec, Uint32Vec};

// use crate::witness::components::{
//     memory_address_to_id, memory_id_to_big, range_check_4_3, range_check_7_2_5,
// };
use crate::witness::prelude::*;

pub type InputType = (M31, [M31; 3], [M31; 2], M31);
pub type CudaPackedInputType = [BaseFieldVec; 7];

pub struct CudaClaimGenerator {
    /// pc -> encoded instruction.
    _instructions: (BaseFieldVec, Uint128Vec),

    /// pc -> multiplicity.
    _multiplicities: (BaseFieldVec, Uint32Vec),
}
impl CudaClaimGenerator {
    pub fn new(instructions: Vec<(u32, u128)>) -> Self {
        let instructions_map: HashMap<u32, u128> = HashMap::from_iter(instructions);
        let len = instructions_map.len();

        Self {
            _multiplicities: (
                BaseFieldVec::new_zeroes(len),
                Uint32Vec::new_zeroes(len),
            ),
            _instructions: (
                BaseFieldVec::new_zeroes(len),
                Uint128Vec::new_zeroes(len),
            ),
        }
    }

    pub fn add_cuda_inputs(&self, _packed_inputs: &[CudaPackedInputType]) {
        // _packed_inputs.into_par_iter().for_each(|packed_input| {
        //     packed_input.unpack().into_par_iter().for_each(|input| {
        //         self.add_input(&input);
        //     });
        // });
    }

    // Instruction is determined by PC.
    // pub fn add_input(&self, (pc, ..): &InputType) {
    //     self.multiplicities
    //         .get(pc)
    //         .unwrap()
    //         .fetch_add(1, Ordering::Relaxed);
    // }
}
