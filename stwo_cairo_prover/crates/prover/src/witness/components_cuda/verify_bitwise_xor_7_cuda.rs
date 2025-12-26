#![allow(unused_parens)]
#![allow(dead_code)]
use cairo_air::components::verify_bitwise_xor_7::{
    InteractionClaim, LOG_SIZE,  N_TRACE_COLUMNS,
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
        let inputs_vec = cuda_inputs.iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        unsafe {
            bindings_airs::verify_bitwise_xor_7_mults_init(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                self.mults.device_ptr,
                LOG_SIZE,
            );
        }
    }
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_simd(mults: Vec<PackedM31>) -> (ComponentTrace<N_TRACE_COLUMNS>, CudaLookupData) {
    let log_n_packed_rows = LOG_SIZE - LOG_N_LANES;
    let (mut trace, mut lookup_data) = unsafe {
        (
            ComponentTrace::<N_TRACE_COLUMNS>::uninitialized(LOG_SIZE),
            CudaLookupData::uninitialized(log_n_packed_rows),
        )
    };

    let bitwisexor_7_0 = BitwiseXor::new(7, 0);
    let bitwisexor_7_1 = BitwiseXor::new(7, 1);
    let bitwisexor_7_2 = BitwiseXor::new(7, 2);

    (trace.par_iter_mut(), lookup_data.par_iter_mut())
        .into_par_iter()
        .enumerate()
        .for_each(|(row_index, (mut row, lookup_data))| {
            let bitwisexor_7_0 = bitwisexor_7_0.packed_at(row_index);
            let bitwisexor_7_1 = bitwisexor_7_1.packed_at(row_index);
            let bitwisexor_7_2 = bitwisexor_7_2.packed_at(row_index);
            *lookup_data.verify_bitwise_xor_7_0 = [bitwisexor_7_0, bitwisexor_7_1, bitwisexor_7_2];
            let mult_at_row = *mults.get(row_index).unwrap_or(&PackedM31::zero());
            *row[0] = mult_at_row;
            *lookup_data.mults = mult_at_row;
        });

    (trace, lookup_data)
}

#[derive(Uninitialized, IterMut, ParIterMut)]
struct CudaLookupData {
    verify_bitwise_xor_7_0: Vec<[PackedM31; 3]>,
    mults: Vec<PackedM31>,
}

pub struct CudaInteractionClaimGenerator {
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        verify_bitwise_xor_7: &relations::VerifyBitwiseXor_7,
    ) -> InteractionClaim {
        let mut logup_gen = LogupTraceGenerator::new(LOG_SIZE);

        // Sum last logup term.
        let mut col_gen = logup_gen.new_col();
        (
            col_gen.par_iter_mut(),
            &self.lookup_data.verify_bitwise_xor_7_0,
            self.lookup_data.mults,
        )
            .into_par_iter()
            .for_each(|(writer, values, mults)| {
                let denom = verify_bitwise_xor_7.combine(values);
                writer.write_frac(-PackedQM31::one() * mults, denom);
            });
        col_gen.finalize_col();

        let (trace, claimed_sum) = logup_gen.finalize_last();
        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
