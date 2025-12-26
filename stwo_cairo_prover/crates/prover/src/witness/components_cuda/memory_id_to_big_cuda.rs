use std::simd::Simd;

use cairo_air::components::memory_id_to_big::{InteractionClaim, MEMORY_ID_SIZE};
use cairo_air::relations;
use itertools::Itertools;
use stwo_cairo_common::preprocessed_columns::preprocessed_utils::SIMD_ENUMERATION_0;
use rayon::iter::{
    IntoParallelIterator, ParallelIterator,
};
use stwo_cairo_adapter::memory::{
    Memory, LARGE_MEMORY_VALUE_ID_BASE,
};
use stwo_cairo_common::memory::{MEMORY_ADDRESS_BOUND, N_M31_IN_FELT252, N_M31_IN_SMALL_FELT252};
use stwo_cairo_common::prover_types::simd::PackedFelt252;
use stwo::stwo_cuda::bindings_airs;

use crate::witness::prelude::*;
use crate::witness::utils::TreeBuilder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec, Uint128Vec};

use stwo_cairo_adapter::memory::EncodedMemoryValueId;
use stwo_cairo_adapter::memory::MemoryValueId;

pub type InputType = M31;
pub type PackedInputType = PackedM31;
pub type CudaPackedInputType = [BaseFieldVec; 1];


/// Generates the trace and the claim for the id -> f252 memory table.
/// Generates 2 table, one for large values and one for small values. A large value is a full 28
/// limb Felt252. The small values are currently 8 limbs, for a maximum of 72 bits.
/// The separation is done to reduce zeroed out ('unused') trace cells.
pub struct CudaClaimGenerator {
    pub transposed_big_values: [Uint32Vec; 8],
    pub big_mults: Uint32Vec,
    pub small_values: Uint128Vec,
    pub small_mults: Uint32Vec,
}
impl CudaClaimGenerator {
    pub fn new(mem: &Memory) -> Self {
        // TODO(spapini): More repetitions, for efficiency.
        let mut big_values: Vec<[u32; 8]> = mem.f252_values.clone();
        let big_size = std::cmp::max(big_values.len().next_power_of_two(), N_LANES);
        big_values.resize(big_size, [0; 8]);
        // do transpose the big values.
        let rows = big_values.len();
        let mut transposed_big_values: Vec<Vec<u32>> = vec![vec![0u32; rows]; 8];
        for (i, row) in big_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                transposed_big_values[j][i] = val;
            }
        }
        let big_values_cuda = std::array::from_fn(|i| {
            Uint32Vec::from_vec(transposed_big_values[i].clone())
        });
        let big_mults_cuda = Uint32Vec::new_uninitialized(big_size);

        let mut small_values = mem.small_values.clone();
        let small_size = std::cmp::max(small_values.len().next_power_of_two(), N_LANES);
        small_values.resize(small_size, 0);
        let small_values_cuda = Uint128Vec::from_vec(small_values);
        let small_mults_cuda = Uint32Vec::new_uninitialized(small_size);
        assert!(
            big_size + small_size <= MEMORY_ADDRESS_BOUND,
            "Assertion failed, condition `big_size ({big_size}) + small_size ({small_size}) <= \
            MEMORY_ADDRESS_BOUND ({MEMORY_ADDRESS_BOUND})` is not satisfied."
        );

        Self {
            transposed_big_values: big_values_cuda,
            big_mults: big_mults_cuda,
            small_values: small_values_cuda,
            small_mults: small_mults_cuda,
        }
    }

    fn deduce_output_finese_cuda(&self, id: M31) -> [M31; N_M31_IN_FELT252] {
        let felt252 = BaseFieldVec::new_zeroes(N_M31_IN_FELT252);

        let transposed_big_values_vec = self.transposed_big_values
            .iter()
            .map(|x| x.device_ptr)
            .collect_vec();
        unsafe {
            bindings_airs::memory_id_to_big_deduce_finese_cuda(
                transposed_big_values_vec.as_ptr(),
                self.small_values.device_ptr,
                id.0,
                felt252.device_ptr,
            );
        }

        felt252.to_vec().try_into().unwrap()
    }

    pub fn deduce_output_packed(&self, ids: PackedM31) -> PackedFelt252 {
        // to m31 arrat
        let scalar_ids = ids.to_array();

        // for each id generate Felt252 value
        let mut all_outputs = Vec::with_capacity(N_LANES);
        for &id in &scalar_ids {
            let output = self.deduce_output_finese_cuda(id);
            all_outputs.push(output);
        }

        // transpose [id][element] -> [element][id]
        let mut element_wise: Vec<Vec<M31>> = vec![Vec::with_capacity(N_LANES); N_M31_IN_FELT252];
        for id_output in all_outputs {
            for (element_idx, &m) in id_output.iter().enumerate() {
                element_wise[element_idx].push(m);
            }
        }

        // to packed m31 arrat
        let packed_m31s: [PackedM31; N_M31_IN_FELT252] = element_wise
            .into_iter()
            .map(|elements| {
                // make sure all elements are the same length
                let arr: [M31; N_LANES] = elements.try_into()
                    .unwrap_or_else(|_| panic!("Element count mismatch, expected {}", N_LANES));
                PackedM31::from_array(arr)
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Final array size should match N_M31_IN_FELT252");

        PackedFelt252 {
            value: packed_m31s
        }
    }

    pub fn add_cuda_input(&self, encoded_memory_id: &M31) {
        match EncodedMemoryValueId(encoded_memory_id.0).decode() {
            MemoryValueId::F252(id) => {
                self.big_mults.increase_at(id);
            }
            MemoryValueId::Small(id) => {
                self.small_mults.increase_at(id);
            }
            MemoryValueId::Empty => panic!("Attempted add_input on empty memory cell."),
        }
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        let inputs_vec = cuda_inputs.iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();
        unsafe {
            bindings_airs::memory_id_to_big_add_inputs(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                self.big_mults.device_ptr,
                self.big_mults.size.ilog2(),
                self.small_mults.device_ptr,
                self.small_mults.size.ilog2(),
            );
        }
    }

}


#[derive(Debug)]
pub struct CudaInteractionClaimGenerator {
    pub big_values: [Vec<PackedM31>; N_M31_IN_FELT252],
    pub big_multiplicities: Vec<PackedM31>,
    pub small_values: [Vec<PackedM31>; N_M31_IN_SMALL_FELT252],
    pub small_multiplicities: Vec<PackedM31>,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<SimdBackend>,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
    ) -> InteractionClaim {
        let (big_trace, big_claimed_sum) =
            self.gen_big_memory_interaction_trace(lookup_elements, range9_9_lookup_elements);
        tree_builder.extend_evals(big_trace);

        let (small_trace, small_claimed_sum) =
            self.gen_small_memory_interaction_trace(lookup_elements, range9_9_lookup_elements);
        tree_builder.extend_evals(small_trace);

        InteractionClaim {
            small_claimed_sum,
            big_claimed_sums: vec![big_claimed_sum],
        }
    }

    fn gen_big_memory_interaction_trace(
        &self,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
    ) -> (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        QM31,
    ) {
        let big_table_log_size = self.big_values[0].len().ilog2() + LOG_N_LANES;
        let mut big_values_logup_gen = LogupTraceGenerator::new(big_table_log_size);

        // Every element is 9-bit.
        for (limb0, limb1, limb2, lim3) in self.big_values.iter().tuples() {
            let mut col_gen = big_values_logup_gen.new_col();
            (col_gen.par_iter_mut(), limb0, limb1, limb2, lim3)
                .into_par_iter()
                .for_each(|(writer, limb0, limb1, limb2, limb3)| {
                    let denom0: PackedQM31 = range9_9_lookup_elements.combine(&[*limb0, *limb1]);
                    let denom1: PackedQM31 = range9_9_lookup_elements.combine(&[*limb2, *limb3]);
                    writer.write_frac(denom0 + denom1, denom0 * denom1);
                });
            col_gen.finalize_col();
        }

        // Yield large values.
        let mut col_gen = big_values_logup_gen.new_col();
        let large_memory_value_id_tag = Simd::splat(LARGE_MEMORY_VALUE_ID_BASE);
        for vec_row in 0..1 << (big_table_log_size - LOG_N_LANES) {
            let id_and_value: [_; N_M31_IN_FELT252 + MEMORY_ID_SIZE] = std::array::from_fn(|i| {
                if i == 0 {
                    unsafe {
                        PackedM31::from_simd_unchecked(
                            (SIMD_ENUMERATION_0 + Simd::splat((vec_row * N_LANES) as u32))
                                | large_memory_value_id_tag,
                        )
                    }
                } else {
                    self.big_values[i - 1][vec_row]
                }
            });
            let denom: PackedQM31 = lookup_elements.combine(&id_and_value);
            col_gen.write_frac(vec_row, (-self.big_multiplicities[vec_row]).into(), denom);
        }
        col_gen.finalize_col();

        big_values_logup_gen.finalize_last()
    }

    fn gen_small_memory_interaction_trace(
        &self,
        lookup_elements: &relations::MemoryIdToBig,
        range9_9_lookup_elements: &relations::RangeCheck_9_9,
    ) -> (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        QM31,
    ) {
        let small_table_log_size = self.small_values[0].len().ilog2() + LOG_N_LANES;
        let mut small_values_logup_gen = LogupTraceGenerator::new(small_table_log_size);

        // Every element is 9-bit.
        for (l, r) in self.small_values.iter().tuples() {
            let mut col_gen = small_values_logup_gen.new_col();
            (col_gen.par_iter_mut(), l, r)
                .into_par_iter()
                .for_each(|(writer, l1, l2)| {
                    // TOOD(alont) Add 2-batching.
                    writer.write_frac(
                        PackedQM31::broadcast(M31(1).into()),
                        range9_9_lookup_elements.combine(&[*l1, *l2]),
                    );
                });
            col_gen.finalize_col();
        }

        // Yield small values.
        let mut col_gen = small_values_logup_gen.new_col();
        for vec_row in 0..1 << (small_table_log_size - LOG_N_LANES) {
            let id_and_value: [_; N_M31_IN_SMALL_FELT252 + MEMORY_ID_SIZE] =
                std::array::from_fn(|i| {
                    if i == 0 {
                        unsafe {
                            PackedM31::from_simd_unchecked(
                                SIMD_ENUMERATION_0 + Simd::splat((vec_row * N_LANES) as u32),
                            )
                        }
                    } else {
                        self.small_values[i - 1][vec_row]
                    }
                });
            let denom: PackedQM31 = lookup_elements.combine(&id_and_value);
            col_gen.write_frac(vec_row, (-self.small_multiplicities[vec_row]).into(), denom);
        }
        col_gen.finalize_col();

        small_values_logup_gen.finalize_last()
    }
}