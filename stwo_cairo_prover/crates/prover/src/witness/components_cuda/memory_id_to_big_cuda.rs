use cairo_air::components::memory_id_to_big::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint128Vec, Uint32Vec};
use stwo::stwo_cuda::{bindings, bindings_airs};
use stwo_cairo_adapter::memory::{EncodedMemoryValueId, Memory, MemoryValueId};
use stwo_cairo_common::memory::{MEMORY_ADDRESS_BOUND, N_M31_IN_SMALL_FELT252};

/// Number of M31 elements in a Felt252 value (28 limbs of 9 bits each).
const N_M31_IN_FELT252: usize = 28;
use stwo_cairo_common::prover_types::simd::PackedFelt252;

use crate::witness::prelude::*;
use crate::witness::utils::TreeBuilder;

pub type InputType = M31;
pub type PackedInputType = PackedM31;
pub type CudaPackedInputType = [BaseFieldVec; 1];

pub struct CudaClaimGenerator {
    pub transposed_big_values: [Uint32Vec; 8],
    pub big_mults: Uint32Vec,
    pub small_values: Uint128Vec,
    pub small_mults: Uint32Vec,
}
impl CudaClaimGenerator {
    pub fn new(mem: &Memory) -> Self {
        let mut big_values: Vec<[u32; 8]> = mem.f252_values.clone();
        let big_size = std::cmp::max(big_values.len().next_power_of_two(), N_LANES);
        big_values.resize(big_size, [0; 8]);
        let rows = big_values.len();
        let mut transposed_big_values: Vec<Vec<u32>> = vec![vec![0u32; rows]; 8];
        for (i, row) in big_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                transposed_big_values[j][i] = val;
            }
        }
        let big_values_cuda =
            std::array::from_fn(|i| Uint32Vec::from_vec(transposed_big_values[i].clone()));
        let big_mults_cuda = Uint32Vec::new_zeroes(big_size);

        let mut small_values = mem.small_values.clone();
        let small_size = std::cmp::max(small_values.len().next_power_of_two(), N_LANES);
        small_values.resize(small_size, 0);
        let small_values_cuda = Uint128Vec::from_vec(small_values);
        let small_mults_cuda = Uint32Vec::new_zeroes(small_size);
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

        let transposed_big_values_vec = self
            .transposed_big_values
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
        let scalar_ids = ids.to_array();

        let mut all_outputs = Vec::with_capacity(N_LANES);
        for &id in &scalar_ids {
            let output = self.deduce_output_finese_cuda(id);
            all_outputs.push(output);
        }

        let mut element_wise: Vec<Vec<M31>> = vec![Vec::with_capacity(N_LANES); N_M31_IN_FELT252];
        for id_output in all_outputs {
            for (element_idx, &m) in id_output.iter().enumerate() {
                element_wise[element_idx].push(m);
            }
        }

        let packed_m31s: [PackedM31; N_M31_IN_FELT252] = element_wise
            .into_iter()
            .map(|elements| {
                let arr: [M31; N_LANES] = elements
                    .try_into()
                    .unwrap_or_else(|_| panic!("Element count mismatch, expected {}", N_LANES));
                PackedM31::from_array(arr)
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Final array size should match N_M31_IN_FELT252");

        PackedFelt252 { value: packed_m31s }
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
        let inputs_vec = cuda_inputs
            .iter()
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

    pub fn get_multiplicities_sum(&self) -> (u64, u64) {
        let big_mults = self.big_mults.to_vec();
        let small_mults = self.small_mults.to_vec();
        let big_sum: u64 = big_mults.iter().map(|&x| x as u64).sum();
        let small_sum: u64 = small_mults.iter().map(|&x| x as u64).sum();
        (big_sum, small_sum)
    }

    pub fn merge_simd_multiplicities(&mut self, simd_big_mults: &[u32], simd_small_mults: &[u32]) {
        // Upload SIMD mults to GPU and add in-place — no download/upload roundtrip.
        let gpu_big = Uint32Vec::from_vec(simd_big_mults.to_vec());
        self.big_mults.add_from(&gpu_big);

        let gpu_small = Uint32Vec::from_vec(simd_small_mults.to_vec());
        self.small_mults.add_from(&gpu_small);
    }

    pub fn write_trace_cuda(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        range_check_9_9_trace_generator: &super::range_check::rc_9_9::CudaClaimGenerator,
        log_max_big_size: u32,
    ) -> (Claim, CudaInteractionClaimGeneratorCuda) {
        let big_size = self.big_mults.size;
        let small_size = self.small_mults.size;

        let max_big_size = 1usize << log_max_big_size;
        let n_big_tables = (big_size + max_big_size - 1) / max_big_size;
        let table_size = max_big_size;

        let mut big_value_columns_all: Vec<[BaseFieldVec; N_M31_IN_FELT252]> =
            Vec::with_capacity(n_big_tables);
        let mut big_multiplicities_all: Vec<Uint32Vec> = Vec::with_capacity(n_big_tables);
        let mut big_log_sizes: Vec<u32> = Vec::with_capacity(n_big_tables);

        for table_idx in 0..n_big_tables {
            let offset = table_idx * table_size;
            let n_values = std::cmp::min(table_size, big_size - offset);
            let current_table_size = n_values.next_power_of_two();
            let current_log_size = current_table_size.ilog2();

            let trace_columns: Vec<Col<CudaBackend, BaseField>> = (0..N_M31_IN_FELT252 + 1)
                .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(current_table_size) })
                .collect();

            let transposed_ptrs: Vec<*const u32> = self
                .transposed_big_values
                .iter()
                .map(|v| unsafe { v.device_ptr.add(offset) })
                .collect();

            let trace_column_ptrs: Vec<*const u32> =
                trace_columns.iter().map(|col| col.device_ptr).collect();

            unsafe {
                bindings_airs::memory_id_to_big_generate_big_trace(
                    transposed_ptrs.as_ptr(),
                    self.big_mults.device_ptr.add(offset),
                    current_table_size as u32,
                    n_values as u32,
                    trace_column_ptrs.as_ptr(),
                );
            }

            let value_columns: [BaseFieldVec; N_M31_IN_FELT252] =
                std::array::from_fn(|i| trace_columns[i].clone());

            let mults_for_interaction = {
                let buffer = Uint32Vec::new_zeroes(current_table_size);
                unsafe {
                    bindings::copy_uint32_t_vec_from_device_to_device(
                        self.big_mults.device_ptr.add(offset) as *mut u32,
                        buffer.device_ptr as *mut u32,
                        n_values as u32,
                    );
                }
                buffer
            };

            let domain = CanonicCoset::new(current_log_size).circle_domain();
            let trace: Vec<_> = trace_columns
                .into_iter()
                .map(|col| {
                    CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
                })
                .collect();
            tree_builder.extend_evals(trace);

            // Add range check inputs from value columns (on GPU).
            // Each pair maps to relation_index = pair_idx % 8.
            for (pair_idx, (col0, col1)) in value_columns.iter().tuples().enumerate() {
                let input = [col0.clone(), col1.clone()];
                range_check_9_9_trace_generator
                    .add_cuda_inputs_for_relation(&[input], pair_idx % 8);
            }

            big_value_columns_all.push(value_columns);
            big_multiplicities_all.push(mults_for_interaction);
            big_log_sizes.push(current_log_size);
        }

        // Process small table.
        let n_small_values = small_size;
        let small_table_size = small_size.next_power_of_two();
        let small_log_size = small_table_size.ilog2();

        let small_trace_columns: Vec<Col<CudaBackend, BaseField>> = (0..N_M31_IN_SMALL_FELT252 + 1)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(small_table_size) })
            .collect();

        let small_trace_ptrs: Vec<*const u32> = small_trace_columns
            .iter()
            .map(|col| col.device_ptr)
            .collect();

        unsafe {
            bindings_airs::memory_id_to_big_generate_small_trace(
                self.small_values.device_ptr as *const u128,
                self.small_mults.device_ptr,
                small_table_size as u32,
                n_small_values as u32,
                small_trace_ptrs.as_ptr(),
            );
        }

        let small_value_columns: [BaseFieldVec; N_M31_IN_SMALL_FELT252] =
            std::array::from_fn(|i| small_trace_columns[i].clone());

        let small_mults_for_interaction = {
            let buffer = Uint32Vec::new_zeroes(small_table_size);
            unsafe {
                bindings::copy_uint32_t_vec_from_device_to_device(
                    self.small_mults.device_ptr as *mut u32,
                    buffer.device_ptr as *mut u32,
                    n_small_values as u32,
                );
            }
            buffer
        };

        let small_domain = CanonicCoset::new(small_log_size).circle_domain();
        let small_trace: Vec<_> = small_trace_columns
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(small_domain, col)
            })
            .collect();
        tree_builder.extend_evals(small_trace);

        // Add range check inputs for small table.
        // Each pair maps to relation_index = pair_idx % 4.
        for (pair_idx, (col0, col1)) in small_value_columns.iter().tuples().enumerate() {
            let input = [col0.clone(), col1.clone()];
            range_check_9_9_trace_generator.add_cuda_inputs_for_relation(&[input], pair_idx % 4);
        }

        let claim = Claim {
            big_log_sizes: big_log_sizes.clone(),
            small_log_size,
        };

        (
            claim,
            CudaInteractionClaimGeneratorCuda {
                big_value_columns: big_value_columns_all,
                big_multiplicities: big_multiplicities_all,
                big_log_sizes,
                small_value_columns,
                small_multiplicities: small_mults_for_interaction,
                small_log_size,
            },
        )
    }
}

pub struct CudaInteractionClaimGeneratorCuda {
    pub big_value_columns: Vec<[BaseFieldVec; N_M31_IN_FELT252]>,
    pub big_multiplicities: Vec<Uint32Vec>,
    pub big_log_sizes: Vec<u32>,
    pub small_value_columns: [BaseFieldVec; N_M31_IN_SMALL_FELT252],
    pub small_multiplicities: Uint32Vec,
    pub small_log_size: u32,
}

const N_BIG_INTERACTION_COLS: usize = 32;
const N_SMALL_INTERACTION_COLS: usize = 12;

impl CudaInteractionClaimGeneratorCuda {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        use stwo_cairo_common::memory::LARGE_MEMORY_VALUE_ID_BASE;

        let mut offset = 0u32;
        let mut big_claimed_sums: Vec<QM31> = Vec::with_capacity(self.big_value_columns.len());

        use super::cuda_lookup_helper::*;

        // Create modified lookup elements with relation constants baked in.
        let modified_mem =
            create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
        let modified_rc99: [_; 8] = std::array::from_fn(|i| {
            create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[i])
        });

        let mem_ptr = &modified_mem as *const _ as *mut std::os::raw::c_void;
        let rc99_ptrs: [*mut std::os::raw::c_void; 8] =
            std::array::from_fn(|i| &modified_rc99[i] as *const _ as *mut std::os::raw::c_void);

        for (table_idx, (value_columns, multiplicities)) in self
            .big_value_columns
            .iter()
            .zip(self.big_multiplicities.iter())
            .enumerate()
        {
            let log_size = self.big_log_sizes[table_idx];
            let trace_size = 1usize << log_size;

            let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..N_BIG_INTERACTION_COLS)
                .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(trace_size) })
                .collect();

            let cuda_claimed_sum: Col<CudaBackend, BaseField> =
                unsafe { Col::<CudaBackend, BaseField>::uninitialized(4) };

            let value_column_ptrs: Vec<*const u32> =
                value_columns.iter().map(|col| col.device_ptr).collect();

            let interaction_trace_ptrs: Vec<*const u32> =
                interaction_trace.iter().map(|col| col.device_ptr).collect();

            unsafe {
                bindings_airs::memory_id_to_big_generate_big_interaction_trace(
                    mem_ptr,      // memory_id_to_big
                    rc99_ptrs[0], // range_check_9_9
                    rc99_ptrs[1], // range_check_9_9_b
                    rc99_ptrs[2], // range_check_9_9_c
                    rc99_ptrs[3], // range_check_9_9_d
                    rc99_ptrs[4], // range_check_9_9_e
                    rc99_ptrs[5], // range_check_9_9_f
                    rc99_ptrs[6], // range_check_9_9_g
                    rc99_ptrs[7], // range_check_9_9_h
                    value_column_ptrs.as_ptr(),
                    multiplicities.device_ptr,
                    trace_size as u32,
                    offset | LARGE_MEMORY_VALUE_ID_BASE,
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
            big_claimed_sums.push(claimed_sum);

            let domain = CanonicCoset::new(log_size).circle_domain();
            let trace: Vec<_> = interaction_trace
                .into_iter()
                .map(|col| {
                    CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
                })
                .collect();
            tree_builder.extend_evals(trace);

            offset += trace_size as u32;
        }

        // Small table interaction trace.
        let small_trace_size = 1usize << self.small_log_size;

        let small_interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0
            ..N_SMALL_INTERACTION_COLS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(small_trace_size) })
            .collect();

        let small_cuda_claimed_sum: Col<CudaBackend, BaseField> =
            unsafe { Col::<CudaBackend, BaseField>::uninitialized(4) };

        let small_value_column_ptrs: Vec<*const u32> = self
            .small_value_columns
            .iter()
            .map(|col| col.device_ptr)
            .collect();

        let small_interaction_trace_ptrs: Vec<*const u32> = small_interaction_trace
            .iter()
            .map(|col| col.device_ptr)
            .collect();

        unsafe {
            bindings_airs::memory_id_to_big_generate_small_interaction_trace(
                mem_ptr,      // memory_id_to_big
                rc99_ptrs[0], // range_check_9_9
                rc99_ptrs[1], // range_check_9_9_b
                rc99_ptrs[2], // range_check_9_9_c
                rc99_ptrs[3], // range_check_9_9_d
                small_value_column_ptrs.as_ptr(),
                self.small_multiplicities.device_ptr,
                small_trace_size as u32,
                small_interaction_trace_ptrs.as_ptr(),
                small_cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        let small_claimed_sum_vec = small_cuda_claimed_sum.to_cpu();
        let small_claimed_sum = SecureField::from_m31_array([
            small_claimed_sum_vec[0],
            small_claimed_sum_vec[1],
            small_claimed_sum_vec[2],
            small_claimed_sum_vec[3],
        ]);

        let small_domain = CanonicCoset::new(self.small_log_size).circle_domain();
        let small_trace: Vec<_> = small_interaction_trace
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(small_domain, col)
            })
            .collect();
        tree_builder.extend_evals(small_trace);

        let claimed_sum = small_claimed_sum + big_claimed_sums.iter().copied().sum::<SecureField>();
        InteractionClaim {
            small_claimed_sum,
            big_claimed_sums,
            claimed_sum,
        }
    }
}
