#![allow(unused_parens)]
use std::sync::atomic::{AtomicU32, Ordering};

use cairo_air::components::verify_instruction::{Claim, InteractionClaim, N_TRACE_COLUMNS};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::conversion::Unpack;
use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_cairo_adapter::decode::deconstruct_instruction;
use stwo_cairo_adapter::HashMap;

use super::{
    memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_4_3_cuda, range_check_7_2_5_cuda,
};
use crate::witness::utils::TreeBuilder;

const N_INTERACTION_TRACE_COLUMNS: usize = 3;

pub type InputType = (M31, [M31; 3], [M31; 2], M31);
pub type PackedInputType = (PackedM31, [PackedM31; 3], [PackedM31; 2], PackedM31);
pub type CudaPackedInputType = [BaseFieldVec; 7];

struct CudaLookupData {
    range_check_7_2_5_0: [BaseFieldVec; 3],
    range_check_4_3_0: [BaseFieldVec; 2],
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_id_to_big_0: [BaseFieldVec; 29],
    verify_instruction_0: [BaseFieldVec; 7],
    multiplicities: BaseFieldVec,
}

pub struct CudaClaimGenerator {
    instructions: HashMap<u32, u128>,
    multiplicities: HashMap<u32, AtomicU32>,
}

impl CudaClaimGenerator {
    pub fn new(instructions: Vec<(u32, u128)>) -> Self {
        let instructions_map = HashMap::from_iter(instructions);
        let keys = instructions_map.keys().copied();
        let mut multiplicities = HashMap::with_capacity(keys.len());
        multiplicities.extend(keys.zip(std::iter::repeat_with(|| AtomicU32::new(0))));

        Self {
            instructions: instructions_map,
            multiplicities,
        }
    }

    pub fn add_input(&self, (pc, ..): &InputType) {
        self.multiplicities
            .get(&pc.0)
            .unwrap()
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_packed_inputs(&self, packed_inputs: &[PackedInputType]) {
        packed_inputs.into_par_iter().for_each(|packed_input| {
            packed_input.unpack().into_par_iter().for_each(|input| {
                self.add_input(&input);
            });
        });
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        for input in cuda_inputs {
            let pc_vec = &input[0];
            let pc_values = pc_vec.to_cpu();
            for pc in pc_values {
                if let Some(mult) = self.multiplicities.get(&pc.0) {
                    mult.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_4_3_cuda_state: &range_check_4_3_cuda::CudaClaimGenerator,
        range_check_7_2_5_cuda_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let (mut inputs, mut mults) = self
            .multiplicities
            .into_iter()
            .sorted_by_key(|(pc, _)| *pc)
            .map(|(pc, multiplicity)| {
                let (offsets, flags, opcode_extension) =
                    deconstruct_instruction(*self.instructions.get(&pc).unwrap());
                let multiplicity = M31(multiplicity.into_inner());
                ((M31(pc), offsets, flags, opcode_extension), multiplicity)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let n_rows = inputs.len();
        assert_ne!(n_rows, 0, "verify_instruction must have at least one row");

        let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
        let log_size = size.ilog2();
        let trace_size = 1usize << log_size;
        let need_padding = n_rows != size;

        if need_padding {
            inputs.resize(size, *inputs.first().unwrap());
            mults.resize(size, M31(0));
        }

        let verify_instruction_inputs: [BaseFieldVec; 7] = std::array::from_fn(|col_idx| {
            let values: Vec<M31> = inputs
                .iter()
                .map(|(pc, offsets, flags, opcode_ext)| match col_idx {
                    0 => *pc,
                    1 => offsets[0],
                    2 => offsets[1],
                    3 => offsets[2],
                    4 => flags[0],
                    5 => flags[1],
                    6 => *opcode_ext,
                    _ => unreachable!(),
                })
                .collect();
            BaseFieldVec::from_vec(values)
        });

        let multiplicities_gpu = BaseFieldVec::from_vec(mults);

        let trace_columns: [BaseFieldVec; N_TRACE_COLUMNS] =
            std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size));

        let lookup_data = CudaLookupData {
            range_check_7_2_5_0: std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size)),
            range_check_4_3_0: std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size)),
            memory_address_to_id_0: std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size)),
            memory_id_to_big_0: std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size)),
            verify_instruction_0: std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size)),
            multiplicities: multiplicities_gpu.clone(),
        };

        let sub_inputs_memory_address_to_id: [BaseFieldVec; 1] =
            std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size));
        let sub_inputs_memory_id_to_big: [BaseFieldVec; 1] =
            std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size));
        let sub_inputs_range_check_7_2_5: [BaseFieldVec; 3] =
            std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size));
        let sub_inputs_range_check_4_3: [BaseFieldVec; 2] =
            std::array::from_fn(|_| BaseFieldVec::new_zeroes(trace_size));

        let trace_ptrs: Vec<*const u32> = trace_columns.iter().map(|c| c.device_ptr).collect();
        let lookup_rc_7_2_5_ptrs: Vec<*const u32> = lookup_data
            .range_check_7_2_5_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_rc_4_3_ptrs: Vec<*const u32> = lookup_data
            .range_check_4_3_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_addr_to_id_ptrs: Vec<*const u32> = lookup_data
            .memory_address_to_id_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_id_to_big_ptrs: Vec<*const u32> = lookup_data
            .memory_id_to_big_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_verify_instr_ptrs: Vec<*const u32> = lookup_data
            .verify_instruction_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let sub_addr_to_id_ptrs: Vec<*const u32> = sub_inputs_memory_address_to_id
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let sub_id_to_big_ptrs: Vec<*const u32> = sub_inputs_memory_id_to_big
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let sub_rc_7_2_5_ptrs: Vec<*const u32> = sub_inputs_range_check_7_2_5
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let sub_rc_4_3_ptrs: Vec<*const u32> = sub_inputs_range_check_4_3
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let input_ptrs: Vec<*const u32> = verify_instruction_inputs
            .iter()
            .map(|c| c.device_ptr)
            .collect();

        let addr_to_id_ptr = memory_address_to_id_cuda_state.address_to_raw_id.device_ptr;

        unsafe {
            bindings_airs::generate_verify_instruction_trace(
                trace_ptrs.as_ptr(),
                lookup_addr_to_id_ptrs.as_ptr(),
                lookup_id_to_big_ptrs.as_ptr(),
                lookup_rc_7_2_5_ptrs.as_ptr(),
                lookup_rc_4_3_ptrs.as_ptr(),
                lookup_verify_instr_ptrs.as_ptr(),
                sub_addr_to_id_ptrs.as_ptr(),
                sub_id_to_big_ptrs.as_ptr(),
                sub_rc_7_2_5_ptrs.as_ptr(),
                sub_rc_4_3_ptrs.as_ptr(),
                input_ptrs.as_ptr(),
                multiplicities_gpu.device_ptr,
                addr_to_id_ptr,
                size as u32,
                log_size,
            );
        }

        // Route sub-component inputs to CUDA generators (stay on GPU).
        memory_address_to_id_cuda_state.add_cuda_inputs(&[sub_inputs_memory_address_to_id]);
        memory_id_to_big_cuda_state.add_cuda_inputs(&[sub_inputs_memory_id_to_big]);

        range_check_7_2_5_cuda_state.add_cuda_inputs(&[sub_inputs_range_check_7_2_5]);

        range_check_4_3_cuda_state.add_cuda_inputs(&[sub_inputs_range_check_4_3]);

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
                log_size,
                lookup_data,
            },
        )
    }
}

pub struct CudaInteractionClaimGenerator {
    log_size: u32,
    lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_size = 1usize << self.log_size;

        let interaction_trace: Vec<BaseFieldVec> = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| BaseFieldVec::new_zeroes(trace_size))
            .collect();

        let claimed_sum_gpu = BaseFieldVec::new_zeroes(4);

        let lookup_rc_7_2_5_ptrs: Vec<*const u32> = self
            .lookup_data
            .range_check_7_2_5_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_rc_4_3_ptrs: Vec<*const u32> = self
            .lookup_data
            .range_check_4_3_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_addr_to_id_ptrs: Vec<*const u32> = self
            .lookup_data
            .memory_address_to_id_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_id_to_big_ptrs: Vec<*const u32> = self
            .lookup_data
            .memory_id_to_big_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();
        let lookup_verify_instr_ptrs: Vec<*const u32> = self
            .lookup_data
            .verify_instruction_0
            .iter()
            .map(|c| c.device_ptr)
            .collect();

        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|c| c.device_ptr).collect();

        unsafe {
            use super::cuda_lookup_helper::*;
            let mod_rc725 = create_modified_lookup_for_cuda(lookup_elements, RC_7_2_5_RELATION_ID);
            let mod_rc43 = create_modified_lookup_for_cuda(lookup_elements, RC_4_3_RELATION_ID);
            let mod_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
            let mod_vi =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_INSTRUCTION_RELATION_ID);

            bindings_airs::generate_verify_instruction_interaction_trace(
                interaction_trace_ptrs.as_ptr(),
                lookup_rc_7_2_5_ptrs.as_ptr(),
                lookup_rc_4_3_ptrs.as_ptr(),
                lookup_addr_to_id_ptrs.as_ptr(),
                lookup_id_to_big_ptrs.as_ptr(),
                lookup_verify_instr_ptrs.as_ptr(),
                self.lookup_data.multiplicities.device_ptr,
                &mod_rc725 as *const _ as *mut std::os::raw::c_void,
                &mod_rc43 as *const _ as *mut std::os::raw::c_void,
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_vi as *const _ as *mut std::os::raw::c_void,
                self.log_size,
                claimed_sum_gpu.device_ptr as *mut u32,
            );
        }

        let claimed_sum_cpu = claimed_sum_gpu.to_cpu();
        let claimed_sum = SecureField::from_m31_array(std::array::from_fn(|i| claimed_sum_cpu[i]));

        let domain = CanonicCoset::new(self.log_size).circle_domain();
        let cuda_evals: Vec<_> = interaction_trace
            .into_iter()
            .map(|col| {
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(domain, col)
            })
            .collect();

        tree_builder.extend_evals(cuda_evals);

        InteractionClaim { claimed_sum }
    }
}
