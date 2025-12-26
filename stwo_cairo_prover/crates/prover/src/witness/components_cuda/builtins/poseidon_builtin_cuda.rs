// CUDA witness generation for poseidon_builtin component
// This handles Poseidon hash builtin operations

#![allow(unused_parens)]
use cairo_air::components::poseidon_builtin::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big};
use stwo::prover::backend::cuda::CudaBackend;

use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

pub const N_TRACE_COLUMNS: usize = 341;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 68;  // 17 logup columns × 4 M31

use itertools::Itertools;
use stwo::prover::backend::{Col, Column};
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}

macro_rules! init_subcomponent_basefield_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size)))
    };
}

macro_rules! collect_lookup_ptrs {
    ($lookup_data:expr, $field:ident) => {
        $lookup_data.$field.iter().map(|x| x.device_ptr).collect_vec()
    };
}


pub struct CudaClaimGenerator {
    pub log_size: u32,
    pub segment_start: u32,
}

impl CudaClaimGenerator {
    pub fn new(log_size: u32, segment_start: u32) -> Self {
        Self {
            log_size,
            segment_start,
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        // Also pass SIMD generators for multiplicity tracking
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        let n_rows = 1usize << log_size;

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            n_rows,
            log_size,
            self.segment_start,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
        );

        // Add to CUDA generators
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        let memory_size = memory_address_to_id_simd_state.memory_size();
        for input_arr in &sub_component_inputs.memory_address_to_id {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for addr in cpu_data.iter().take(n_rows) {
                if addr.0 != 0 && addr.0 <= memory_size {
                    memory_address_to_id_simd_state.add_input(addr);
                }
            }
        }
        for input_arr in &sub_component_inputs.memory_id_to_big {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for id in cpu_data.iter().take(n_rows) {
                if id.0 != 0 {
                    memory_id_to_big_simd_state.add_input(id);
                }
            }
        }

        // Extract base trace column pointers before converting to evals
        // These pointers remain valid because the data is moved to tree_builder
        let base_trace_ptrs: Vec<*const u32> = trace.data.iter()
            .map(|col| col.device_ptr)
            .collect();

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size, poseidon_builtin_segment_start: self.segment_start },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
                base_trace_ptrs,
            },
        )
    }
}

pub struct CudaSubComponentInputs {
    // 6 memory_address_to_id lookups
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 6],
    // 6 memory_id_to_big lookups
    pub memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 6],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    n_rows: usize,
    log_size: u32,
    segment_start: u32,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    Box<CudaLookupData>,
    CudaSubComponentInputs,
) {
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            Box::new(CudaLookupData {
                // 6 memory_address_to_id lookups (2 elements each)
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                memory_address_to_id_3: init_lookup_array!(log_size),
                memory_address_to_id_4: init_lookup_array!(log_size),
                memory_address_to_id_5: init_lookup_array!(log_size),
                // 6 memory_id_to_big lookups (29 elements each)
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                memory_id_to_big_3: init_lookup_array!(log_size),
                memory_id_to_big_4: init_lookup_array!(log_size),
                memory_id_to_big_5: init_lookup_array!(log_size),
                // 2 range_check_3_3_3_3_3 lookups (5 elements each)
                range_check_3_3_3_3_3_0: init_lookup_array!(log_size),
                range_check_3_3_3_3_3_1: init_lookup_array!(log_size),
                // 6 range_check_4_4_4_4 lookups (4 elements each)
                range_check_4_4_4_4_0: init_lookup_array!(log_size),
                range_check_4_4_4_4_1: init_lookup_array!(log_size),
                range_check_4_4_4_4_2: init_lookup_array!(log_size),
                range_check_4_4_4_4_3: init_lookup_array!(log_size),
                range_check_4_4_4_4_4: init_lookup_array!(log_size),
                range_check_4_4_4_4_5: init_lookup_array!(log_size),
                // 3 range_check_4_4 lookups (2 elements each)
                range_check_4_4_0: init_lookup_array!(log_size),
                range_check_4_4_1: init_lookup_array!(log_size),
                range_check_4_4_2: init_lookup_array!(log_size),
                // 2 poseidon_full_round_chain lookups (32 elements each)
                poseidon_full_round_chain_0: init_lookup_array!(log_size),
                poseidon_full_round_chain_1: init_lookup_array!(log_size),
                // Base trace columns 120-283 for interaction kernels (164 columns)
                base_trace_cols: init_lookup_array!(log_size),
            }),
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers - 6 memory_address_to_id
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_address_to_id_3 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_3);
    let lookup_memory_address_to_id_4 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_4);
    let lookup_memory_address_to_id_5 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_5);

    // Collect lookup data pointers - 6 memory_id_to_big
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_memory_id_to_big_3 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_3);
    let lookup_memory_id_to_big_4 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_4);
    let lookup_memory_id_to_big_5 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_5);

    // Collect lookup data pointers - 2 range_check_3_3_3_3_3
    let lookup_range_check_3_3_3_3_3_0 = collect_lookup_ptrs!(lookup_data, range_check_3_3_3_3_3_0);
    let lookup_range_check_3_3_3_3_3_1 = collect_lookup_ptrs!(lookup_data, range_check_3_3_3_3_3_1);

    // Collect lookup data pointers - 6 range_check_4_4_4_4
    let lookup_range_check_4_4_4_4_0 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_0);
    let lookup_range_check_4_4_4_4_1 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_1);
    let lookup_range_check_4_4_4_4_2 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_2);
    let lookup_range_check_4_4_4_4_3 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_3);
    let lookup_range_check_4_4_4_4_4 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_4);
    let lookup_range_check_4_4_4_4_5 = collect_lookup_ptrs!(lookup_data, range_check_4_4_4_4_5);

    // Collect lookup data pointers - 3 range_check_4_4
    let lookup_range_check_4_4_0 = collect_lookup_ptrs!(lookup_data, range_check_4_4_0);
    let lookup_range_check_4_4_1 = collect_lookup_ptrs!(lookup_data, range_check_4_4_1);
    let lookup_range_check_4_4_2 = collect_lookup_ptrs!(lookup_data, range_check_4_4_2);

    // Collect lookup data pointers - 2 poseidon_full_round_chain
    let lookup_poseidon_full_round_chain_0 = collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_0);
    let lookup_poseidon_full_round_chain_1 = collect_lookup_ptrs!(lookup_data, poseidon_full_round_chain_1);

    // Collect lookup data pointers - base_trace_cols (cols 120-283)
    let lookup_base_trace_cols = collect_lookup_ptrs!(lookup_data, base_trace_cols);

    // Transpose memory_id_to_big big_values for GPU access
    let memory_id_to_big_transposed_big_values_vec: Vec<_> = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect();

    // Call CUDA kernel
    unsafe {
        bindings_airs::gen_poseidon_builtin_trace(
            traces_vec.as_ptr(),
            log_size,
            segment_start,
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,

            // 6 memory_address_to_id lookups
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            lookup_memory_address_to_id_3.as_ptr(),
            lookup_memory_address_to_id_4.as_ptr(),
            lookup_memory_address_to_id_5.as_ptr(),

            // 6 memory_id_to_big lookups
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            lookup_memory_id_to_big_3.as_ptr(),
            lookup_memory_id_to_big_4.as_ptr(),
            lookup_memory_id_to_big_5.as_ptr(),

            // 2 range_check_3_3_3_3_3 lookups
            lookup_range_check_3_3_3_3_3_0.as_ptr(),
            lookup_range_check_3_3_3_3_3_1.as_ptr(),

            // 6 range_check_4_4_4_4 lookups
            lookup_range_check_4_4_4_4_0.as_ptr(),
            lookup_range_check_4_4_4_4_1.as_ptr(),
            lookup_range_check_4_4_4_4_2.as_ptr(),
            lookup_range_check_4_4_4_4_3.as_ptr(),
            lookup_range_check_4_4_4_4_4.as_ptr(),
            lookup_range_check_4_4_4_4_5.as_ptr(),

            // 3 range_check_4_4 lookups
            lookup_range_check_4_4_0.as_ptr(),
            lookup_range_check_4_4_1.as_ptr(),
            lookup_range_check_4_4_2.as_ptr(),

            // 2 poseidon_full_round_chain lookups
            lookup_poseidon_full_round_chain_0.as_ptr(),
            lookup_poseidon_full_round_chain_1.as_ptr(),

            // Base trace cols 120-283 (164 columns)
            lookup_base_trace_cols.as_ptr(),
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    // 6 memory_address_to_id lookups (2 elements each)
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    memory_address_to_id_3: [BaseFieldVec; 2],
    memory_address_to_id_4: [BaseFieldVec; 2],
    memory_address_to_id_5: [BaseFieldVec; 2],
    // 6 memory_id_to_big lookups (29 elements each)
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    memory_id_to_big_3: [BaseFieldVec; 29],
    memory_id_to_big_4: [BaseFieldVec; 29],
    memory_id_to_big_5: [BaseFieldVec; 29],
    // 2 range_check_3_3_3_3_3 lookups (5 elements each)
    range_check_3_3_3_3_3_0: [BaseFieldVec; 5],
    range_check_3_3_3_3_3_1: [BaseFieldVec; 5],
    // 6 range_check_4_4_4_4 lookups (4 elements each)
    range_check_4_4_4_4_0: [BaseFieldVec; 4],
    range_check_4_4_4_4_1: [BaseFieldVec; 4],
    range_check_4_4_4_4_2: [BaseFieldVec; 4],
    range_check_4_4_4_4_3: [BaseFieldVec; 4],
    range_check_4_4_4_4_4: [BaseFieldVec; 4],
    range_check_4_4_4_4_5: [BaseFieldVec; 4],
    // 3 range_check_4_4 lookups (2 elements each)
    range_check_4_4_0: [BaseFieldVec; 2],
    range_check_4_4_1: [BaseFieldVec; 2],
    range_check_4_4_2: [BaseFieldVec; 2],
    // 2 poseidon_full_round_chain lookups (32 elements each)
    poseidon_full_round_chain_0: [BaseFieldVec; 32],
    poseidon_full_round_chain_1: [BaseFieldVec; 32],
    // Base trace columns 120-283 for interaction kernels (164 columns)
    base_trace_cols: [BaseFieldVec; 164],
}

pub struct CudaInteractionClaimGenerator {
    #[allow(dead_code)]
    n_rows: usize,
    log_size: u32,
    lookup_data: Box<CudaLookupData>,
    /// Base trace column pointers - used for reconstructing lookup data in interaction kernel
    base_trace_ptrs: Vec<*const u32>,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        poseidon_full_round_chain: &relations::PoseidonFullRoundChain,
        range_check_felt_252_width_27: &relations::RangeCheckFelt252Width27,
        cube_252: &relations::Cube252,
        range_check_3_3_3_3_3: &relations::RangeCheck_3_3_3_3_3,
        range_check_4_4_4_4: &relations::RangeCheck_4_4_4_4,
        range_check_4_4: &relations::RangeCheck_4_4,
        poseidon_3_partial_rounds_chain: &relations::Poseidon3PartialRoundsChain,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        // Allocate GPU buffer for claimed_sum (4 M31 = 1 QM31)
        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Allocate GPU buffers for interaction trace columns (18 logup cols × 4 M31 = 72 columns)
        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect lookup data pointers - memory_address_to_id
        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_address_to_id_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_3);
        let lookup_memory_address_to_id_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_4);
        let lookup_memory_address_to_id_5_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_5);

        // Collect lookup data pointers - memory_id_to_big
        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_memory_id_to_big_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_3);
        let lookup_memory_id_to_big_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_4);
        let lookup_memory_id_to_big_5_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_5);

        // Collect lookup data pointers - range_check_3_3_3_3_3
        let lookup_range_check_3_3_3_3_3_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_3_3_3_3_0);
        let lookup_range_check_3_3_3_3_3_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_3_3_3_3_3_1);

        // Collect lookup data pointers - range_check_4_4_4_4
        let lookup_range_check_4_4_4_4_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_0);
        let lookup_range_check_4_4_4_4_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_1);
        let lookup_range_check_4_4_4_4_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_2);
        let lookup_range_check_4_4_4_4_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_3);
        let lookup_range_check_4_4_4_4_4_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_4);
        let lookup_range_check_4_4_4_4_5_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_4_4_5);

        // Collect lookup data pointers - range_check_4_4
        let lookup_range_check_4_4_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_0);
        let lookup_range_check_4_4_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_1);
        let lookup_range_check_4_4_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_4_4_2);

        // Collect lookup data pointers - poseidon_full_round_chain
        let lookup_poseidon_full_round_chain_0_vec = collect_lookup_ptrs!(self.lookup_data, poseidon_full_round_chain_0);
        let lookup_poseidon_full_round_chain_1_vec = collect_lookup_ptrs!(self.lookup_data, poseidon_full_round_chain_1);

        // Collect lookup data pointers - base_trace_cols (cols 120-283)
        let lookup_base_trace_cols_vec = collect_lookup_ptrs!(self.lookup_data, base_trace_cols);

        let interaction_trace_vec: Vec<*const u32> = interaction_trace
            .iter()
            .map(|col| col.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *const u32;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *const u32;
            let poseidon_full_round_chain_ptr = poseidon_full_round_chain as *const _ as *const u32;
            let range_check_felt_252_width_27_ptr = range_check_felt_252_width_27 as *const _ as *const u32;
            let cube_252_ptr = cube_252 as *const _ as *const u32;
            let range_check_3_3_3_3_3_ptr = range_check_3_3_3_3_3 as *const _ as *const u32;
            let range_check_4_4_4_4_ptr = range_check_4_4_4_4 as *const _ as *const u32;
            let range_check_4_4_ptr = range_check_4_4 as *const _ as *const u32;
            let poseidon_3_partial_rounds_chain_ptr = poseidon_3_partial_rounds_chain as *const _ as *const u32;

            bindings_airs::gen_poseidon_builtin_interaction_trace(
                interaction_trace_vec.as_ptr(),
                trace_log_size,

                // Relation structs
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                poseidon_full_round_chain_ptr,
                range_check_felt_252_width_27_ptr,
                cube_252_ptr,
                range_check_3_3_3_3_3_ptr,
                range_check_4_4_4_4_ptr,
                range_check_4_4_ptr,
                poseidon_3_partial_rounds_chain_ptr,

                // Base trace columns for reconstructing all lookup data
                self.base_trace_ptrs.as_ptr(),

                // memory_address_to_id lookups (6)
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_address_to_id_3_vec.as_ptr(),
                lookup_memory_address_to_id_4_vec.as_ptr(),
                lookup_memory_address_to_id_5_vec.as_ptr(),

                // memory_id_to_big lookups (6)
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_memory_id_to_big_3_vec.as_ptr(),
                lookup_memory_id_to_big_4_vec.as_ptr(),
                lookup_memory_id_to_big_5_vec.as_ptr(),

                // range_check_3_3_3_3_3 lookups (2)
                lookup_range_check_3_3_3_3_3_0_vec.as_ptr(),
                lookup_range_check_3_3_3_3_3_1_vec.as_ptr(),

                // range_check_4_4_4_4 lookups (6)
                lookup_range_check_4_4_4_4_0_vec.as_ptr(),
                lookup_range_check_4_4_4_4_1_vec.as_ptr(),
                lookup_range_check_4_4_4_4_2_vec.as_ptr(),
                lookup_range_check_4_4_4_4_3_vec.as_ptr(),
                lookup_range_check_4_4_4_4_4_vec.as_ptr(),
                lookup_range_check_4_4_4_4_5_vec.as_ptr(),

                // range_check_4_4 lookups (3)
                lookup_range_check_4_4_0_vec.as_ptr(),
                lookup_range_check_4_4_1_vec.as_ptr(),
                lookup_range_check_4_4_2_vec.as_ptr(),

                // poseidon_full_round_chain lookups (2)
                lookup_poseidon_full_round_chain_0_vec.as_ptr(),
                lookup_poseidon_full_round_chain_1_vec.as_ptr(),

                // base_trace_cols (cols 120-283, 164 columns)
                lookup_base_trace_cols_vec.as_ptr(),

                cuda_claimed_sum.device_ptr as *const u32,
            );
        }

        // Get claimed_sum from GPU
        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([
            claimed_sum_vec[0],
            claimed_sum_vec[1],
            claimed_sum_vec[2],
            claimed_sum_vec[3],
        ]);

        // Extend tree builder with the interaction trace
        let domain = CanonicCoset::new(trace_log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}

use cairo_air::relations;

#[cfg(test)]
pub mod tests {
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use test_log::test;
    use dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_adapter::utils::{run_program_and_adapter, ProgramType};
    use stwo::core::fields::m31::M31;
    use crate::witness::components::{
        cube_252, memory_address_to_id, memory_id_to_big,
        poseidon_3_partial_rounds_chain, poseidon_full_round_chain,
        range_check_3_3_3_3_3, range_check_4_4, range_check_4_4_4_4,
        range_check_felt_252_width_27, poseidon_builtin,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;

    #[test]
    fn test_poseidon_builtin_cpu_ref() {
        use cairo_air::relations;
        use cairo_air::components::poseidon_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use crate::debug_tools::assert_constraints::assert_component;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for poseidon builtin segment
        use stwo_cairo_adapter::builtins::POSEIDON_MEMORY_CELLS;

        let poseidon_segment = input.builtin_segments.poseidon.as_ref()
            .expect("Expected poseidon builtin segment");

        let segment_length = poseidon_segment.stop_ptr - poseidon_segment.begin_addr;
        let n_instances = segment_length / POSEIDON_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = poseidon_segment.begin_addr as u32;


        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let mut cube_252_state = cube_252::ClaimGenerator::default();
        let mut poseidon_3_partial_rounds_chain_state = poseidon_3_partial_rounds_chain::ClaimGenerator::default();
        let mut poseidon_full_round_chain_state = poseidon_full_round_chain::ClaimGenerator::default();
        let range_check_3_3_3_3_3_state = range_check_3_3_3_3_3::ClaimGenerator::new();
        let range_check_4_4_state = range_check_4_4::ClaimGenerator::new();
        let range_check_4_4_4_4_state = range_check_4_4_4_4::ClaimGenerator::new();
        let mut range_check_felt_252_width_27_state = range_check_felt_252_width_27::ClaimGenerator::default();

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_state.get_id(addr);
            memory_address_to_id_state.add_input(&addr);
            memory_id_to_big_state.add_input(&id);
        }

        // Create poseidon_builtin claim generator
        let poseidon_claim_gen = poseidon_builtin::ClaimGenerator::new(log_size, segment_start);

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let poseidon_full_round_chain_relation = relations::PoseidonFullRoundChain::dummy();
        let range_check_felt_252_width_27_relation = relations::RangeCheckFelt252Width27::dummy();
        let cube_252_relation = relations::Cube252::dummy();
        let range_check_3_3_3_3_3_relation = relations::RangeCheck_3_3_3_3_3::dummy();
        let range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();
        let range_check_4_4_relation = relations::RangeCheck_4_4::dummy();
        let poseidon_3_partial_rounds_chain_relation = relations::Poseidon3PartialRoundsChain::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace - use the actual log_size from the claim
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (poseidon_claim, poseidon_interaction_gen) = poseidon_claim_gen.write_trace(
            &mut mock_tree_builder,
            &mut cube_252_state,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &mut poseidon_3_partial_rounds_chain_state,
            &mut poseidon_full_round_chain_state,
            &range_check_3_3_3_3_3_state,
            &range_check_4_4_state,
            &range_check_4_4_4_4_state,
            &mut range_check_felt_252_width_27_state,
        );

        mock_tree_builder.finalize_interaction();


        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let poseidon_interaction_claim = poseidon_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &poseidon_full_round_chain_relation,
            &range_check_felt_252_width_27_relation,
            &cube_252_relation,
            &range_check_3_3_3_3_3_relation,
            &range_check_4_4_4_4_relation,
            &range_check_4_4_relation,
            &poseidon_3_partial_rounds_chain_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();


        // Create component and verify with assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let poseidon_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("poseidon_builtin"),
                claim: poseidon_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                poseidon_full_round_chain_lookup_elements: relations::PoseidonFullRoundChain::dummy(),
                range_check_felt_252_width_27_lookup_elements: relations::RangeCheckFelt252Width27::dummy(),
                cube_252_lookup_elements: relations::Cube252::dummy(),
                range_check_3_3_3_3_3_lookup_elements: relations::RangeCheck_3_3_3_3_3::dummy(),
                range_check_4_4_4_4_lookup_elements: relations::RangeCheck_4_4_4_4::dummy(),
                range_check_4_4_lookup_elements: relations::RangeCheck_4_4::dummy(),
                poseidon_3_partial_rounds_chain_lookup_elements: relations::Poseidon3PartialRoundsChain::dummy(),
            },
            poseidon_interaction_claim.claimed_sum,
        );

        assert_component(&poseidon_component, &trace)
    }

    #[test]
    fn test_poseidon_builtin_trace_gen_by_cpu_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::poseidon_builtin::{Component, Eval};
        use stwo::core::fields::m31::BaseField;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;
        use crate::debug_tools::assert_constraints::assert_component;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for poseidon builtin segment
        use stwo_cairo_adapter::builtins::POSEIDON_MEMORY_CELLS;

        let poseidon_segment = input.builtin_segments.poseidon.as_ref()
            .expect("Expected poseidon builtin segment");

        let segment_length = poseidon_segment.stop_ptr - poseidon_segment.begin_addr;
        let n_instances = segment_length / POSEIDON_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = poseidon_segment.begin_addr as u32;


        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let mut cube_252_state = cube_252::ClaimGenerator::default();
        let mut poseidon_3_partial_rounds_chain_state = poseidon_3_partial_rounds_chain::ClaimGenerator::default();
        let mut poseidon_full_round_chain_state = poseidon_full_round_chain::ClaimGenerator::default();
        let range_check_3_3_3_3_3_state = range_check_3_3_3_3_3::ClaimGenerator::new();
        let range_check_4_4_state = range_check_4_4::ClaimGenerator::new();
        let range_check_4_4_4_4_state = range_check_4_4_4_4::ClaimGenerator::new();
        let mut range_check_felt_252_width_27_state = range_check_felt_252_width_27::ClaimGenerator::default();

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_state.get_id(addr);
            memory_address_to_id_state.add_input(&addr);
            memory_id_to_big_state.add_input(&id);
        }

        // Create poseidon_builtin claim generator
        let poseidon_claim_gen = poseidon_builtin::ClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let poseidon_full_round_chain_relation = relations::PoseidonFullRoundChain::dummy();
        let range_check_felt_252_width_27_relation = relations::RangeCheckFelt252Width27::dummy();
        let cube_252_relation = relations::Cube252::dummy();
        let range_check_3_3_3_3_3_relation = relations::RangeCheck_3_3_3_3_3::dummy();
        let range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();
        let range_check_4_4_relation = relations::RangeCheck_4_4::dummy();
        let poseidon_3_partial_rounds_chain_relation = relations::Poseidon3PartialRoundsChain::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace - use the actual log_size from the claim
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (poseidon_claim, poseidon_interaction_gen) = poseidon_claim_gen.write_trace(
            &mut mock_tree_builder,
            &mut cube_252_state,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &mut poseidon_3_partial_rounds_chain_state,
            &mut poseidon_full_round_chain_state,
            &range_check_3_3_3_3_3_state,
            &range_check_4_4_state,
            &range_check_4_4_4_4_state,
            &mut range_check_felt_252_width_27_state,
        );

        mock_tree_builder.finalize_interaction();


        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let poseidon_interaction_claim = poseidon_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &poseidon_full_round_chain_relation,
            &range_check_felt_252_width_27_relation,
            &cube_252_relation,
            &range_check_3_3_3_3_3_relation,
            &range_check_4_4_4_4_relation,
            &range_check_4_4_relation,
            &poseidon_3_partial_rounds_chain_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // Convert trace to CUDA format
        let trace0_vec: Vec<_> = if !trace[0].is_empty() {
            trace[0].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace1_vec: Vec<_> = if trace.len() > 1 && !trace[1].is_empty() {
            trace[1].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace2_vec: Vec<_> = if trace.len() > 2 && !trace[2].is_empty() {
            trace[2].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace0_evaluations_vec: Vec<_> = trace0_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();
        let trace1_evaluations_vec: Vec<_> = trace1_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();
        let trace2_evaluations_vec: Vec<_> = trace2_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();

        // Create mock CUDA buffers with correct size
        let domain_size = 1 << poseidon_claim.log_size;
        let n_constraints = 500; // Reasonable upper bound for constraints (Poseidon has many constraints)

        let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        // Create component for CUDA evaluator
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let poseidon_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("poseidon_builtin"),
                claim: poseidon_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                poseidon_full_round_chain_lookup_elements: relations::PoseidonFullRoundChain::dummy(),
                range_check_felt_252_width_27_lookup_elements: relations::RangeCheckFelt252Width27::dummy(),
                cube_252_lookup_elements: relations::Cube252::dummy(),
                range_check_3_3_3_3_3_lookup_elements: relations::RangeCheck_3_3_3_3_3::dummy(),
                range_check_4_4_4_4_lookup_elements: relations::RangeCheck_4_4_4_4::dummy(),
                range_check_4_4_lookup_elements: relations::RangeCheck_4_4::dummy(),
                poseidon_3_partial_rounds_chain_lookup_elements: relations::Poseidon3PartialRoundsChain::dummy(),
            },
            poseidon_interaction_claim.claimed_sum,
        );
        assert_component(&poseidon_component, &trace);

        // Call CUDA evaluator
        let eval_ptr = &poseidon_component.eval as *const _ as *mut std::os::raw::c_void;
        unsafe {
            stwo::stwo_cuda::bindings::evaluate_constraint_quotients_on_domain(
                mock_accum_col_columns_0.device_ptr,
                mock_accum_col_columns_1.device_ptr,
                mock_accum_col_columns_2.device_ptr,
                mock_accum_col_columns_3.device_ptr,
                trace0_evaluations_vec.as_ptr(),
                trace0_evaluations_vec.len() as u32,
                trace1_evaluations_vec.as_ptr(),
                trace1_evaluations_vec.len() as u32,
                trace2_evaluations_vec.as_ptr(),
                trace2_evaluations_vec.len() as u32,
                mock_random_coeff_powers.device_ptr,
                mock_gpu_denom_inv.device_ptr,
                poseidon_claim.log_size as u32,
                poseidon_claim.log_size as u32,
                poseidon_component.info.n_constraints as u32,
                poseidon_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    poseidon_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << poseidon_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }

        println!("poseidon_builtin CUDA evaluator test completed successfully!");
    }

    #[test]
    fn test_poseidon_builtin_trace_gen_by_cuda_and_verify_by_cpu() {
        use cairo_air::relations;
        use cairo_air::components::poseidon_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use crate::witness::components_cuda::{memory_address_to_id_cuda, memory_id_to_big_cuda};
        use crate::debug_tools::assert_constraints::assert_component;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for poseidon builtin segment
        use stwo_cairo_adapter::builtins::POSEIDON_MEMORY_CELLS;

        let poseidon_segment = input.builtin_segments.poseidon.as_ref()
            .expect("Expected poseidon builtin segment");

        let segment_length = poseidon_segment.stop_ptr - poseidon_segment.begin_addr;
        let n_instances = segment_length / POSEIDON_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = poseidon_segment.begin_addr as u32;

        // Initialize CUDA generators
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_state.get_id(addr);
            memory_address_to_id_cuda_state.add_cuda_input(&addr);
            memory_id_to_big_cuda_state.add_cuda_input(&id);
        }

        // Create CUDA poseidon_builtin claim generator
        let poseidon_cuda_claim_gen = super::CudaClaimGenerator::new(log_size, segment_start);

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace - generate using CUDA
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (_poseidon_claim, poseidon_interaction_gen) = poseidon_cuda_claim_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        // Generate CUDA interaction trace
        let cuda_memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let cuda_memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let cuda_poseidon_full_round_chain_relation = relations::PoseidonFullRoundChain::dummy();
        let cuda_range_check_felt_252_width_27_relation = relations::RangeCheckFelt252Width27::dummy();
        let cuda_cube_252_relation = relations::Cube252::dummy();
        let cuda_range_check_3_3_3_3_3_relation = relations::RangeCheck_3_3_3_3_3::dummy();
        let cuda_range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();
        let cuda_range_check_4_4_relation = relations::RangeCheck_4_4::dummy();
        let cuda_poseidon_3_partial_rounds_chain_relation = relations::Poseidon3PartialRoundsChain::dummy();

        let mut cuda_tree_builder = mock_commitment_scheme.tree_builder();
        let poseidon_interaction_claim = poseidon_interaction_gen.write_interaction_trace(
            &mut cuda_tree_builder,
            &cuda_memory_address_to_id_relation,
            &cuda_memory_id_to_big_relation,
            &cuda_poseidon_full_round_chain_relation,
            &cuda_range_check_felt_252_width_27_relation,
            &cuda_cube_252_relation,
            &cuda_range_check_3_3_3_3_3_relation,
            &cuda_range_check_4_4_4_4_relation,
            &cuda_range_check_4_4_relation,
            &cuda_poseidon_3_partial_rounds_chain_relation,
        );
        cuda_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        let base_trace = &trace[1];  // Tree 1 is base trace
        let n_rows = 1usize << log_size;

        use stwo::prover::backend::Column;

        // Generate CPU trace for comparison
        let cpu_memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let cpu_memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let cpu_poseidon_full_round_chain_relation = relations::PoseidonFullRoundChain::dummy();
        let cpu_range_check_felt_252_width_27_relation = relations::RangeCheckFelt252Width27::dummy();
        let cpu_cube_252_relation = relations::Cube252::dummy();
        let cpu_range_check_3_3_3_3_3_relation = relations::RangeCheck_3_3_3_3_3::dummy();
        let cpu_range_check_4_4_4_4_relation = relations::RangeCheck_4_4_4_4::dummy();
        let cpu_range_check_4_4_relation = relations::RangeCheck_4_4::dummy();
        let cpu_poseidon_3_partial_rounds_chain_relation = relations::Poseidon3PartialRoundsChain::dummy();

        let mut cpu_full_mock_scheme = MockCommitmentScheme::default();
        let cpu_full_preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut cpu_full_tree_builder = cpu_full_mock_scheme.tree_builder();
        cpu_full_tree_builder.extend_evals(cpu_full_preprocessed_trace.gen_trace());
        cpu_full_tree_builder.finalize_interaction();

        let cpu_full_memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let cpu_full_memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let mut cpu_full_cube_252_state = cube_252::ClaimGenerator::default();
        let mut cpu_full_poseidon_3_partial_rounds_chain_state = poseidon_3_partial_rounds_chain::ClaimGenerator::default();
        let mut cpu_full_poseidon_full_round_chain_state = poseidon_full_round_chain::ClaimGenerator::default();
        let cpu_full_range_check_3_3_3_3_3_state = range_check_3_3_3_3_3::ClaimGenerator::new();
        let cpu_full_range_check_4_4_state = range_check_4_4::ClaimGenerator::new();
        let cpu_full_range_check_4_4_4_4_state = range_check_4_4_4_4::ClaimGenerator::new();
        let mut cpu_full_range_check_felt_252_width_27_state = range_check_felt_252_width_27::ClaimGenerator::default();

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = cpu_full_memory_address_to_id_state.get_id(addr);
            cpu_full_memory_address_to_id_state.add_input(&addr);
            cpu_full_memory_id_to_big_state.add_input(&id);
        }

        let cpu_full_poseidon_claim_gen = poseidon_builtin::ClaimGenerator::new(log_size, segment_start);
        let mut cpu_full_tree_builder = cpu_full_mock_scheme.tree_builder();
        let (_cpu_full_claim, cpu_full_interaction_gen) = cpu_full_poseidon_claim_gen.write_trace(
            &mut cpu_full_tree_builder,
            &mut cpu_full_cube_252_state,
            &cpu_full_memory_address_to_id_state,
            &cpu_full_memory_id_to_big_state,
            &mut cpu_full_poseidon_3_partial_rounds_chain_state,
            &mut cpu_full_poseidon_full_round_chain_state,
            &cpu_full_range_check_3_3_3_3_3_state,
            &cpu_full_range_check_4_4_state,
            &cpu_full_range_check_4_4_4_4_state,
            &mut cpu_full_range_check_felt_252_width_27_state,
        );
        cpu_full_tree_builder.finalize_interaction();

        // Generate CPU interaction trace
        let mut cpu_full_int_tree_builder = cpu_full_mock_scheme.tree_builder();
        let cpu_full_interaction_claim = cpu_full_interaction_gen.write_interaction_trace(
            &mut cpu_full_int_tree_builder,
            &cpu_memory_address_to_id_relation,
            &cpu_memory_id_to_big_relation,
            &cpu_poseidon_full_round_chain_relation,
            &cpu_range_check_felt_252_width_27_relation,
            &cpu_cube_252_relation,
            &cpu_range_check_3_3_3_3_3_relation,
            &cpu_range_check_4_4_4_4_relation,
            &cpu_range_check_4_4_relation,
            &cpu_poseidon_3_partial_rounds_chain_relation,
        );
        cpu_full_int_tree_builder.finalize_interaction();

        // Get CPU traces after all trees are finalized
        let cpu_full_trace = cpu_full_mock_scheme.trace_domain_evaluations();
        let cpu_base_trace = &cpu_full_trace[1];
        let cpu_interaction_trace = &cpu_full_trace[2];

        // Compare CUDA and CPU traces
        let cuda_interaction_trace = &trace[2];

        // Count differences
        let mut total_base_diffs = 0;
        let mut total_interaction_diffs = 0;
        for row in 0..n_rows {
            for col in 0..341 {
                let cuda_val = base_trace[col].at(row).0;
                let cpu_val = cpu_base_trace[col][row].0;
                if cuda_val != cpu_val {
                    total_base_diffs += 1;
                }
            }
            for col in 0..68.min(cuda_interaction_trace.len()) {
                let cuda_val = cuda_interaction_trace[col].at(row).0;
                let cpu_val = cpu_interaction_trace[col][row].0;
                if cuda_val != cpu_val {
                    total_interaction_diffs += 1;
                }
            }
        }

        // Assert traces match
        assert_eq!(total_base_diffs, 0, "Base trace should match exactly");
        assert_eq!(total_interaction_diffs, 0, "Interaction trace should match exactly");
        assert_eq!(
            poseidon_interaction_claim.claimed_sum,
            cpu_full_interaction_claim.claimed_sum,
            "Claimed sum should match"
        );
    }
}
