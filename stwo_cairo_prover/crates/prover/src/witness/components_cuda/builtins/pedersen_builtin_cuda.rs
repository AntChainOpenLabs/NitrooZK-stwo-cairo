// CUDA witness generation for pedersen_builtin component
// This handles Pedersen hash operations using elliptic curve scalar multiplication
//
// Full GPU approach:
// - GPU uploads PEDERSEN_TABLE (~1.8GB) once at initialization
// - GPU computes partial_ec_mul (EC point additions with 252-bit field arithmetic)
// - GPU handles memory lookups and trace assembly
// - GPU generates interaction trace

#![allow(unused_parens)]
use cairo_air::components::pedersen_builtin::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big, partial_ec_mul};
use stwo::prover::backend::cuda::CudaBackend;

use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_cairo_common::preprocessed_columns::pedersen::{PEDERSEN_TABLE, PEDERSEN_TABLE_N_ROWS};

pub const N_TRACE_COLUMNS: usize = 351;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 20;

use itertools::Itertools;
use rayon::prelude::*;
use stwo::prover::backend::{Col, Column};
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use std::sync::Once;

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

macro_rules! collect_sub_input_ptrs {
    ($sub_inputs:expr, $field:ident) => {
        $sub_inputs.$field
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec()
    };
}

// ============================================================================
// PEDERSEN_TABLE GPU Upload (done once, reused for all pedersen_builtin calls)
// ============================================================================

// Static variables for one-time GPU table upload
static PEDERSEN_TABLE_INIT: Once = Once::new();
static mut PEDERSEN_TABLE_GPU_COLUMNS: Option<Vec<BaseFieldVec>> = None;

/// Upload PEDERSEN_TABLE to GPU memory (called once, reused for all traces)
/// Table structure: 56 columns (28 for x limbs, 28 for y limbs) × ~8M rows
fn ensure_pedersen_table_on_gpu() {
    PEDERSEN_TABLE_INIT.call_once(|| {
        let n_rows = PEDERSEN_TABLE_N_ROWS;
        let n_cols = 56; // 28 for x + 28 for y

        // Create GPU buffers for each column
        let mut gpu_columns = Vec::with_capacity(n_cols);

        // Upload each column to GPU
        for col_idx in 0..n_cols {
            // Extract column data from PEDERSEN_TABLE
            // PEDERSEN_TABLE stores (x, y) points, each as 28 M31 limbs
            // col_idx 0-27: x limbs, col_idx 28-55: y limbs
            let col_data: Vec<M31> = (0..n_rows)
                .map(|row_idx| {
                    let coords = PEDERSEN_TABLE.get_row_coordinates(row_idx);
                    if col_idx < 28 {
                        coords[0].get_limbs()[col_idx]
                    } else {
                        coords[1].get_limbs()[col_idx - 28]
                    }
                })
                .collect();

            gpu_columns.push(BaseFieldVec::from_vec(col_data));
        }

        // Get device pointers
        let col_ptrs: Vec<*const u32> = gpu_columns.iter()
            .map(|col| col.device_ptr)
            .collect();

        // Initialize the table on GPU
        unsafe {
            bindings_airs::pedersen_table_init(col_ptrs.as_ptr(), n_rows as u32);

            // Store the GPU buffers to keep them alive
            PEDERSEN_TABLE_GPU_COLUMNS = Some(gpu_columns);
        }
    });
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
        partial_ec_mul_state: &mut partial_ec_mul::ClaimGenerator,
        // Also pass SIMD generators for multiplicity tracking (needed for final memory traces)
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        let n_rows = 1usize << log_size;

        // Ensure PEDERSEN_TABLE is uploaded to GPU (one-time initialization)
        ensure_pedersen_table_on_gpu();

        // Full GPU trace generation - no CPU precomputation needed
        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda_full_gpu(
            n_rows,
            log_size,
            self.segment_start,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
            partial_ec_mul_state,
        );

        // Add to CUDA generators
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        // Copy GPU data to CPU and add to SIMD generators
        // Skip zero values (padding sentinel) and out-of-bounds addresses
        let memory_size = memory_address_to_id_simd_state.memory_size();
        for input_arr in &sub_component_inputs.memory_address_to_id {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for addr in cpu_data.iter().take(n_rows) {
                // Skip zero (padding sentinel) and addresses exceeding memory bounds
                if addr.0 != 0 && addr.0 <= memory_size {
                    memory_address_to_id_simd_state.add_input(addr);
                }
            }
        }
        for input_arr in &sub_component_inputs.memory_id_to_big {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for id in cpu_data.iter().take(n_rows) {
                // Skip zero (padding sentinel) - memory_id_to_big handles bounds internally
                if id.0 != 0 {
                    memory_id_to_big_simd_state.add_input(id);
                }
            }
        }

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size, pedersen_builtin_segment_start: self.segment_start },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

pub struct CudaSubComponentInputs {
    // 3 memory_address_to_id lookups
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 3],
    // 3 memory_id_to_big lookups
    pub memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 3],
}

/// Full GPU trace generation - computes partial_ec_mul on GPU using PEDERSEN_TABLE
#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda_full_gpu(
    n_rows: usize,
    log_size: u32,
    segment_start: u32,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    partial_ec_mul_state: &mut partial_ec_mul::ClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    Box<CudaLookupData>,
    CudaSubComponentInputs,
) {
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            Box::new(CudaLookupData {
                // 3 memory_address_to_id lookups (2 elements each)
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                // 3 memory_id_to_big lookups (29 elements each)
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                // 8 partial_ec_mul lookups (73 elements each) - computed on GPU
                partial_ec_mul_0: init_lookup_array!(log_size),
                partial_ec_mul_1: init_lookup_array!(log_size),
                partial_ec_mul_2: init_lookup_array!(log_size),
                partial_ec_mul_3: init_lookup_array!(log_size),
                partial_ec_mul_4: init_lookup_array!(log_size),
                partial_ec_mul_5: init_lookup_array!(log_size),
                partial_ec_mul_6: init_lookup_array!(log_size),
                partial_ec_mul_7: init_lookup_array!(log_size),
                // 2 range_check_5_4 lookups (2 elements each)
                range_check_5_4_0: init_lookup_array!(log_size),
                range_check_5_4_1: init_lookup_array!(log_size),
                // 4 range_check_8 lookups (1 element each)
                range_check_8_0: init_lookup_array!(log_size),
                range_check_8_1: init_lookup_array!(log_size),
                range_check_8_2: init_lookup_array!(log_size),
                range_check_8_3: init_lookup_array!(log_size),
            }),
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers - 3 memory_address_to_id
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);

    // Collect lookup data pointers - 3 memory_id_to_big
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);

    // Collect lookup data pointers - 8 partial_ec_mul (computed on GPU)
    let lookup_partial_ec_mul_0 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_0);
    let lookup_partial_ec_mul_1 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_1);
    let lookup_partial_ec_mul_2 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_2);
    let lookup_partial_ec_mul_3 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_3);
    let lookup_partial_ec_mul_4 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_4);
    let lookup_partial_ec_mul_5 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_5);
    let lookup_partial_ec_mul_6 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_6);
    let lookup_partial_ec_mul_7 = collect_lookup_ptrs!(lookup_data, partial_ec_mul_7);

    // Collect lookup data pointers - 2 range_check_5_4
    let lookup_range_check_5_4_0 = collect_lookup_ptrs!(lookup_data, range_check_5_4_0);
    let lookup_range_check_5_4_1 = collect_lookup_ptrs!(lookup_data, range_check_5_4_1);

    // Collect lookup data pointers - 4 range_check_8
    let lookup_range_check_8_0 = collect_lookup_ptrs!(lookup_data, range_check_8_0);
    let lookup_range_check_8_1 = collect_lookup_ptrs!(lookup_data, range_check_8_1);
    let lookup_range_check_8_2 = collect_lookup_ptrs!(lookup_data, range_check_8_2);
    let lookup_range_check_8_3 = collect_lookup_ptrs!(lookup_data, range_check_8_3);

    // Collect sub-component inputs
    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);

    // Transpose memory_id_to_big big_values for GPU access
    let memory_id_to_big_transposed_big_values_vec: Vec<_> = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect();

    // Call full GPU kernel - computes partial_ec_mul on GPU using PEDERSEN_TABLE
    unsafe {
        bindings_airs::gen_pedersen_builtin_trace_full_gpu(
            traces_vec.as_ptr(),
            N_TRACE_COLUMNS as u32,

            // 3 memory_address_to_id lookups
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),

            // 3 memory_id_to_big lookups
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),

            // 8 partial_ec_mul lookups (computed on GPU)
            lookup_partial_ec_mul_0.as_ptr(),
            lookup_partial_ec_mul_1.as_ptr(),
            lookup_partial_ec_mul_2.as_ptr(),
            lookup_partial_ec_mul_3.as_ptr(),
            lookup_partial_ec_mul_4.as_ptr(),
            lookup_partial_ec_mul_5.as_ptr(),
            lookup_partial_ec_mul_6.as_ptr(),
            lookup_partial_ec_mul_7.as_ptr(),

            // 2 range_check_5_4 lookups
            lookup_range_check_5_4_0.as_ptr(),
            lookup_range_check_5_4_1.as_ptr(),

            // 4 range_check_8 lookups
            lookup_range_check_8_0.as_ptr(),
            lookup_range_check_8_1.as_ptr(),
            lookup_range_check_8_2.as_ptr(),
            lookup_range_check_8_3.as_ptr(),

            // Sub-component inputs
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),

            // Builtin segment info
            segment_start,

            // Memory lookup tables
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,

            n_rows as u32,
            log_size,
        );
    }
    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    // 3 memory_address_to_id lookups (2 elements each)
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    // 3 memory_id_to_big lookups (29 elements each)
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    // 8 partial_ec_mul lookups (73 elements each)
    partial_ec_mul_0: [BaseFieldVec; 73],
    partial_ec_mul_1: [BaseFieldVec; 73],
    partial_ec_mul_2: [BaseFieldVec; 73],
    partial_ec_mul_3: [BaseFieldVec; 73],
    partial_ec_mul_4: [BaseFieldVec; 73],
    partial_ec_mul_5: [BaseFieldVec; 73],
    partial_ec_mul_6: [BaseFieldVec; 73],
    partial_ec_mul_7: [BaseFieldVec; 73],
    // 2 range_check_5_4 lookups (2 elements each)
    range_check_5_4_0: [BaseFieldVec; 2],
    range_check_5_4_1: [BaseFieldVec; 2],
    // 4 range_check_8 lookups (1 element each)
    range_check_8_0: [BaseFieldVec; 1],
    range_check_8_1: [BaseFieldVec; 1],
    range_check_8_2: [BaseFieldVec; 1],
    range_check_8_3: [BaseFieldVec; 1],
}

pub struct CudaInteractionClaimGenerator {
    n_rows: usize,
    log_size: u32,
    lookup_data: Box<CudaLookupData>,
    // partial_ec_mul lookups are now computed on GPU and stored in lookup_data
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        partial_ec_mul: &relations::PartialEcMul,
        range_check_5_4: &relations::RangeCheck_5_4,
        range_check_8: &relations::RangeCheck_8,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        // Allocate GPU buffer for claimed_sum (4 M31 = 1 QM31)
        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Allocate GPU buffers for interaction trace columns (10 logup cols × 4 M31 = 40 columns)
        let interaction_trace: Vec<Col<CudaBackend, BaseField>> = (0..4 * N_INTERACTION_TRACE_COLUMNS / 2)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect lookup data pointers
        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_partial_ec_mul_0_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_0);
        let lookup_partial_ec_mul_1_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_1);
        let lookup_partial_ec_mul_2_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_2);
        let lookup_partial_ec_mul_3_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_3);
        let lookup_partial_ec_mul_4_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_4);
        let lookup_partial_ec_mul_5_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_5);
        let lookup_partial_ec_mul_6_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_6);
        let lookup_partial_ec_mul_7_vec = collect_lookup_ptrs!(self.lookup_data, partial_ec_mul_7);
        let lookup_range_check_5_4_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_5_4_0);
        let lookup_range_check_5_4_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_5_4_1);
        let lookup_range_check_8_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_8_0);
        let lookup_range_check_8_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_8_1);
        let lookup_range_check_8_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_8_2);
        let lookup_range_check_8_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_8_3);

        let interaction_trace_vec: Vec<*const u32> = interaction_trace
            .iter()
            .map(|col| col.device_ptr)
            .collect_vec();

        unsafe {
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *const u32;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *const u32;
            let partial_ec_mul_ptr = partial_ec_mul as *const _ as *const u32;
            let range_check_5_4_ptr = range_check_5_4 as *const _ as *const u32;
            let range_check_8_ptr = range_check_8 as *const _ as *const u32;

            bindings_airs::gen_pedersen_builtin_interaction_trace(
                interaction_trace_vec.as_ptr(),
                (N_INTERACTION_TRACE_COLUMNS / 2 * 4) as u32,

                // memory_address_to_id lookups (3)
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),

                // memory_id_to_big lookups (3)
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),

                // partial_ec_mul lookups (8)
                lookup_partial_ec_mul_0_vec.as_ptr(),
                lookup_partial_ec_mul_1_vec.as_ptr(),
                lookup_partial_ec_mul_2_vec.as_ptr(),
                lookup_partial_ec_mul_3_vec.as_ptr(),
                lookup_partial_ec_mul_4_vec.as_ptr(),
                lookup_partial_ec_mul_5_vec.as_ptr(),
                lookup_partial_ec_mul_6_vec.as_ptr(),
                lookup_partial_ec_mul_7_vec.as_ptr(),

                // range_check_5_4 lookups (2)
                lookup_range_check_5_4_0_vec.as_ptr(),
                lookup_range_check_5_4_1_vec.as_ptr(),

                // range_check_8 lookups (4)
                lookup_range_check_8_0_vec.as_ptr(),
                lookup_range_check_8_1_vec.as_ptr(),
                lookup_range_check_8_2_vec.as_ptr(),
                lookup_range_check_8_3_vec.as_ptr(),

                // Relation structs
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                partial_ec_mul_ptr,
                range_check_5_4_ptr,
                range_check_8_ptr,

                cuda_claimed_sum.device_ptr as *const u32,
                self.n_rows as u32,
                trace_log_size,
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
    use test_log::test;
    use dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_adapter::utils::{run_program_and_adapter, ProgramType};
    use stwo::core::fields::m31::M31;
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use crate::witness::components::{
        memory_address_to_id, memory_id_to_big,
        partial_ec_mul,
        range_check_5_4, range_check_8,
        pedersen_builtin,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;

    #[test]
    fn test_pedersen_builtin_cpu_ref() {
        use cairo_air::relations;
        use cairo_air::components::pedersen_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use crate::debug_tools::assert_constraints::assert_component;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for pedersen builtin segment
        use stwo_cairo_adapter::builtins::PEDERSEN_MEMORY_CELLS;

        let pedersen_segment = input.builtin_segments.pedersen.as_ref()
            .expect("Expected pedersen builtin segment");

        let segment_length = pedersen_segment.stop_ptr - pedersen_segment.begin_addr;
        let n_instances = segment_length / PEDERSEN_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = pedersen_segment.begin_addr as u32;

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let mut partial_ec_mul_state = partial_ec_mul::ClaimGenerator::default();
        let range_check_5_4_state = range_check_5_4::ClaimGenerator::new();
        let range_check_8_state = range_check_8::ClaimGenerator::new();

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

        // Create pedersen_builtin claim generator
        let pedersen_claim_gen = pedersen_builtin::ClaimGenerator::new(log_size, segment_start);

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let partial_ec_mul_relation = relations::PartialEcMul::dummy();
        let range_check_5_4_relation = relations::RangeCheck_5_4::dummy();
        let range_check_8_relation = relations::RangeCheck_8::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace - use canonical because this is CPU reference test
        let preprocessed_trace = PreProcessedTrace::canonical();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (pedersen_claim, pedersen_interaction_gen) = pedersen_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &mut partial_ec_mul_state,
            &range_check_5_4_state,
            &range_check_8_state,
        );

        mock_tree_builder.finalize_interaction();

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let pedersen_interaction_claim = pedersen_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &range_check_5_4_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_8_relation,
            &partial_ec_mul_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // Create component and verify with assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let pedersen_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("pedersen_builtin"),
                claim: pedersen_claim.clone(),
                range_check_5_4_lookup_elements: relations::RangeCheck_5_4::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_8_lookup_elements: relations::RangeCheck_8::dummy(),
                partial_ec_mul_lookup_elements: relations::PartialEcMul::dummy(),
            },
            pedersen_interaction_claim.claimed_sum,
        );

        assert_component(&pedersen_component, &trace)
    }

    #[test]
    fn test_pedersen_builtin_trace_gen_by_cpu_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::pedersen_builtin::{Component, Eval};
        use stwo::core::fields::m31::BaseField;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
        use stwo::prover::backend::Column;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for pedersen builtin segment
        use stwo_cairo_adapter::builtins::PEDERSEN_MEMORY_CELLS;

        let pedersen_segment = input.builtin_segments.pedersen.as_ref()
            .expect("Expected pedersen builtin segment");

        let segment_length = pedersen_segment.stop_ptr - pedersen_segment.begin_addr;
        let n_instances = segment_length / PEDERSEN_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = pedersen_segment.begin_addr as u32;

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let mut partial_ec_mul_state = partial_ec_mul::ClaimGenerator::default();
        let range_check_5_4_state = range_check_5_4::ClaimGenerator::new();
        let range_check_8_state = range_check_8::ClaimGenerator::new();

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

        // Create pedersen_builtin claim generator
        let pedersen_claim_gen = pedersen_builtin::ClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let partial_ec_mul_relation = relations::PartialEcMul::dummy();
        let range_check_5_4_relation = relations::RangeCheck_5_4::dummy();
        let range_check_8_relation = relations::RangeCheck_8::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace - use canonical because this is CPU reference test
        let preprocessed_trace = PreProcessedTrace::canonical();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (pedersen_claim, pedersen_interaction_gen) = pedersen_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &mut partial_ec_mul_state,
            &range_check_5_4_state,
            &range_check_8_state,
        );

        mock_tree_builder.finalize_interaction();

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let pedersen_interaction_claim = pedersen_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &range_check_5_4_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_8_relation,
            &partial_ec_mul_relation,
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

        // Create component first to get actual constraint count
        let tree_span_provider_temp = &mut TraceLocationAllocator::default();
        let pedersen_component_temp = Component::new(
            tree_span_provider_temp,
            Eval {
                eval_id: fnv1a_eval_id_gen("pedersen_builtin"),
                claim: pedersen_claim.clone(),
                range_check_5_4_lookup_elements: relations::RangeCheck_5_4::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_8_lookup_elements: relations::RangeCheck_8::dummy(),
                partial_ec_mul_lookup_elements: relations::PartialEcMul::dummy(),
            },
            pedersen_interaction_claim.claimed_sum,
        );

        // Create mock CUDA buffers with correct size
        let domain_size = 1 << pedersen_claim.log_size;
        let n_constraints = pedersen_component_temp.info.n_constraints.max(500);

        let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        // Use the component we already created
        let pedersen_component = pedersen_component_temp;

        // Call CUDA evaluator
        let eval_ptr = &pedersen_component.eval as *const _ as *mut std::os::raw::c_void;
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
                pedersen_claim.log_size as u32,
                pedersen_claim.log_size as u32,
                pedersen_component.info.n_constraints as u32,
                pedersen_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    pedersen_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << pedersen_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }
    }

    #[test]
    fn test_pedersen_builtin_trace_gen_by_cuda_and_verify_by_cpu() {
        use cairo_air::relations;
        use cairo_air::components::pedersen_builtin::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use crate::witness::components_cuda::{memory_address_to_id_cuda, memory_id_to_big_cuda};
        use crate::debug_tools::assert_constraints::assert_component;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for pedersen builtin segment
        use stwo_cairo_adapter::builtins::PEDERSEN_MEMORY_CELLS;

        let pedersen_segment = input.builtin_segments.pedersen.as_ref()
            .expect("Expected pedersen builtin segment");

        let segment_length = pedersen_segment.stop_ptr - pedersen_segment.begin_addr;
        let n_instances = segment_length / PEDERSEN_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = pedersen_segment.begin_addr as u32;

        println!("pedersen_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize CUDA generators
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let mut partial_ec_mul_state = partial_ec_mul::ClaimGenerator::default();

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

        // Create CUDA pedersen_builtin claim generator
        let pedersen_cuda_claim_gen = super::CudaClaimGenerator::new(log_size, segment_start);

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace - generate using CUDA
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (pedersen_claim, pedersen_interaction_gen) = pedersen_cuda_claim_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &mut partial_ec_mul_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        println!("pedersen_builtin_claim log_size: {:?}", pedersen_claim.log_size);

        // Generate CUDA interaction trace using lookup data from CUDA base trace generation
        let cuda_range_check_5_4_relation = relations::RangeCheck_5_4::dummy();
        let cuda_memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let cuda_memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let cuda_range_check_8_relation = relations::RangeCheck_8::dummy();
        let cuda_partial_ec_mul_relation = relations::PartialEcMul::dummy();

        let mut cuda_tree_builder = mock_commitment_scheme.tree_builder();
        let pedersen_interaction_claim = pedersen_interaction_gen.write_interaction_trace(
            &mut cuda_tree_builder,
            &cuda_memory_address_to_id_relation,
            &cuda_memory_id_to_big_relation,
            &cuda_partial_ec_mul_relation,
            &cuda_range_check_5_4_relation,
            &cuda_range_check_8_relation,
        );
        cuda_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("pedersen_builtin_interaction_claim.claimed_sum: {:?}", pedersen_interaction_claim.claimed_sum);

        // Create component and verify with assert_component (CPU verification)
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let pedersen_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("pedersen_builtin"),
                claim: pedersen_claim.clone(),
                range_check_5_4_lookup_elements: relations::RangeCheck_5_4::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_8_lookup_elements: relations::RangeCheck_8::dummy(),
                partial_ec_mul_lookup_elements: relations::PartialEcMul::dummy(),
            },
            pedersen_interaction_claim.claimed_sum,
        );

        assert_component(&pedersen_component, &trace)
    }
}
