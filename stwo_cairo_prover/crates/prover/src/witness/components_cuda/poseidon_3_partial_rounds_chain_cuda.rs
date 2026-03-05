#![allow(unused_parens)]

use cairo_air::components::poseidon_3_partial_rounds_chain::{
    Claim, InteractionClaim, N_TRACE_COLUMNS,
};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use crate::witness::components_cuda::range_check::{rc_4_4, rc_4_4_4_4};
use crate::witness::components_cuda::{cube_252_cuda, range_check_252_width_27_cuda};
use crate::witness::prelude::*;

/// Number of interaction trace columns for poseidon_3_partial_rounds_chain.
/// 9 logup columns * 4 (for SecureField) = 36
pub const N_INTERACTION_TRACE_COLUMNS: usize = 9;

/// Input format: (index, round_number, [state0, state1, state2, state3] in Width27)
pub type PackedInputType = (PackedM31, PackedM31, [PackedFelt252Width27; 4]);

/// CUDA packed input type:
/// - input_limb_0: index (1 column)
/// - input_limb_1: round_number (1 column)
/// - state_0: 10 columns (Width27 format)
/// - state_1: 10 columns (Width27 format)
/// - state_2: 10 columns (Width27 format)
/// - state_3: 10 columns (Width27 format)
pub struct CudaPackedInputType {
    pub input_limb_0: BaseFieldVec,
    pub input_limb_1: BaseFieldVec,
    pub state_0: [BaseFieldVec; 10],
    pub state_1: [BaseFieldVec; 10],
    pub state_2: [BaseFieldVec; 10],
    pub state_3: [BaseFieldVec; 10],
}

macro_rules! collect_ptrs {
    ($arr:expr) => {
        $arr.iter().map(|x| x.device_ptr).collect_vec()
    };
}

pub struct CudaClaimGenerator {
    /// Input limb 0 (index)
    pub input_limb_0: BaseFieldVec,
    /// Input limb 1 (round number)
    pub input_limb_1: BaseFieldVec,
    /// State[0] in Width27 format (10 columns)
    pub state_0: [BaseFieldVec; 10],
    /// State[1] in Width27 format (10 columns)
    pub state_1: [BaseFieldVec; 10],
    /// State[2] in Width27 format (10 columns)
    pub state_2: [BaseFieldVec; 10],
    /// State[3] in Width27 format (10 columns)
    pub state_3: [BaseFieldVec; 10],
    /// Number of rows
    pub size: u32,
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            input_limb_0: BaseFieldVec::new_zeroes(1),
            input_limb_1: BaseFieldVec::new_zeroes(1),
            state_0: std::array::from_fn(|_| BaseFieldVec::new_zeroes(1)),
            state_1: std::array::from_fn(|_| BaseFieldVec::new_zeroes(1)),
            state_2: std::array::from_fn(|_| BaseFieldVec::new_zeroes(1)),
            state_3: std::array::from_fn(|_| BaseFieldVec::new_zeroes(1)),
            size: 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size <= 1
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        // Cube252 CUDA generator (for adding inputs)
        cube_252_state: &mut cube_252_cuda::CudaClaimGenerator,
        // Poseidon round keys (static table - we just need the table data)
        poseidon_round_keys_table: &[[BaseFieldVec; 10]; 3],
        // Range check dependencies
        range_check_4_4_state: &rc_4_4::CudaClaimGenerator,
        range_check_4_4_4_4_state: &rc_4_4_4_4::CudaClaimGenerator,
        // Range check 252 width 27 (output states go here)
        range_check_252_width_27_state: &mut range_check_252_width_27_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let n_rows = self.size as usize;
        let log_size = n_rows.next_power_of_two().ilog2();
        let padded_size = 1usize << log_size;

        // Pad input arrays to next power of 2 (CUDA kernel expects this)
        // IMPORTANT: Must pad by replicating the first packed input (first 16 rows) to match SIMD
        // behavior
        const N_LANES: usize = 16;
        if n_rows < padded_size {
            let pad_with_first_packed = |vec: &mut BaseFieldVec| {
                let mut padded = BaseFieldVec::new_uninitialized(padded_size);
                padded.copy_from(vec);
                let first_tile = BaseFieldVec::from_borrowed_ptr(vec.device_ptr, N_LANES);
                let mut offset = n_rows;
                while offset < padded_size {
                    let chunk = std::cmp::min(N_LANES, padded_size - offset);
                    if chunk == N_LANES {
                        padded.copy_from_offset(&first_tile, offset);
                    } else {
                        let partial = BaseFieldVec::from_borrowed_ptr(vec.device_ptr, chunk);
                        padded.copy_from_offset(&partial, offset);
                    }
                    offset += N_LANES;
                }
                *vec = padded;
            };

            pad_with_first_packed(&mut self.input_limb_0);
            pad_with_first_packed(&mut self.input_limb_1);
            for i in 0..10 {
                pad_with_first_packed(&mut self.state_0[i]);
                pad_with_first_packed(&mut self.state_1[i]);
                pad_with_first_packed(&mut self.state_2[i]);
                pad_with_first_packed(&mut self.state_3[i]);
            }
        }

        // Generate trace
        let trace = write_trace_cuda(
            &self.input_limb_0,
            &self.input_limb_1,
            &self.state_0,
            &self.state_1,
            &self.state_2,
            &self.state_3,
            n_rows,
            log_size,
            poseidon_round_keys_table,
        );

        // Add cube_252 inputs (3 per chain row, one per partial round).
        // Round 0: cubes state_3 (input cols 32-41)
        cube_252_state.add_cuda_inputs(&self.state_3);
        // Round 1: cubes the linear combination output (trace cols 93-102)
        let cube_252_cols_1: [BaseFieldVec; 10] = std::array::from_fn(|i| {
            BaseFieldVec::from_borrowed_ptr(trace.data[93 + i].device_ptr, trace.data[93 + i].size)
        });
        cube_252_state.add_cuda_inputs(&cube_252_cols_1);
        // Round 2: cubes the linear combination output (trace cols 125-134)
        let cube_252_cols_2: [BaseFieldVec; 10] = std::array::from_fn(|i| {
            BaseFieldVec::from_borrowed_ptr(
                trace.data[125 + i].device_ptr,
                trace.data[125 + i].size,
            )
        });
        cube_252_state.add_cuda_inputs(&cube_252_cols_2);

        // Add range_check_252_width_27 inputs from trace columns
        let rc_felt_cols_0: [BaseFieldVec; 10] = std::array::from_fn(|i| {
            BaseFieldVec::from_borrowed_ptr(trace.data[82 + i].device_ptr, trace.data[82 + i].size)
        });
        let rc_felt_cols_1: [BaseFieldVec; 10] = std::array::from_fn(|i| {
            BaseFieldVec::from_borrowed_ptr(
                trace.data[114 + i].device_ptr,
                trace.data[114 + i].size,
            )
        });
        let rc_felt_cols_2: [BaseFieldVec; 10] = std::array::from_fn(|i| {
            BaseFieldVec::from_borrowed_ptr(
                trace.data[146 + i].device_ptr,
                trace.data[146 + i].size,
            )
        });

        range_check_252_width_27_state.add_cuda_inputs(&rc_felt_cols_0);
        range_check_252_width_27_state.add_cuda_inputs(&rc_felt_cols_1);
        range_check_252_width_27_state.add_cuda_inputs(&rc_felt_cols_2);

        // Add to range check multiplicities
        // IMPORTANT: Must use padded_size (not n_rows) to match SIMD behavior where
        // padding rows also contribute to range check multiplicities
        add_to_multiplicities(
            &trace,
            padded_size as u32,
            cube_252_state,
            range_check_4_4_state,
            range_check_4_4_4_4_state,
        );

        // Clone trace data for interaction generator before consuming for evals.
        let trace_data_clone: [BaseFieldVec; N_TRACE_COLUMNS] = trace.data.clone();

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                trace_data: trace_data_clone,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, inputs: &[PackedInputType]) {
        // Convert SIMD packed inputs to CUDA format
        let input_limb_0_elements: Vec<M31> = inputs
            .iter()
            .flat_map(|(limb0, ..)| limb0.to_array())
            .collect();
        let input_limb_1_elements: Vec<M31> = inputs
            .iter()
            .flat_map(|(_, limb1, _)| limb1.to_array())
            .collect();

        self.input_limb_0 = BaseFieldVec::from_vec(input_limb_0_elements);
        self.input_limb_1 = BaseFieldVec::from_vec(input_limb_1_elements);

        // Convert state arrays (4 states instead of 3)
        for i in 0..10 {
            let state_0_elements: Vec<M31> = inputs
                .iter()
                .flat_map(|(_, _, states)| states[0].get_m31(i).to_array())
                .collect();
            self.state_0[i] = BaseFieldVec::from_vec(state_0_elements);

            let state_1_elements: Vec<M31> = inputs
                .iter()
                .flat_map(|(_, _, states)| states[1].get_m31(i).to_array())
                .collect();
            self.state_1[i] = BaseFieldVec::from_vec(state_1_elements);

            let state_2_elements: Vec<M31> = inputs
                .iter()
                .flat_map(|(_, _, states)| states[2].get_m31(i).to_array())
                .collect();
            self.state_2[i] = BaseFieldVec::from_vec(state_2_elements);

            let state_3_elements: Vec<M31> = inputs
                .iter()
                .flat_map(|(_, _, states)| states[3].get_m31(i).to_array())
                .collect();
            self.state_3[i] = BaseFieldVec::from_vec(state_3_elements);
        }

        self.size = self.input_limb_0.size as u32;
    }

    pub fn add_cuda_inputs(&mut self, inputs: &CudaPackedInputType) {
        if inputs.input_limb_0.size == 0 {
            return; // Skip empty inputs
        }
        if self.size <= 1 {
            self.input_limb_0 = inputs.input_limb_0.clone();
            self.input_limb_1 = inputs.input_limb_1.clone();
            for i in 0..10 {
                self.state_0[i] = inputs.state_0[i].clone();
                self.state_1[i] = inputs.state_1[i].clone();
                self.state_2[i] = inputs.state_2[i].clone();
                self.state_3[i] = inputs.state_3[i].clone();
            }
        } else {
            self.input_limb_0.extend(&inputs.input_limb_0);
            self.input_limb_1.extend(&inputs.input_limb_1);
            for i in 0..10 {
                self.state_0[i].extend(&inputs.state_0[i]);
                self.state_1[i].extend(&inputs.state_1[i]);
                self.state_2[i].extend(&inputs.state_2[i]);
                self.state_3[i].extend(&inputs.state_3[i]);
            }
        }
        self.size = self.input_limb_0.size as u32;
    }
}

fn write_trace_cuda(
    input_limb_0: &BaseFieldVec,
    input_limb_1: &BaseFieldVec,
    state_0: &[BaseFieldVec; 10],
    state_1: &[BaseFieldVec; 10],
    state_2: &[BaseFieldVec; 10],
    state_3: &[BaseFieldVec; 10],
    n_rows: usize,
    log_size: u32,
    poseidon_round_keys_table: &[[BaseFieldVec; 10]; 3],
) -> CudaComponentTrace<N_TRACE_COLUMNS> {
    let trace = unsafe { CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size) };

    let state_0_ptrs = collect_ptrs!(state_0);
    let state_1_ptrs = collect_ptrs!(state_1);
    let state_2_ptrs = collect_ptrs!(state_2);
    let state_3_ptrs = collect_ptrs!(state_3);
    let trace_ptrs = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Flatten poseidon round keys table (3 x 10 = 30 columns)
    let round_keys_ptrs: Vec<*const u32> = poseidon_round_keys_table
        .iter()
        .flat_map(|arr| arr.iter().map(|v| v.device_ptr))
        .collect();

    unsafe {
        bindings_airs::poseidon_3_partial_rounds_chain_generate_trace(
            input_limb_0.device_ptr,
            input_limb_1.device_ptr,
            state_0_ptrs.as_ptr(),
            state_1_ptrs.as_ptr(),
            state_2_ptrs.as_ptr(),
            state_3_ptrs.as_ptr(),
            1 << log_size,
            n_rows as u32,
            trace_ptrs.as_ptr(),
            round_keys_ptrs.as_ptr(),
        );
    }

    trace
}

/// Adds range check multiplicities from the poseidon_3_partial_rounds_chain trace.
///
/// Calls a CUDA kernel that computes carry values from trace columns and directly
/// updates rc_4_4 and rc_4_4_4_4 multiplicity tables on GPU using atomicAdd.
fn add_to_multiplicities(
    trace: &CudaComponentTrace<N_TRACE_COLUMNS>,
    n_rows: u32,
    _cube_252_state: &cube_252_cuda::CudaClaimGenerator,
    range_check_4_4_state: &rc_4_4::CudaClaimGenerator,
    range_check_4_4_4_4_state: &rc_4_4_4_4::CudaClaimGenerator,
) {
    let trace_ptrs: Vec<*const u32> = trace.data.iter().map(|c| c.device_ptr).collect();

    unsafe {
        bindings_airs::poseidon_3_partial_rounds_chain_add_to_multiplicities(
            trace_ptrs.as_ptr(),
            n_rows,
            // cube_252 - not used by kernel but required by FFI signature
            std::ptr::null(),
            0,
            // poseidon_round_keys - not used by kernel
            std::ptr::null(),
            // range_check_252_width_27 - not used by kernel
            std::ptr::null(),
            0,
            // rc_4_4 multiplicities
            range_check_4_4_state.multiplicities.device_ptr as *const u32,
            range_check_4_4_state.log_size,
            // rc_4_4_4_4 multiplicities
            range_check_4_4_4_4_state.multiplicities.device_ptr as *const u32,
            range_check_4_4_4_4_state.log_size,
        );
    }
}

pub struct CudaInteractionClaimGenerator {
    pub log_size: u32,
    pub trace_data: [BaseFieldVec; N_TRACE_COLUMNS],
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;
        let trace_size = 1u32 << trace_log_size;

        // Allocate interaction trace columns (9 logup columns * 4 for qm31)
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size as usize))
            .collect_vec();

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let trace_ptrs = self.trace_data.iter().map(|c| c.device_ptr).collect_vec();
        let interaction_trace_ptrs = interaction_trace.iter().map(|c| c.device_ptr).collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_cube_252 =
                create_modified_lookup_for_cuda(lookup_elements, CUBE_252_RELATION_ID);
            let mod_round_keys =
                create_modified_lookup_for_cuda(lookup_elements, POSEIDON_ROUND_KEYS_RELATION_ID);
            let mod_rc_252 = create_modified_lookup_for_cuda(
                lookup_elements,
                RANGE_CHECK_252_WIDTH_27_RELATION_ID,
            );
            let mod_rc_4_4 =
                create_modified_lookup_for_cuda(lookup_elements, RANGE_CHECK_4_4_RELATION_ID);
            let mod_rc_4_4_4_4 =
                create_modified_lookup_for_cuda(lookup_elements, RC_4_4_4_4_RELATION_ID);
            let mod_p3prc = create_modified_lookup_for_cuda(
                lookup_elements,
                POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID,
            );

            bindings_airs::poseidon_3_partial_rounds_chain_generate_interaction_trace(
                trace_ptrs.as_ptr(),
                trace_size,
                // Lookup elements (each with relation-specific modified z)
                &mod_cube_252 as *const _ as *mut std::os::raw::c_void,
                &mod_round_keys as *const _ as *mut std::os::raw::c_void,
                &mod_rc_252 as *const _ as *mut std::os::raw::c_void,
                &mod_rc_4_4 as *const _ as *mut std::os::raw::c_void,
                &mod_rc_4_4_4_4 as *const _ as *mut std::os::raw::c_void,
                &mod_p3prc as *const _ as *mut std::os::raw::c_void,
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

        let domain = CanonicCoset::new(trace_log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
