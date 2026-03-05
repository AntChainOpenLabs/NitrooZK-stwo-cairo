#![allow(unused_parens)]

use cairo_air::components::poseidon_full_round_chain::{Claim, InteractionClaim, N_TRACE_COLUMNS};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use crate::witness::components_cuda::cube_252_cuda;
use crate::witness::components_cuda::range_check::rc_3_3_3_3_3;
use crate::witness::prelude::*;

/// Number of interaction trace columns for poseidon_full_round_chain.
/// 6 logup columns * 4 (for SecureField) = 24
pub const N_INTERACTION_TRACE_COLUMNS: usize = 6;

/// Input format: (index, round_number, [state0, state1, state2] in Width27)
pub type PackedInputType = (PackedM31, PackedM31, [PackedFelt252Width27; 3]);

/// CUDA packed input type:
/// - input_limb_0: index (1 column)
/// - input_limb_1: round_number (1 column)
/// - state_0: 10 columns (Width27 format)
/// - state_1: 10 columns (Width27 format)
/// - state_2: 10 columns (Width27 format)
pub struct CudaPackedInputType {
    pub input_limb_0: BaseFieldVec,
    pub input_limb_1: BaseFieldVec,
    pub state_0: [BaseFieldVec; 10],
    pub state_1: [BaseFieldVec; 10],
    pub state_2: [BaseFieldVec; 10],
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
        // Range check 3_3_3_3_3
        range_check_3_3_3_3_3_state: &rc_3_3_3_3_3::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let n_rows = self.size as usize;
        let log_size = n_rows.next_power_of_two().ilog2();
        let padded_size = 1usize << log_size;
        println!(
            "poseidon_full_round_chain write_trace: n_rows:{}, trace_log_size: {}, padded_size: {}",
            n_rows, log_size, padded_size
        );

        // Pad input arrays to next power of 2 (CUDA kernel expects this)
        // IMPORTANT: Must pad by replicating the first packed input (first 16 rows) to match SIMD
        // behavior
        const N_LANES: usize = 16;
        if n_rows < padded_size {
            let padding_count = padded_size - n_rows;

            let pad_with_first_packed = |vec: &mut BaseFieldVec| {
                let input_cpu = vec.to_cpu();
                let first_packed: Vec<M31> = input_cpu[0..N_LANES].to_vec();
                let padding_vec: Vec<M31> = first_packed
                    .iter()
                    .cycle()
                    .take(padding_count)
                    .cloned()
                    .collect();
                let padding = BaseFieldVec::from_vec(padding_vec);
                vec.extend(&padding);
            };

            pad_with_first_packed(&mut self.input_limb_0);
            pad_with_first_packed(&mut self.input_limb_1);
            for i in 0..10 {
                pad_with_first_packed(&mut self.state_0[i]);
                pad_with_first_packed(&mut self.state_1[i]);
                pad_with_first_packed(&mut self.state_2[i]);
            }
        }

        // Generate trace
        let mut trace = write_trace_cuda(
            &self.input_limb_0,
            &self.input_limb_1,
            &self.state_0,
            &self.state_1,
            &self.state_2,
            n_rows,
            log_size,
            poseidon_round_keys_table,
        );

        // Fix enabler column (125): the CUDA kernel sets enabler=1 for ALL padded rows,
        // but padding rows should have enabler=0 to match SIMD behavior.
        if n_rows < padded_size {
            let mut enabler_cpu = trace.data[125].to_cpu();
            for i in n_rows..padded_size {
                enabler_cpu[i] = M31::from_u32_unchecked(0);
            }
            trace.data[125] = BaseFieldVec::from_vec(enabler_cpu);
        }

        // Add cube_252 inputs (3 per row: state_0, state_1, state_2)
        cube_252_state.add_cuda_inputs(&self.state_0);
        cube_252_state.add_cuda_inputs(&self.state_1);
        cube_252_state.add_cuda_inputs(&self.state_2);

        // Add to range check multiplicities
        // IMPORTANT: Must use padded_size (not n_rows) to match SIMD behavior where
        // padding rows also contribute to range check multiplicities
        add_to_multiplicities(
            &trace,
            padded_size as u32,
            cube_252_state,
            range_check_3_3_3_3_3_state,
        );

        // Clone trace data for interaction generator before consuming for evals
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

        // Convert state arrays
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
            }
        } else {
            self.input_limb_0.extend(&inputs.input_limb_0);
            self.input_limb_1.extend(&inputs.input_limb_1);
            for i in 0..10 {
                self.state_0[i].extend(&inputs.state_0[i]);
                self.state_1[i].extend(&inputs.state_1[i]);
                self.state_2[i].extend(&inputs.state_2[i]);
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
    _n_rows: usize,
    log_size: u32,
    poseidon_round_keys_table: &[[BaseFieldVec; 10]; 3],
) -> CudaComponentTrace<N_TRACE_COLUMNS> {
    let trace = unsafe { CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size) };

    let state_0_ptrs = collect_ptrs!(state_0);
    let state_1_ptrs = collect_ptrs!(state_1);
    let state_2_ptrs = collect_ptrs!(state_2);
    let trace_ptrs = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Flatten poseidon round keys table (3 x 10 = 30 columns)
    let round_keys_ptrs: Vec<*const u32> = poseidon_round_keys_table
        .iter()
        .flat_map(|arr| arr.iter().map(|v| v.device_ptr))
        .collect();

    unsafe {
        bindings_airs::poseidon_full_round_chain_generate_trace(
            input_limb_0.device_ptr,
            input_limb_1.device_ptr,
            state_0_ptrs.as_ptr(),
            state_1_ptrs.as_ptr(),
            state_2_ptrs.as_ptr(),
            1 << log_size,
            trace_ptrs.as_ptr(),
            round_keys_ptrs.as_ptr(),
        );
    }

    trace
}

/// Adds range check multiplicities from the poseidon_full_round_chain trace.
///
/// Downloads trace columns and computes carry values to populate rc_3_3_3_3_3 multiplicities.
fn add_to_multiplicities(
    trace: &CudaComponentTrace<N_TRACE_COLUMNS>,
    n_rows: u32,
    _cube_252_state: &cube_252_cuda::CudaClaimGenerator,
    range_check_3_3_3_3_3_state: &rc_3_3_3_3_3::CudaClaimGenerator,
) {
    let n = n_rows as usize;

    let get_col = |col_idx: usize| -> Vec<M31> { trace.data[col_idx].to_cpu() };

    let cube0_cols: Vec<Vec<M31>> = (32..42).map(|i| get_col(i)).collect();
    let cube1_cols: Vec<Vec<M31>> = (42..52).map(|i| get_col(i)).collect();
    let cube2_cols: Vec<Vec<M31>> = (52..62).map(|i| get_col(i)).collect();
    let key0_cols: Vec<Vec<M31>> = (62..72).map(|i| get_col(i)).collect();
    let key1_cols: Vec<Vec<M31>> = (72..82).map(|i| get_col(i)).collect();
    let key2_cols: Vec<Vec<M31>> = (82..92).map(|i| get_col(i)).collect();
    let combo0_cols: Vec<Vec<M31>> = (92..102).map(|i| get_col(i)).collect();
    let p_coef_0_col = get_col(102);
    let combo1_cols: Vec<Vec<M31>> = (103..113).map(|i| get_col(i)).collect();
    let p_coef_1_col = get_col(113);
    let combo2_cols: Vec<Vec<M31>> = (114..124).map(|i| get_col(i)).collect();
    let p_coef_2_col = get_col(124);

    let mut rc_3_3_3_3_3_inputs: [[Vec<M31>; 5]; 6] = Default::default();
    for set in &mut rc_3_3_3_3_3_inputs {
        for v in set.iter_mut() {
            *v = Vec::with_capacity(n);
        }
    }

    let m31_1 = M31::from(1u32);
    let m31_2 = M31::from(2u32);
    let m31_3 = M31::from(3u32);
    let m31_16 = M31::from(16u32);
    let m31_136 = M31::from(136u32);

    for row in 0..n {
        // ========== Linear Combination 0: Coefs (3, 1, 1, 1) ==========
        let p_coef_0 = p_coef_0_col[row];
        let val0_0 = m31_3 * cube0_cols[0][row]
            + cube1_cols[0][row]
            + cube2_cols[0][row]
            + key0_cols[0][row]
            - combo0_cols[0][row];
        let carry_0_0 = (val0_0 - p_coef_0) * m31_16;

        let mut carries_0 = [M31::from(0u32); 9];
        carries_0[0] = carry_0_0;
        for i in 1..7 {
            let val = m31_3 * cube0_cols[i][row]
                + cube1_cols[i][row]
                + cube2_cols[i][row]
                + key0_cols[i][row]
                - combo0_cols[i][row];
            carries_0[i] = (carries_0[i - 1] + val) * m31_16;
        }

        let val7_0 = m31_3 * cube0_cols[7][row]
            + cube1_cols[7][row]
            + cube2_cols[7][row]
            + key0_cols[7][row]
            - combo0_cols[7][row];
        carries_0[7] = (carries_0[6] + val7_0 - p_coef_0 * m31_136) * m31_16;

        let val8_0 = m31_3 * cube0_cols[8][row]
            + cube1_cols[8][row]
            + cube2_cols[8][row]
            + key0_cols[8][row]
            - combo0_cols[8][row];
        carries_0[8] = (carries_0[7] + val8_0) * m31_16;

        rc_3_3_3_3_3_inputs[0][0].push(p_coef_0 + m31_1);
        rc_3_3_3_3_3_inputs[0][1].push(carries_0[0] + m31_1);
        rc_3_3_3_3_3_inputs[0][2].push(carries_0[1] + m31_1);
        rc_3_3_3_3_3_inputs[0][3].push(carries_0[2] + m31_1);
        rc_3_3_3_3_3_inputs[0][4].push(carries_0[3] + m31_1);

        rc_3_3_3_3_3_inputs[1][0].push(carries_0[4] + m31_1);
        rc_3_3_3_3_3_inputs[1][1].push(carries_0[5] + m31_1);
        rc_3_3_3_3_3_inputs[1][2].push(carries_0[6] + m31_1);
        rc_3_3_3_3_3_inputs[1][3].push(carries_0[7] + m31_1);
        rc_3_3_3_3_3_inputs[1][4].push(carries_0[8] + m31_1);

        // ========== Linear Combination 1: Coefs (1, -1, 1, 1) ==========
        let p_coef_1 = p_coef_1_col[row];
        let val0_1 =
            cube0_cols[0][row] - cube1_cols[0][row] + cube2_cols[0][row] + key1_cols[0][row]
                - combo1_cols[0][row];
        let carry_0_1 = (val0_1 - p_coef_1) * m31_16;

        let mut carries_1 = [M31::from(0u32); 9];
        carries_1[0] = carry_0_1;
        for i in 1..7 {
            let val =
                cube0_cols[i][row] - cube1_cols[i][row] + cube2_cols[i][row] + key1_cols[i][row]
                    - combo1_cols[i][row];
            carries_1[i] = (carries_1[i - 1] + val) * m31_16;
        }

        let val7_1 =
            cube0_cols[7][row] - cube1_cols[7][row] + cube2_cols[7][row] + key1_cols[7][row]
                - combo1_cols[7][row];
        carries_1[7] = (carries_1[6] + val7_1 - p_coef_1 * m31_136) * m31_16;

        let val8_1 =
            cube0_cols[8][row] - cube1_cols[8][row] + cube2_cols[8][row] + key1_cols[8][row]
                - combo1_cols[8][row];
        carries_1[8] = (carries_1[7] + val8_1) * m31_16;

        rc_3_3_3_3_3_inputs[2][0].push(p_coef_1 + m31_2);
        rc_3_3_3_3_3_inputs[2][1].push(carries_1[0] + m31_2);
        rc_3_3_3_3_3_inputs[2][2].push(carries_1[1] + m31_2);
        rc_3_3_3_3_3_inputs[2][3].push(carries_1[2] + m31_2);
        rc_3_3_3_3_3_inputs[2][4].push(carries_1[3] + m31_2);

        rc_3_3_3_3_3_inputs[3][0].push(carries_1[4] + m31_2);
        rc_3_3_3_3_3_inputs[3][1].push(carries_1[5] + m31_2);
        rc_3_3_3_3_3_inputs[3][2].push(carries_1[6] + m31_2);
        rc_3_3_3_3_3_inputs[3][3].push(carries_1[7] + m31_2);
        rc_3_3_3_3_3_inputs[3][4].push(carries_1[8] + m31_2);

        // ========== Linear Combination 2: Coefs (1, 1, -2, 1) ==========
        let p_coef_2 = p_coef_2_col[row];
        let val0_2 = cube0_cols[0][row] + cube1_cols[0][row] - m31_2 * cube2_cols[0][row]
            + key2_cols[0][row]
            - combo2_cols[0][row];
        let carry_0_2 = (val0_2 - p_coef_2) * m31_16;

        let mut carries_2 = [M31::from(0u32); 9];
        carries_2[0] = carry_0_2;
        for i in 1..7 {
            let val = cube0_cols[i][row] + cube1_cols[i][row] - m31_2 * cube2_cols[i][row]
                + key2_cols[i][row]
                - combo2_cols[i][row];
            carries_2[i] = (carries_2[i - 1] + val) * m31_16;
        }

        let val7_2 = cube0_cols[7][row] + cube1_cols[7][row] - m31_2 * cube2_cols[7][row]
            + key2_cols[7][row]
            - combo2_cols[7][row];
        carries_2[7] = (carries_2[6] + val7_2 - p_coef_2 * m31_136) * m31_16;

        let val8_2 = cube0_cols[8][row] + cube1_cols[8][row] - m31_2 * cube2_cols[8][row]
            + key2_cols[8][row]
            - combo2_cols[8][row];
        carries_2[8] = (carries_2[7] + val8_2) * m31_16;

        rc_3_3_3_3_3_inputs[4][0].push(p_coef_2 + m31_3);
        rc_3_3_3_3_3_inputs[4][1].push(carries_2[0] + m31_3);
        rc_3_3_3_3_3_inputs[4][2].push(carries_2[1] + m31_3);
        rc_3_3_3_3_3_inputs[4][3].push(carries_2[2] + m31_3);
        rc_3_3_3_3_3_inputs[4][4].push(carries_2[3] + m31_3);

        rc_3_3_3_3_3_inputs[5][0].push(carries_2[4] + m31_3);
        rc_3_3_3_3_3_inputs[5][1].push(carries_2[5] + m31_3);
        rc_3_3_3_3_3_inputs[5][2].push(carries_2[6] + m31_3);
        rc_3_3_3_3_3_inputs[5][3].push(carries_2[7] + m31_3);
        rc_3_3_3_3_3_inputs[5][4].push(carries_2[8] + m31_3);
    }

    // Convert to BaseFieldVec and add to range check generator
    let cuda_rc_3_3_3_3_3_inputs: [[BaseFieldVec; 5]; 6] =
        rc_3_3_3_3_3_inputs.map(|arr| arr.map(|v| BaseFieldVec::from_vec(v)));

    range_check_3_3_3_3_3_state.add_cuda_inputs(&cuda_rc_3_3_3_3_3_inputs);
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

        // Allocate interaction trace columns (6 logup columns * 4 for qm31)
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
            let mod_rc_3_3_3_3_3 =
                create_modified_lookup_for_cuda(lookup_elements, RANGE_CHECK_3_3_3_3_3_RELATION_ID);
            let mod_pfrc = create_modified_lookup_for_cuda(
                lookup_elements,
                POSEIDON_FULL_ROUND_CHAIN_RELATION_ID,
            );

            bindings_airs::poseidon_full_round_chain_generate_interaction_trace(
                trace_ptrs.as_ptr(),
                trace_size,
                // Lookup elements (each with relation-specific modified z)
                &mod_cube_252 as *const _ as *mut std::os::raw::c_void,
                &mod_round_keys as *const _ as *mut std::os::raw::c_void,
                &mod_rc_3_3_3_3_3 as *const _ as *mut std::os::raw::c_void,
                &mod_pfrc as *const _ as *mut std::os::raw::c_void,
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

        println!(
            "poseidon_full_round_chain cuda claimed_sum: {:?}",
            claimed_sum
        );
        InteractionClaim { claimed_sum }
    }
}
