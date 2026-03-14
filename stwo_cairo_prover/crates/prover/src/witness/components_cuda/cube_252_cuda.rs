#![allow(unused_parens)]

use cairo_air::components::cube_252::{Claim, InteractionClaim, N_TRACE_COLUMNS};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
#[allow(unused_imports)]
use stwo::core::fields::FieldExpOps;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use crate::witness::components_cuda::range_check::{rc_20, rc_9_9};
use crate::witness::prelude::*;

/// Number of interaction trace columns for cube_252.
/// 50 logup columns * 4 (for SecureField) = 200
pub const N_INTERACTION_TRACE_COLUMNS: usize = 50;

/// Input format: Width27 (10 x 27-bit limbs representing a Felt252)
pub type PackedInputType = PackedFelt252Width27;

/// CUDA packed input type - 10 columns for Width27 format
pub type CudaPackedInputType = [BaseFieldVec; 10];

// Macros for lookup data handling (following blake_g pattern)
macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}

macro_rules! init_subcomponent_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
        })
    };
}

macro_rules! collect_lookup_ptrs {
    ($lookup_data:expr, $field:ident) => {
        $lookup_data
            .$field
            .iter()
            .map(|x| x.device_ptr)
            .collect_vec()
    };
}

macro_rules! collect_sub_input_ptrs {
    ($sub_inputs:expr, $field:ident) => {
        $sub_inputs
            .$field
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec()
    };
}

macro_rules! collect_input_ptrs {
    ($input:expr) => {
        $input.iter().map(|x| x.device_ptr).collect_vec()
    };
}

// Type aliases for sub-component inputs (following blake_g pattern)
type Rc9_9CudaPackedInputType = [BaseFieldVec; 2];
type Rc19CudaPackedInputType = [BaseFieldVec; 1];

/// CUDA sub-component inputs for cube_252 (following blake_g pattern)
struct CudaSubComponentInputs {
    // range_check_9_9 variants: 2 elements each
    range_check_9_9: [Rc9_9CudaPackedInputType; 6],
    range_check_9_9_b: [Rc9_9CudaPackedInputType; 6],
    range_check_9_9_c: [Rc9_9CudaPackedInputType; 6],
    range_check_9_9_d: [Rc9_9CudaPackedInputType; 6],
    range_check_9_9_e: [Rc9_9CudaPackedInputType; 6],
    range_check_9_9_f: [Rc9_9CudaPackedInputType; 6],
    range_check_9_9_g: [Rc9_9CudaPackedInputType; 3],
    range_check_9_9_h: [Rc9_9CudaPackedInputType; 3],
    // range_check_19 variants: 1 element each
    range_check_19: [Rc19CudaPackedInputType; 8],
    range_check_19_b: [Rc19CudaPackedInputType; 8],
    range_check_19_c: [Rc19CudaPackedInputType; 8],
    range_check_19_d: [Rc19CudaPackedInputType; 6],
    range_check_19_e: [Rc19CudaPackedInputType; 6],
    range_check_19_f: [Rc19CudaPackedInputType; 6],
    range_check_19_g: [Rc19CudaPackedInputType; 6],
    range_check_19_h: [Rc19CudaPackedInputType; 8],
}

pub struct CudaClaimGenerator {
    /// Input data in Width27 format (10 columns)
    pub packed_inputs: [BaseFieldVec; 10],
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
            packed_inputs: std::array::from_fn(|_| BaseFieldVec::new_zeroes(1)),
            size: 1,
        }
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        // Range check 9_9 (all 8 relation indices)
        range_check_9_9_state: &rc_9_9::CudaClaimGenerator,
        // Range check 20 (all 8 relation indices)
        range_check_20_state: &rc_20::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let n_rows = self.size as usize;
        let log_size = n_rows.next_power_of_two().ilog2();
        let padded_size = 1usize << log_size;
        // Pad input buffers to power-of-two size (kernel expects padded_size elements)
        // GPU in-place padding — no download/upload roundtrip.
        const N_LANES: usize = 16;
        if n_rows < padded_size {
            for input in self.packed_inputs.iter_mut() {
                input.pad_with_cycle(n_rows, padded_size, N_LANES);
            }
        }

        // Generate trace, lookup data, and sub-component inputs
        let (trace, _lookup_data, sub_component_inputs) =
            write_trace_cuda(&self.packed_inputs, n_rows, log_size);

        // Add sub-component inputs to range check states.
        // rc_9_9 has 8 variants, each mapping to a different relation index.
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9, 0);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_b, 1);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_c, 2);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_d, 3);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_e, 4);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_f, 5);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_g, 6);
        range_check_9_9_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_9_9_h, 7);
        // rc_20 has 8 variants (legacy names rc_19*), each mapping to a different relation index.
        // NOTE: The CUDA kernel's carry-to-variant mapping is rotated by 1 from the SIMD code:
        //   CUDA a (carry_0,8,16,24) = SIMD b → relation 1
        //   CUDA b (carry_1,9,17,25) = SIMD c → relation 2
        //   CUDA c (carry_2,10,18,26) = SIMD d → relation 3
        //   CUDA d (carry_3,11,19) = SIMD e → relation 4
        //   CUDA e (carry_4,12,20) = SIMD f → relation 5
        //   CUDA f (carry_5,13,21) = SIMD g → relation 6
        //   CUDA g (carry_6,14,22) = SIMD h → relation 7
        //   CUDA h (k,carry_7,15,23) = SIMD a → relation 0
        range_check_20_state.add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19, 1);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_b, 2);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_c, 3);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_d, 4);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_e, 5);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_f, 6);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_g, 7);
        range_check_20_state
            .add_cuda_inputs_for_relation(&sub_component_inputs.range_check_19_h, 0);

        // Clone trace data for interaction generator
        let trace_data_clone: [BaseFieldVec; N_TRACE_COLUMNS] = trace.data.clone();

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                _n_rows: n_rows,
                log_size,
                trace_data: trace_data_clone,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {
        // Convert SIMD packed inputs to CUDA format
        // For each of the 10 limb positions, extract values from all lanes
        self.packed_inputs = (0..10)
            .map(|i| {
                let elements: Vec<M31> = packed_inputs
                    .iter()
                    .flat_map(|input| input.get_m31(i).to_array())
                    .collect();
                BaseFieldVec::from_vec(elements)
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Should have exactly 10 elements");

        self.size = self.packed_inputs[0].size as u32;
    }

    pub fn add_cuda_inputs(&mut self, inputs: &CudaPackedInputType) {
        if inputs[0].size == 0 {
            return; // Skip empty inputs
        }
        if self.size <= 1 {
            // First real data: replace initial placeholder zeros instead of extending.
            for (input_slice, new_slice) in self.packed_inputs.iter_mut().zip(inputs.iter()) {
                *input_slice = new_slice.clone();
            }
        } else {
            for (input_slice, new_slice) in self.packed_inputs.iter_mut().zip(inputs.iter()) {
                input_slice.extend(new_slice);
            }
        }
        self.size = self.packed_inputs[0].size as u32;
    }
}

/// CUDA lookup data for cube_252 (matches SIMD LookupData structure)
struct CudaLookupData {
    // Self lookup: 20 elements (10 input + 10 output limbs)
    cube_252_0: [BaseFieldVec; 20],
    // Range check 9_9 lookups: 2 elements each
    range_check_9_9_0: [BaseFieldVec; 2],
    range_check_9_9_1: [BaseFieldVec; 2],
    range_check_9_9_2: [BaseFieldVec; 2],
    range_check_9_9_3: [BaseFieldVec; 2],
    range_check_9_9_4: [BaseFieldVec; 2],
    range_check_9_9_5: [BaseFieldVec; 2],
    range_check_9_9_b_0: [BaseFieldVec; 2],
    range_check_9_9_b_1: [BaseFieldVec; 2],
    range_check_9_9_b_2: [BaseFieldVec; 2],
    range_check_9_9_b_3: [BaseFieldVec; 2],
    range_check_9_9_b_4: [BaseFieldVec; 2],
    range_check_9_9_b_5: [BaseFieldVec; 2],
    range_check_9_9_c_0: [BaseFieldVec; 2],
    range_check_9_9_c_1: [BaseFieldVec; 2],
    range_check_9_9_c_2: [BaseFieldVec; 2],
    range_check_9_9_c_3: [BaseFieldVec; 2],
    range_check_9_9_c_4: [BaseFieldVec; 2],
    range_check_9_9_c_5: [BaseFieldVec; 2],
    range_check_9_9_d_0: [BaseFieldVec; 2],
    range_check_9_9_d_1: [BaseFieldVec; 2],
    range_check_9_9_d_2: [BaseFieldVec; 2],
    range_check_9_9_d_3: [BaseFieldVec; 2],
    range_check_9_9_d_4: [BaseFieldVec; 2],
    range_check_9_9_d_5: [BaseFieldVec; 2],
    range_check_9_9_e_0: [BaseFieldVec; 2],
    range_check_9_9_e_1: [BaseFieldVec; 2],
    range_check_9_9_e_2: [BaseFieldVec; 2],
    range_check_9_9_e_3: [BaseFieldVec; 2],
    range_check_9_9_e_4: [BaseFieldVec; 2],
    range_check_9_9_e_5: [BaseFieldVec; 2],
    range_check_9_9_f_0: [BaseFieldVec; 2],
    range_check_9_9_f_1: [BaseFieldVec; 2],
    range_check_9_9_f_2: [BaseFieldVec; 2],
    range_check_9_9_f_3: [BaseFieldVec; 2],
    range_check_9_9_f_4: [BaseFieldVec; 2],
    range_check_9_9_f_5: [BaseFieldVec; 2],
    range_check_9_9_g_0: [BaseFieldVec; 2],
    range_check_9_9_g_1: [BaseFieldVec; 2],
    range_check_9_9_g_2: [BaseFieldVec; 2],
    range_check_9_9_h_0: [BaseFieldVec; 2],
    range_check_9_9_h_1: [BaseFieldVec; 2],
    range_check_9_9_h_2: [BaseFieldVec; 2],
    // Range check 19 lookups: 1 element each
    range_check_19_0: [BaseFieldVec; 1],
    range_check_19_1: [BaseFieldVec; 1],
    range_check_19_2: [BaseFieldVec; 1],
    range_check_19_3: [BaseFieldVec; 1],
    range_check_19_4: [BaseFieldVec; 1],
    range_check_19_5: [BaseFieldVec; 1],
    range_check_19_6: [BaseFieldVec; 1],
    range_check_19_7: [BaseFieldVec; 1],
    range_check_19_b_0: [BaseFieldVec; 1],
    range_check_19_b_1: [BaseFieldVec; 1],
    range_check_19_b_2: [BaseFieldVec; 1],
    range_check_19_b_3: [BaseFieldVec; 1],
    range_check_19_b_4: [BaseFieldVec; 1],
    range_check_19_b_5: [BaseFieldVec; 1],
    range_check_19_b_6: [BaseFieldVec; 1],
    range_check_19_b_7: [BaseFieldVec; 1],
    range_check_19_c_0: [BaseFieldVec; 1],
    range_check_19_c_1: [BaseFieldVec; 1],
    range_check_19_c_2: [BaseFieldVec; 1],
    range_check_19_c_3: [BaseFieldVec; 1],
    range_check_19_c_4: [BaseFieldVec; 1],
    range_check_19_c_5: [BaseFieldVec; 1],
    range_check_19_c_6: [BaseFieldVec; 1],
    range_check_19_c_7: [BaseFieldVec; 1],
    range_check_19_d_0: [BaseFieldVec; 1],
    range_check_19_d_1: [BaseFieldVec; 1],
    range_check_19_d_2: [BaseFieldVec; 1],
    range_check_19_d_3: [BaseFieldVec; 1],
    range_check_19_d_4: [BaseFieldVec; 1],
    range_check_19_d_5: [BaseFieldVec; 1],
    range_check_19_e_0: [BaseFieldVec; 1],
    range_check_19_e_1: [BaseFieldVec; 1],
    range_check_19_e_2: [BaseFieldVec; 1],
    range_check_19_e_3: [BaseFieldVec; 1],
    range_check_19_e_4: [BaseFieldVec; 1],
    range_check_19_e_5: [BaseFieldVec; 1],
    range_check_19_f_0: [BaseFieldVec; 1],
    range_check_19_f_1: [BaseFieldVec; 1],
    range_check_19_f_2: [BaseFieldVec; 1],
    range_check_19_f_3: [BaseFieldVec; 1],
    range_check_19_f_4: [BaseFieldVec; 1],
    range_check_19_f_5: [BaseFieldVec; 1],
    range_check_19_g_0: [BaseFieldVec; 1],
    range_check_19_g_1: [BaseFieldVec; 1],
    range_check_19_g_2: [BaseFieldVec; 1],
    range_check_19_g_3: [BaseFieldVec; 1],
    range_check_19_g_4: [BaseFieldVec; 1],
    range_check_19_g_5: [BaseFieldVec; 1],
    range_check_19_h_0: [BaseFieldVec; 1],
    range_check_19_h_1: [BaseFieldVec; 1],
    range_check_19_h_2: [BaseFieldVec; 1],
    range_check_19_h_3: [BaseFieldVec; 1],
    range_check_19_h_4: [BaseFieldVec; 1],
    range_check_19_h_5: [BaseFieldVec; 1],
    range_check_19_h_6: [BaseFieldVec; 1],
    range_check_19_h_7: [BaseFieldVec; 1],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    inputs: &[BaseFieldVec; 10],
    n_actual_rows: usize,
    log_size: u32,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let padded_size = 1usize << log_size;

    // Allocate trace, lookup data, and sub-component inputs
    let (mut trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                cube_252_0: init_lookup_array!(log_size),
                range_check_9_9_0: init_lookup_array!(log_size),
                range_check_9_9_1: init_lookup_array!(log_size),
                range_check_9_9_2: init_lookup_array!(log_size),
                range_check_9_9_3: init_lookup_array!(log_size),
                range_check_9_9_4: init_lookup_array!(log_size),
                range_check_9_9_5: init_lookup_array!(log_size),
                range_check_9_9_b_0: init_lookup_array!(log_size),
                range_check_9_9_b_1: init_lookup_array!(log_size),
                range_check_9_9_b_2: init_lookup_array!(log_size),
                range_check_9_9_b_3: init_lookup_array!(log_size),
                range_check_9_9_b_4: init_lookup_array!(log_size),
                range_check_9_9_b_5: init_lookup_array!(log_size),
                range_check_9_9_c_0: init_lookup_array!(log_size),
                range_check_9_9_c_1: init_lookup_array!(log_size),
                range_check_9_9_c_2: init_lookup_array!(log_size),
                range_check_9_9_c_3: init_lookup_array!(log_size),
                range_check_9_9_c_4: init_lookup_array!(log_size),
                range_check_9_9_c_5: init_lookup_array!(log_size),
                range_check_9_9_d_0: init_lookup_array!(log_size),
                range_check_9_9_d_1: init_lookup_array!(log_size),
                range_check_9_9_d_2: init_lookup_array!(log_size),
                range_check_9_9_d_3: init_lookup_array!(log_size),
                range_check_9_9_d_4: init_lookup_array!(log_size),
                range_check_9_9_d_5: init_lookup_array!(log_size),
                range_check_9_9_e_0: init_lookup_array!(log_size),
                range_check_9_9_e_1: init_lookup_array!(log_size),
                range_check_9_9_e_2: init_lookup_array!(log_size),
                range_check_9_9_e_3: init_lookup_array!(log_size),
                range_check_9_9_e_4: init_lookup_array!(log_size),
                range_check_9_9_e_5: init_lookup_array!(log_size),
                range_check_9_9_f_0: init_lookup_array!(log_size),
                range_check_9_9_f_1: init_lookup_array!(log_size),
                range_check_9_9_f_2: init_lookup_array!(log_size),
                range_check_9_9_f_3: init_lookup_array!(log_size),
                range_check_9_9_f_4: init_lookup_array!(log_size),
                range_check_9_9_f_5: init_lookup_array!(log_size),
                range_check_9_9_g_0: init_lookup_array!(log_size),
                range_check_9_9_g_1: init_lookup_array!(log_size),
                range_check_9_9_g_2: init_lookup_array!(log_size),
                range_check_9_9_h_0: init_lookup_array!(log_size),
                range_check_9_9_h_1: init_lookup_array!(log_size),
                range_check_9_9_h_2: init_lookup_array!(log_size),
                range_check_19_0: init_lookup_array!(log_size),
                range_check_19_1: init_lookup_array!(log_size),
                range_check_19_2: init_lookup_array!(log_size),
                range_check_19_3: init_lookup_array!(log_size),
                range_check_19_4: init_lookup_array!(log_size),
                range_check_19_5: init_lookup_array!(log_size),
                range_check_19_6: init_lookup_array!(log_size),
                range_check_19_7: init_lookup_array!(log_size),
                range_check_19_b_0: init_lookup_array!(log_size),
                range_check_19_b_1: init_lookup_array!(log_size),
                range_check_19_b_2: init_lookup_array!(log_size),
                range_check_19_b_3: init_lookup_array!(log_size),
                range_check_19_b_4: init_lookup_array!(log_size),
                range_check_19_b_5: init_lookup_array!(log_size),
                range_check_19_b_6: init_lookup_array!(log_size),
                range_check_19_b_7: init_lookup_array!(log_size),
                range_check_19_c_0: init_lookup_array!(log_size),
                range_check_19_c_1: init_lookup_array!(log_size),
                range_check_19_c_2: init_lookup_array!(log_size),
                range_check_19_c_3: init_lookup_array!(log_size),
                range_check_19_c_4: init_lookup_array!(log_size),
                range_check_19_c_5: init_lookup_array!(log_size),
                range_check_19_c_6: init_lookup_array!(log_size),
                range_check_19_c_7: init_lookup_array!(log_size),
                range_check_19_d_0: init_lookup_array!(log_size),
                range_check_19_d_1: init_lookup_array!(log_size),
                range_check_19_d_2: init_lookup_array!(log_size),
                range_check_19_d_3: init_lookup_array!(log_size),
                range_check_19_d_4: init_lookup_array!(log_size),
                range_check_19_d_5: init_lookup_array!(log_size),
                range_check_19_e_0: init_lookup_array!(log_size),
                range_check_19_e_1: init_lookup_array!(log_size),
                range_check_19_e_2: init_lookup_array!(log_size),
                range_check_19_e_3: init_lookup_array!(log_size),
                range_check_19_e_4: init_lookup_array!(log_size),
                range_check_19_e_5: init_lookup_array!(log_size),
                range_check_19_f_0: init_lookup_array!(log_size),
                range_check_19_f_1: init_lookup_array!(log_size),
                range_check_19_f_2: init_lookup_array!(log_size),
                range_check_19_f_3: init_lookup_array!(log_size),
                range_check_19_f_4: init_lookup_array!(log_size),
                range_check_19_f_5: init_lookup_array!(log_size),
                range_check_19_g_0: init_lookup_array!(log_size),
                range_check_19_g_1: init_lookup_array!(log_size),
                range_check_19_g_2: init_lookup_array!(log_size),
                range_check_19_g_3: init_lookup_array!(log_size),
                range_check_19_g_4: init_lookup_array!(log_size),
                range_check_19_g_5: init_lookup_array!(log_size),
                range_check_19_h_0: init_lookup_array!(log_size),
                range_check_19_h_1: init_lookup_array!(log_size),
                range_check_19_h_2: init_lookup_array!(log_size),
                range_check_19_h_3: init_lookup_array!(log_size),
                range_check_19_h_4: init_lookup_array!(log_size),
                range_check_19_h_5: init_lookup_array!(log_size),
                range_check_19_h_6: init_lookup_array!(log_size),
                range_check_19_h_7: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                range_check_9_9: init_subcomponent_array!(log_size),
                range_check_9_9_b: init_subcomponent_array!(log_size),
                range_check_9_9_c: init_subcomponent_array!(log_size),
                range_check_9_9_d: init_subcomponent_array!(log_size),
                range_check_9_9_e: init_subcomponent_array!(log_size),
                range_check_9_9_f: init_subcomponent_array!(log_size),
                range_check_9_9_g: init_subcomponent_array!(log_size),
                range_check_9_9_h: init_subcomponent_array!(log_size),
                range_check_19: init_subcomponent_array!(log_size),
                range_check_19_b: init_subcomponent_array!(log_size),
                range_check_19_c: init_subcomponent_array!(log_size),
                range_check_19_d: init_subcomponent_array!(log_size),
                range_check_19_e: init_subcomponent_array!(log_size),
                range_check_19_f: init_subcomponent_array!(log_size),
                range_check_19_g: init_subcomponent_array!(log_size),
                range_check_19_h: init_subcomponent_array!(log_size),
            },
        )
    };

    // Collect pointers and call CUDA kernel
    // (same as legacy -- the FFI binding signature is unchanged)
    let trace_ptrs = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    let input_ptrs = collect_input_ptrs!(inputs);
    let lookup_cube_252_0_ptrs = collect_lookup_ptrs!(lookup_data, cube_252_0);

    // Sub-component input pointers
    let sub_rc_9_9_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9);
    let sub_rc_9_9_b_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_b);
    let sub_rc_9_9_c_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_c);
    let sub_rc_9_9_d_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_d);
    let sub_rc_9_9_e_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_e);
    let sub_rc_9_9_f_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_f);
    let sub_rc_9_9_g_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_g);
    let sub_rc_9_9_h_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_9_9_h);
    let sub_rc_19_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19);
    let sub_rc_19_b_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_b);
    let sub_rc_19_c_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_c);
    let sub_rc_19_d_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_d);
    let sub_rc_19_e_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_e);
    let sub_rc_19_f_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_f);
    let sub_rc_19_g_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_g);
    let sub_rc_19_h_ptrs = collect_sub_input_ptrs!(sub_component_inputs, range_check_19_h);

    // Collect all lookup data pointers for kernel call
    // rc_9_9 variants: a..f have 6 indices, g,h have 3 indices
    let lk_rc99: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_2),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_3),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_4),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_5),
    ];
    let lk_rc99b: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_b_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_b_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_b_2),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_b_3),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_b_4),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_b_5),
    ];
    let lk_rc99c: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_c_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_c_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_c_2),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_c_3),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_c_4),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_c_5),
    ];
    let lk_rc99d: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_d_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_d_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_d_2),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_d_3),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_d_4),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_d_5),
    ];
    let lk_rc99e: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_e_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_e_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_e_2),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_e_3),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_e_4),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_e_5),
    ];
    let lk_rc99f: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_f_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_f_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_f_2),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_f_3),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_f_4),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_f_5),
    ];
    let lk_rc99g: [Vec<_>; 3] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_g_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_g_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_g_2),
    ];
    let lk_rc99h: [Vec<_>; 3] = [
        collect_lookup_ptrs!(lookup_data, range_check_9_9_h_0),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_h_1),
        collect_lookup_ptrs!(lookup_data, range_check_9_9_h_2),
    ];
    // rc_19 variants: a,b,c,h have 8 indices; d,e,f,g have 6 indices
    let lk_rc19: [Vec<_>; 8] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_5),
        collect_lookup_ptrs!(lookup_data, range_check_19_6),
        collect_lookup_ptrs!(lookup_data, range_check_19_7),
    ];
    let lk_rc19b: [Vec<_>; 8] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_b_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_5),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_6),
        collect_lookup_ptrs!(lookup_data, range_check_19_b_7),
    ];
    let lk_rc19c: [Vec<_>; 8] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_c_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_5),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_6),
        collect_lookup_ptrs!(lookup_data, range_check_19_c_7),
    ];
    let lk_rc19d: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_d_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_d_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_d_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_d_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_d_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_d_5),
    ];
    let lk_rc19e: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_e_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_e_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_e_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_e_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_e_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_e_5),
    ];
    let lk_rc19f: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_f_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_f_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_f_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_f_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_f_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_f_5),
    ];
    let lk_rc19g: [Vec<_>; 6] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_g_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_g_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_g_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_g_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_g_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_g_5),
    ];
    let lk_rc19h: [Vec<_>; 8] = [
        collect_lookup_ptrs!(lookup_data, range_check_19_h_0),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_1),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_2),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_3),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_4),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_5),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_6),
        collect_lookup_ptrs!(lookup_data, range_check_19_h_7),
    ];

    unsafe {
        bindings_airs::generate_cube_252_trace(
            trace_ptrs.as_ptr(),
            lookup_cube_252_0_ptrs.as_ptr(),
            // lookup rc_9_9 (a): 6 variants
            lk_rc99[0].as_ptr(),
            lk_rc99[1].as_ptr(),
            lk_rc99[2].as_ptr(),
            lk_rc99[3].as_ptr(),
            lk_rc99[4].as_ptr(),
            lk_rc99[5].as_ptr(),
            // lookup rc_9_9_b: 6 variants
            lk_rc99b[0].as_ptr(),
            lk_rc99b[1].as_ptr(),
            lk_rc99b[2].as_ptr(),
            lk_rc99b[3].as_ptr(),
            lk_rc99b[4].as_ptr(),
            lk_rc99b[5].as_ptr(),
            // lookup rc_9_9_c: 6 variants
            lk_rc99c[0].as_ptr(),
            lk_rc99c[1].as_ptr(),
            lk_rc99c[2].as_ptr(),
            lk_rc99c[3].as_ptr(),
            lk_rc99c[4].as_ptr(),
            lk_rc99c[5].as_ptr(),
            // lookup rc_9_9_d: 6 variants
            lk_rc99d[0].as_ptr(),
            lk_rc99d[1].as_ptr(),
            lk_rc99d[2].as_ptr(),
            lk_rc99d[3].as_ptr(),
            lk_rc99d[4].as_ptr(),
            lk_rc99d[5].as_ptr(),
            // lookup rc_9_9_e: 6 variants
            lk_rc99e[0].as_ptr(),
            lk_rc99e[1].as_ptr(),
            lk_rc99e[2].as_ptr(),
            lk_rc99e[3].as_ptr(),
            lk_rc99e[4].as_ptr(),
            lk_rc99e[5].as_ptr(),
            // lookup rc_9_9_f: 6 variants
            lk_rc99f[0].as_ptr(),
            lk_rc99f[1].as_ptr(),
            lk_rc99f[2].as_ptr(),
            lk_rc99f[3].as_ptr(),
            lk_rc99f[4].as_ptr(),
            lk_rc99f[5].as_ptr(),
            // lookup rc_9_9_g: 3 variants
            lk_rc99g[0].as_ptr(),
            lk_rc99g[1].as_ptr(),
            lk_rc99g[2].as_ptr(),
            // lookup rc_9_9_h: 3 variants
            lk_rc99h[0].as_ptr(),
            lk_rc99h[1].as_ptr(),
            lk_rc99h[2].as_ptr(),
            // lookup rc_19 (a): 8 variants
            lk_rc19[0].as_ptr(),
            lk_rc19[1].as_ptr(),
            lk_rc19[2].as_ptr(),
            lk_rc19[3].as_ptr(),
            lk_rc19[4].as_ptr(),
            lk_rc19[5].as_ptr(),
            lk_rc19[6].as_ptr(),
            lk_rc19[7].as_ptr(),
            // lookup rc_19_b: 8 variants
            lk_rc19b[0].as_ptr(),
            lk_rc19b[1].as_ptr(),
            lk_rc19b[2].as_ptr(),
            lk_rc19b[3].as_ptr(),
            lk_rc19b[4].as_ptr(),
            lk_rc19b[5].as_ptr(),
            lk_rc19b[6].as_ptr(),
            lk_rc19b[7].as_ptr(),
            // lookup rc_19_c: 8 variants
            lk_rc19c[0].as_ptr(),
            lk_rc19c[1].as_ptr(),
            lk_rc19c[2].as_ptr(),
            lk_rc19c[3].as_ptr(),
            lk_rc19c[4].as_ptr(),
            lk_rc19c[5].as_ptr(),
            lk_rc19c[6].as_ptr(),
            lk_rc19c[7].as_ptr(),
            // lookup rc_19_d: 6 variants
            lk_rc19d[0].as_ptr(),
            lk_rc19d[1].as_ptr(),
            lk_rc19d[2].as_ptr(),
            lk_rc19d[3].as_ptr(),
            lk_rc19d[4].as_ptr(),
            lk_rc19d[5].as_ptr(),
            // lookup rc_19_e: 6 variants
            lk_rc19e[0].as_ptr(),
            lk_rc19e[1].as_ptr(),
            lk_rc19e[2].as_ptr(),
            lk_rc19e[3].as_ptr(),
            lk_rc19e[4].as_ptr(),
            lk_rc19e[5].as_ptr(),
            // lookup rc_19_f: 6 variants
            lk_rc19f[0].as_ptr(),
            lk_rc19f[1].as_ptr(),
            lk_rc19f[2].as_ptr(),
            lk_rc19f[3].as_ptr(),
            lk_rc19f[4].as_ptr(),
            lk_rc19f[5].as_ptr(),
            // lookup rc_19_g: 6 variants
            lk_rc19g[0].as_ptr(),
            lk_rc19g[1].as_ptr(),
            lk_rc19g[2].as_ptr(),
            lk_rc19g[3].as_ptr(),
            lk_rc19g[4].as_ptr(),
            lk_rc19g[5].as_ptr(),
            // lookup rc_19_h: 8 variants
            lk_rc19h[0].as_ptr(),
            lk_rc19h[1].as_ptr(),
            lk_rc19h[2].as_ptr(),
            lk_rc19h[3].as_ptr(),
            lk_rc19h[4].as_ptr(),
            lk_rc19h[5].as_ptr(),
            lk_rc19h[6].as_ptr(),
            lk_rc19h[7].as_ptr(),
            // Sub-component inputs
            sub_rc_9_9_ptrs.as_ptr(),
            sub_rc_9_9_b_ptrs.as_ptr(),
            sub_rc_9_9_c_ptrs.as_ptr(),
            sub_rc_9_9_d_ptrs.as_ptr(),
            sub_rc_9_9_e_ptrs.as_ptr(),
            sub_rc_9_9_f_ptrs.as_ptr(),
            sub_rc_9_9_g_ptrs.as_ptr(),
            sub_rc_9_9_h_ptrs.as_ptr(),
            sub_rc_19_ptrs.as_ptr(),
            sub_rc_19_b_ptrs.as_ptr(),
            sub_rc_19_c_ptrs.as_ptr(),
            sub_rc_19_d_ptrs.as_ptr(),
            sub_rc_19_e_ptrs.as_ptr(),
            sub_rc_19_f_ptrs.as_ptr(),
            sub_rc_19_g_ptrs.as_ptr(),
            sub_rc_19_h_ptrs.as_ptr(),
            // Inputs and trace size
            input_ptrs.as_ptr(),
            log_size,
        );
    }

    // Fix enabler column (140): set to 0 for padding rows
    // GPU in-place zero fill — no download/upload roundtrip.
    if n_actual_rows < padded_size {
        trace.data[140].fill_zero_from(n_actual_rows);
    }

    (trace, lookup_data, sub_component_inputs)
}

pub struct CudaInteractionClaimGenerator {
    pub _n_rows: usize,
    pub log_size: u32,
    // Trace data for interaction trace generation
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

        // Allocate interaction trace columns (50 logup columns * 4 for qm31)
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(trace_size as usize))
            .collect_vec();

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let trace_ptrs = self.trace_data.iter().map(|c| c.device_ptr).collect_vec();
        let interaction_trace_ptrs = interaction_trace.iter().map(|c| c.device_ptr).collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;

            // 1 cube_252 self-relation
            let mod_cube_252 =
                create_modified_lookup_for_cuda(lookup_elements, CUBE_252_RELATION_ID);
            // 8 rc_19 (rc_20) variants
            let mod_rc_19 = create_modified_lookup_for_cuda(lookup_elements, RC_19_RELATION_ID);
            let mod_rc_19_b = create_modified_lookup_for_cuda(lookup_elements, RC_19_B_RELATION_ID);
            let mod_rc_19_c = create_modified_lookup_for_cuda(lookup_elements, RC_19_C_RELATION_ID);
            let mod_rc_19_d = create_modified_lookup_for_cuda(lookup_elements, RC_19_D_RELATION_ID);
            let mod_rc_19_e = create_modified_lookup_for_cuda(lookup_elements, RC_19_E_RELATION_ID);
            let mod_rc_19_f = create_modified_lookup_for_cuda(lookup_elements, RC_19_F_RELATION_ID);
            let mod_rc_19_g = create_modified_lookup_for_cuda(lookup_elements, RC_19_G_RELATION_ID);
            let mod_rc_19_h = create_modified_lookup_for_cuda(lookup_elements, RC_19_H_RELATION_ID);
            // 8 rc_9_9 variants
            let mod_rc_9_9 =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[0]);
            let mod_rc_9_9_b =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[1]);
            let mod_rc_9_9_c =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[2]);
            let mod_rc_9_9_d =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[3]);
            let mod_rc_9_9_e =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[4]);
            let mod_rc_9_9_f =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[5]);
            let mod_rc_9_9_g =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[6]);
            let mod_rc_9_9_h =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[7]);

            bindings_airs::generate_cube_252_interaction_trace(
                trace_ptrs.as_ptr(),
                trace_size,
                // Lookup elements (each with relation-specific modified z)
                &mod_cube_252 as *const _ as *mut std::os::raw::c_void,
                // Rotated by 1 to match CUDA carry-to-variant mapping.
                // CUDA slot a processes SIMD variant b's carries, etc.
                &mod_rc_19_b as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19_c as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19_d as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19_e as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19_f as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19_g as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19_h as *const _ as *mut std::os::raw::c_void,
                &mod_rc_19 as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9 as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_b as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_c as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_d as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_e as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_f as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_g as *const _ as *mut std::os::raw::c_void,
                &mod_rc_9_9_h as *const _ as *mut std::os::raw::c_void,
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
