#![allow(unused_parens)]

use cairo_air::components::blake_g::{Claim, InteractionClaim, N_TRACE_COLUMNS};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Col, Column};
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use super::verify_bitwise_xor::{
    vbx_12, vbx_4, vbx_7, vbx_8, vbx_8_b, vbx_9, CudaPackedInputType as VbxCudaPackedInputType,
};
use crate::witness::prelude::*;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 9;

pub type PackedInputType = [PackedUInt32; 6];
pub type CudaPackedInputType = [Uint32Vec; 6];

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

pub struct CudaClaimGenerator {
    pub packed_inputs: [Uint32Vec; 6],
    pub size: u32,
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            packed_inputs: std::array::from_fn(|_| Uint32Vec::new_zeroes(1)),
            size: 1,
        }
    }
    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        verify_bitwise_xor_12_state: &vbx_12::CudaClaimGenerator,
        verify_bitwise_xor_4_state: &vbx_4::CudaClaimGenerator,
        verify_bitwise_xor_7_state: &vbx_7::CudaClaimGenerator,
        verify_bitwise_xor_8_state: &vbx_8::CudaClaimGenerator,
        verify_bitwise_xor_8_b_state: &vbx_8_b::CudaClaimGenerator,
        verify_bitwise_xor_9_state: &vbx_9::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        assert!(!self.packed_inputs.is_empty());
        // n_rows is the actual number of data rows, stored in self.size
        // (packed_inputs.len() would give 6, the array size, not the row count)
        let n_rows: usize = self.size as usize;
        // Use next_power_of_two().ilog2() to ensure we don't truncate when size is not a power of 2
        let padded_size = n_rows.next_power_of_two();
        let log_size = padded_size.ilog2();
        // Pad inputs to next power of 2 by repeating first row (like SIMD version does)
        // GPU in-place padding — no download/upload roundtrip.
        // blake_g uses resize with a single scalar value, which is equivalent to
        // cycling with cycle_len=1 (repeating the first element).
        if padded_size > n_rows {
            for i in 0..6 {
                self.packed_inputs[i].pad_with_cycle(n_rows, padded_size, 1);
            }
        }

        let (trace, lookup_data, sub_component_inputs) =
            write_trace_cuda(self.packed_inputs, n_rows);

        verify_bitwise_xor_8_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
        verify_bitwise_xor_8_b_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8_b);
        verify_bitwise_xor_12_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_12);
        verify_bitwise_xor_4_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_4);
        verify_bitwise_xor_7_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_7);
        verify_bitwise_xor_9_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_9);

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {
        // println!("packed_inputs: {:?}", packed_inputs);

        self.packed_inputs = (0..6)
            .map(|i| {
                let elements: Vec<_> = packed_inputs
                    .iter()
                    .flat_map(|input| input[i].as_array())
                    .collect();
                Uint32Vec::from_vec(unsafe { std::mem::transmute(elements) })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Inputs should have exactly 6 elements");

        // for i in 0..self.packed_inputs.len() {
        //     println!("self.packed_inputs[{}]: {:?}", i, self.packed_inputs[i].to_vec());
        // }
        self.size = self.packed_inputs[0].size as u32;
    }

    pub fn add_cuda_inputs(&mut self, inputs: &[CudaPackedInputType]) {
        // On first call, replace the initial zeroed data instead of extending
        let mut first = self.size <= 1;
        for input in inputs {
            if first {
                // Replace initial dummy data
                for (input_slice, new_slice) in self.packed_inputs.iter_mut().zip(input.iter()) {
                    *input_slice = new_slice.clone();
                }
                self.size = input[0].size as u32;
                first = false;
            } else {
                // Extend existing data
                for (input_slice, new_slice) in self.packed_inputs.iter_mut().zip(input.iter()) {
                    input_slice.extend(new_slice);
                }
                self.size += input[0].size as u32;
            }
        }
    }
}

struct CudaSubComponentInputs {
    verify_bitwise_xor_8: [VbxCudaPackedInputType; 4],
    verify_bitwise_xor_8_b: [VbxCudaPackedInputType; 4],
    verify_bitwise_xor_12: [vbx_12::CudaPackedInputType; 2],
    verify_bitwise_xor_4: [VbxCudaPackedInputType; 2],
    verify_bitwise_xor_7: [VbxCudaPackedInputType; 2],
    verify_bitwise_xor_9: [VbxCudaPackedInputType; 2],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    inputs: [Uint32Vec; 6],
    n_rows: usize,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    // n_rows is the actual row count from self.size, NOT inputs.len() which would be 6 (array
    // length) Use next_power_of_two().ilog2() to ensure we don't truncate when size is not a
    // power of 2
    let log_size = n_rows.next_power_of_two().ilog2();
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                blake_g_0: init_lookup_array!(log_size),
                verify_bitwise_xor_12_0: init_lookup_array!(log_size),
                verify_bitwise_xor_12_1: init_lookup_array!(log_size),
                verify_bitwise_xor_4_0: init_lookup_array!(log_size),
                verify_bitwise_xor_4_1: init_lookup_array!(log_size),
                verify_bitwise_xor_7_0: init_lookup_array!(log_size),
                verify_bitwise_xor_7_1: init_lookup_array!(log_size),
                verify_bitwise_xor_8_0: init_lookup_array!(log_size),
                verify_bitwise_xor_8_1: init_lookup_array!(log_size),
                verify_bitwise_xor_8_2: init_lookup_array!(log_size),
                verify_bitwise_xor_8_3: init_lookup_array!(log_size),
                verify_bitwise_xor_8_4: init_lookup_array!(log_size),
                verify_bitwise_xor_8_5: init_lookup_array!(log_size),
                verify_bitwise_xor_8_6: init_lookup_array!(log_size),
                verify_bitwise_xor_8_7: init_lookup_array!(log_size),
                verify_bitwise_xor_9_0: init_lookup_array!(log_size),
                verify_bitwise_xor_9_1: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_bitwise_xor_8: init_subcomponent_array!(log_size),
                verify_bitwise_xor_8_b: init_subcomponent_array!(log_size),
                verify_bitwise_xor_12: init_subcomponent_array!(log_size),
                verify_bitwise_xor_4: init_subcomponent_array!(log_size),
                verify_bitwise_xor_7: init_subcomponent_array!(log_size),
                verify_bitwise_xor_9: init_subcomponent_array!(log_size),
            },
        )
    };
    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    // collect lookup_data pointers
    let lookup_blake_g_0_vec = collect_lookup_ptrs!(lookup_data, blake_g_0);
    let lookup_verify_bitwise_xor_12_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_12_0);
    let lookup_verify_bitwise_xor_12_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_12_1);
    let lookup_verify_bitwise_xor_4_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_4_0);
    let lookup_verify_bitwise_xor_4_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_4_1);
    let lookup_verify_bitwise_xor_7_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_7_0);
    let lookup_verify_bitwise_xor_7_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_7_1);
    let lookup_verify_bitwise_xor_8_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_0);
    let lookup_verify_bitwise_xor_8_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_1);
    let lookup_verify_bitwise_xor_8_2_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_2);
    let lookup_verify_bitwise_xor_8_3_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_3);
    let lookup_verify_bitwise_xor_8_4_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_4);
    let lookup_verify_bitwise_xor_8_5_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_5);
    let lookup_verify_bitwise_xor_8_6_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_6);
    let lookup_verify_bitwise_xor_8_7_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_7);
    let lookup_verify_bitwise_xor_9_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_0);
    let lookup_verify_bitwise_xor_9_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_1);
    // collect sub_component_inputs pointers
    // CUDA kernel outputs 8 verify_bitwise_xor_8 relations which we split into vbx_8 (4) and
    // vbx_8_b (4)
    let sub_component_inputs_verify_bitwise_xor_8_vec: Vec<_> =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8);
    let sub_component_inputs_verify_bitwise_xor_8_b_vec: Vec<_> =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8_b);
    let mut sub_component_inputs_verify_bitwise_xor_8_combined_vec =
        sub_component_inputs_verify_bitwise_xor_8_vec;
    sub_component_inputs_verify_bitwise_xor_8_combined_vec
        .extend(sub_component_inputs_verify_bitwise_xor_8_b_vec);
    let sub_component_inputs_verify_bitwise_xor_12_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_12);
    let sub_component_inputs_verify_bitwise_xor_4_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_4);
    let sub_component_inputs_verify_bitwise_xor_7_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_7);
    let sub_component_inputs_verify_bitwise_xor_9_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_9);

    let blake_g_input_vec = collect_input_ptrs!(inputs);

    unsafe {
        bindings_airs::generate_blake_g_traces(
            traces_vec.as_ptr(),
            lookup_blake_g_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_12_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_12_1_vec.as_ptr(),
            lookup_verify_bitwise_xor_4_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_4_1_vec.as_ptr(),
            lookup_verify_bitwise_xor_7_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_7_1_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_1_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_2_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_3_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_4_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_5_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_6_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_7_vec.as_ptr(),
            lookup_verify_bitwise_xor_9_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_9_1_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_8_combined_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_12_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_4_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_7_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_9_vec.as_ptr(),
            blake_g_input_vec.as_ptr(),
            log_size as u32,
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

// #[derive(Uninitialized)]
struct CudaLookupData {
    blake_g_0: [BaseFieldVec; 20],
    verify_bitwise_xor_12_0: [BaseFieldVec; 3],
    verify_bitwise_xor_12_1: [BaseFieldVec; 3],
    verify_bitwise_xor_4_0: [BaseFieldVec; 3],
    verify_bitwise_xor_4_1: [BaseFieldVec; 3],
    verify_bitwise_xor_7_0: [BaseFieldVec; 3],
    verify_bitwise_xor_7_1: [BaseFieldVec; 3],
    verify_bitwise_xor_8_0: [BaseFieldVec; 3],
    verify_bitwise_xor_8_1: [BaseFieldVec; 3],
    verify_bitwise_xor_8_2: [BaseFieldVec; 3],
    verify_bitwise_xor_8_3: [BaseFieldVec; 3],
    verify_bitwise_xor_8_4: [BaseFieldVec; 3],
    verify_bitwise_xor_8_5: [BaseFieldVec; 3],
    verify_bitwise_xor_8_6: [BaseFieldVec; 3],
    verify_bitwise_xor_8_7: [BaseFieldVec; 3],
    verify_bitwise_xor_9_0: [BaseFieldVec; 3],
    verify_bitwise_xor_9_1: [BaseFieldVec; 3],
}

/// SIMD lookup data matching the CPU blake_g LookupData structure
#[allow(dead_code)]
struct SimdLookupData {
    blake_g_0: Vec<[PackedM31; 20]>,
    verify_bitwise_xor_12_0: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_12_1: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_4_0: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_4_1: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_7_0: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_7_1: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_0: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_1: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_2: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_3: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_b_0: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_b_1: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_b_2: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_8_b_3: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_9_0: Vec<[PackedM31; 3]>,
    verify_bitwise_xor_9_1: Vec<[PackedM31; 3]>,
}

#[allow(dead_code)]
const N_LANES: usize = 16;

/// Convert CUDA lookup data array to SIMD format
#[allow(dead_code)]
fn convert_cuda_lookup_to_simd<const N: usize>(
    cuda_data: &[BaseFieldVec; N],
    log_size: u32,
) -> Vec<[PackedM31; N]> {
    let trace_size = 1usize << log_size;
    let n_packed_rows = trace_size / N_LANES;

    // Convert each BaseFieldVec to CPU M31 array
    let cpu_data: [Vec<M31>; N] = std::array::from_fn(|i| cuda_data[i].to_cpu());

    // Convert to packed SIMD format
    (0..n_packed_rows)
        .map(|row| {
            let base = row * N_LANES;
            std::array::from_fn(|col| {
                PackedM31::from_array(std::array::from_fn(|i| cpu_data[col][base + i]))
            })
        })
        .collect()
}

#[allow(dead_code)]
impl SimdLookupData {
    /// Convert CUDA lookup data to SIMD format.
    /// Note: CUDA's verify_bitwise_xor_8_0-3 maps to SIMD's verify_bitwise_xor_8_0-3
    ///       CUDA's verify_bitwise_xor_8_4-7 maps to SIMD's verify_bitwise_xor_8_b_0-3
    fn from_cuda(cuda: &CudaLookupData, log_size: u32) -> Self {
        Self {
            blake_g_0: convert_cuda_lookup_to_simd(&cuda.blake_g_0, log_size),
            verify_bitwise_xor_12_0: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_12_0,
                log_size,
            ),
            verify_bitwise_xor_12_1: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_12_1,
                log_size,
            ),
            verify_bitwise_xor_4_0: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_4_0,
                log_size,
            ),
            verify_bitwise_xor_4_1: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_4_1,
                log_size,
            ),
            verify_bitwise_xor_7_0: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_7_0,
                log_size,
            ),
            verify_bitwise_xor_7_1: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_7_1,
                log_size,
            ),
            verify_bitwise_xor_8_0: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_0,
                log_size,
            ),
            verify_bitwise_xor_8_1: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_1,
                log_size,
            ),
            verify_bitwise_xor_8_2: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_2,
                log_size,
            ),
            verify_bitwise_xor_8_3: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_3,
                log_size,
            ),
            // CUDA _8_4-7 maps to SIMD _8_b_0-3
            verify_bitwise_xor_8_b_0: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_4,
                log_size,
            ),
            verify_bitwise_xor_8_b_1: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_5,
                log_size,
            ),
            verify_bitwise_xor_8_b_2: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_6,
                log_size,
            ),
            verify_bitwise_xor_8_b_3: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_8_7,
                log_size,
            ),
            verify_bitwise_xor_9_0: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_9_0,
                log_size,
            ),
            verify_bitwise_xor_9_1: convert_cuda_lookup_to_simd(
                &cuda.verify_bitwise_xor_9_1,
                log_size,
            ),
        }
    }
}

pub struct CudaInteractionClaimGenerator {
    n_rows: usize,
    log_size: u32,
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        // Use original CUDA kernel for now - SIMD version has compatibility issues with test
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(1 << trace_log_size) })
            .collect_vec();

        let lookup_blake_g_0_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_0);
        let lookup_verify_bitwise_xor_12_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_12_0);
        let lookup_verify_bitwise_xor_12_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_12_1);
        let lookup_verify_bitwise_xor_4_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_4_0);
        let lookup_verify_bitwise_xor_4_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_4_1);
        let lookup_verify_bitwise_xor_7_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_7_0);
        let lookup_verify_bitwise_xor_7_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_7_1);
        let lookup_verify_bitwise_xor_8_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_0);
        let lookup_verify_bitwise_xor_8_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_1);
        let lookup_verify_bitwise_xor_8_2_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_2);
        let lookup_verify_bitwise_xor_8_3_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_3);
        let lookup_verify_bitwise_xor_8_4_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_4);
        let lookup_verify_bitwise_xor_8_5_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_5);
        let lookup_verify_bitwise_xor_8_6_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_6);
        let lookup_verify_bitwise_xor_8_7_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_7);
        let lookup_verify_bitwise_xor_9_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_0);
        let lookup_verify_bitwise_xor_9_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_1);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_blake_g = create_modified_lookup_for_cuda(lookup_elements, BLAKE_G_RELATION_ID);
            let mod_vbx12 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_12_RELATION_ID);
            let mod_vbx4 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_4_RELATION_ID);
            let mod_vbx7 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_7_RELATION_ID);
            let mod_vbx8 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_8_RELATION_ID);
            let mod_vbx8b = create_modified_lookup_for_cuda(
                lookup_elements,
                VERIFY_BITWISE_XOR_8_B_RELATION_ID,
            );
            let mod_vbx9 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_9_RELATION_ID);

            bindings_airs::generate_blake_g_interaction_traces(
                &mod_blake_g as *const _ as *mut std::os::raw::c_void,
                &mod_vbx12 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx4 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx7 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx8 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx8b as *const _ as *mut std::os::raw::c_void,
                &mod_vbx9 as *const _ as *mut std::os::raw::c_void,
                lookup_blake_g_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_12_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_12_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_4_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_4_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_7_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_7_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_2_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_3_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_4_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_5_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_6_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_7_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_9_1_vec.as_ptr(),
                self.n_rows as u32,
                trace_log_size as u32,
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }
        println!(
            "[DEBUG blake_g CUDA] interaction trace: n_rows={}, trace_log_size={}, trace_size={}",
            self.n_rows,
            trace_log_size,
            1 << trace_log_size
        );

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

        println!("cuda claimed sum: {:?}", claimed_sum);
        InteractionClaim { claimed_sum }
    }
}
