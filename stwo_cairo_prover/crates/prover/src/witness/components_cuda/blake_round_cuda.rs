#![allow(unused_parens)]
use std::sync::Arc;

use cairo_air::components::blake_round::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo_cairo_adapter::memory::Memory;

use super::{
    blake_g_cuda, blake_round_sigma_cuda, memory_address_to_id_cuda, memory_id_to_big_cuda,
    range_check_7_2_5_cuda,
};
use crate::witness::prelude::*;

pub type PackedInputType = (PackedM31, PackedM31, ([PackedUInt32; 16], PackedM31));
pub type CudaPackedInputType = (BaseFieldVec, BaseFieldVec, ([Uint32Vec; 16], BaseFieldVec));

pub const N_TRACE_COLUMNS: usize = 212;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 30;

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}
macro_rules! init_subcomponent_basefield_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
        })
    };
}

macro_rules! init_subcomponent_uint32_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| Uint32Vec::new_uninitialized(1 << $log_size))
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
    ($input:expr) => {{
        let mut v = Vec::new();
        v.push($input.0.device_ptr);
        v.push($input.1.device_ptr);
        for x in $input.2 .0.iter() {
            v.push(x.device_ptr);
        }
        v.push($input.2 .1.device_ptr);
        v
    }};
}

pub struct CudaClaimGenerator {
    pub packed_inputs: CudaPackedInputType,
    state: BlakeRound,
}
impl CudaClaimGenerator {
    pub fn new(memory: Arc<Memory>) -> Self {
        let state = BlakeRound::new(memory);
        Self {
            packed_inputs: (
                BaseFieldVec::new_zeroes(1),
                BaseFieldVec::new_zeroes(1),
                (
                    std::array::from_fn(|_| Uint32Vec::new_zeroes(1)),
                    BaseFieldVec::new_zeroes(1),
                ),
            ),
            state,
        }
    }

    /// Returns true if no inputs have been added (only initial zeroes).
    pub fn is_empty(&self) -> bool {
        // Initial state has size 1 with all zeroes
        self.packed_inputs.0.size <= 1
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        blake_g_state: &mut blake_g_cuda::CudaClaimGenerator,
        blake_round_sigma_state: &blake_round_sigma_cuda::CudaClaimGenerator,
        memory_address_to_id_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        // Track the real number of rows before padding
        let n_rows = self.packed_inputs.0.size;
        let padded_size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
        let log_size = padded_size.ilog2();

        println!(
            "[DEBUG] blake_round_cuda.write_trace: n_rows={}, padded_size={}, log_size={}",
            n_rows, padded_size, log_size
        );

        // Pad inputs to next power of 2 (required for CUDA parallel processing)
        // Match SIMD behavior: pad by cycling through first 16 rows (one packed vector)
        // GPU in-place padding — no download/upload roundtrip.
        if padded_size > n_rows {
            const N_LANES: usize = 16;
            let cycle_len = n_rows.min(N_LANES);

            self.packed_inputs.0.pad_with_cycle(n_rows, padded_size, cycle_len);
            self.packed_inputs.1.pad_with_cycle(n_rows, padded_size, cycle_len);
            for j in 0..16 {
                self.packed_inputs.2 .0[j].pad_with_cycle(n_rows, padded_size, cycle_len);
            }
            self.packed_inputs.2 .1.pad_with_cycle(n_rows, padded_size, cycle_len);
        }

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            n_rows,
            self.packed_inputs,
            blake_g_state,
            blake_round_sigma_state,
            memory_address_to_id_state,
            memory_id_to_big_state,
            range_check_7_2_5_state,
        );

        // Pass sub-component inputs. All inputs have padded_size elements since blake_round
        // pads its inputs before calling write_trace_cuda.
        blake_round_sigma_state.add_cuda_inputs(&sub_component_inputs.blake_round_sigma);
        range_check_7_2_5_state.add_cuda_inputs(&sub_component_inputs.range_check_7_2_5);
        memory_address_to_id_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        blake_g_state.add_cuda_inputs(&sub_component_inputs.blake_g);

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

    pub fn add_packed_inputs(&mut self, inputs: &[PackedInputType]) {
        // println!("Adding packed inputs to blake round: {:?}", inputs);

        // println!("cpu packed0: {:?}", inputs.0 );
        // println!("cpu packed0: {:?}", packed1.to_vec());
        // for i in 0..16 {
        //     println!("cpu packed_arr16{i}: {:?}", packed_arr16[i].to_vec());
        // }
        // println!("cpu packed2_1: {:?}", packed2_1.to_vec());

        // packed0
        let packed0 = BaseFieldVec::from_vec(inputs.iter().flat_map(|x| x.0.to_array()).collect());

        // packed1
        let packed1 = BaseFieldVec::from_vec(inputs.iter().flat_map(|x| x.1.to_array()).collect());

        // packed_arr16
        let packed_arr16: [Uint32Vec; 16] = (0..16)
            .map(|i| {
                let elements: Vec<_> = inputs
                    .iter()
                    .flat_map(|input| input.2 .0[i].as_array())
                    .collect();
                Uint32Vec::from_vec(unsafe { std::mem::transmute(elements) })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Should have exactly 16 elements");

        // packed2_1
        let packed2_1 =
            BaseFieldVec::from_vec(inputs.iter().flat_map(|x| x.2 .1.to_array()).collect());

        // println!("cuda packed0: {:?}", packed0.to_vec());
        // println!("cuda packed1: {:?}", packed1.to_vec());
        // for i in 0..16 {
        //     println!("cuda packed_arr16{i}: {:?}", packed_arr16[i].to_vec());
        // }
        // println!("cuda packed2_1: {:?}", packed2_1.to_vec());

        self.packed_inputs = (packed0, packed1, (packed_arr16, packed2_1));
    }

    pub fn add_cuda_inputs(&mut self, inputs: &[CudaPackedInputType]) {
        // On first call, replace the initial zeroed data instead of extending
        let mut first = self.is_empty();
        inputs.iter().for_each(|x| {
            if first {
                // Replace initial dummy data
                self.packed_inputs.0 = x.0.clone();
                self.packed_inputs.1 = x.1.clone();
                for i in 0..16 {
                    self.packed_inputs.2 .0[i] = x.2 .0[i].clone();
                }
                self.packed_inputs.2 .1 = x.2 .1.clone();
                first = false;
            } else {
                // Extend existing data
                self.packed_inputs.0.extend(&x.0);
                self.packed_inputs.1.extend(&x.1);
                for i in 0..16 {
                    self.packed_inputs.2 .0[i].extend(&x.2 .0[i]);
                }
                self.packed_inputs.2 .1.extend(&x.2 .1);
            }
        });
    }

    pub fn deduce_output(
        &self,
        input: (PackedM31, PackedM31, ([PackedUInt32; 16], PackedM31)),
    ) -> (PackedM31, PackedM31, ([PackedUInt32; 16], PackedM31)) {
        self.state.deduce_output(input.0, input.1, input.2)
    }
}

struct CudaSubComponentInputs {
    blake_round_sigma: [blake_round_sigma_cuda::CudaPackedInputType; 1],
    range_check_7_2_5: [range_check_7_2_5_cuda::CudaPackedInputType; 16],
    memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 16],
    memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 16],
    blake_g: [blake_g_cuda::CudaPackedInputType; 8],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    n_rows: usize,
    inputs: CudaPackedInputType,
    blake_g_state: &blake_g_cuda::CudaClaimGenerator,
    blake_round_sigma_state: &blake_round_sigma_cuda::CudaClaimGenerator,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    range_check_7_2_5_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    // inputs.0.size is the padded size (power of 2)
    // n_rows is the actual number of rows with real data
    let log_size = inputs.0.size.ilog2();
    // println!("write_trace_simd for Packed inputs to blake round: {:?}, log_size:{}, cols:{}",
    // inputs, log_size, N_TRACE_COLUMNS);

    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                blake_g_0: init_lookup_array!(log_size),
                blake_g_1: init_lookup_array!(log_size),
                blake_g_2: init_lookup_array!(log_size),
                blake_g_3: init_lookup_array!(log_size),
                blake_g_4: init_lookup_array!(log_size),
                blake_g_5: init_lookup_array!(log_size),
                blake_g_6: init_lookup_array!(log_size),
                blake_g_7: init_lookup_array!(log_size),
                blake_round_0: init_lookup_array!(log_size),
                blake_round_1: init_lookup_array!(log_size),
                blake_round_sigma_0: init_lookup_array!(log_size),
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                memory_address_to_id_3: init_lookup_array!(log_size),
                memory_address_to_id_4: init_lookup_array!(log_size),
                memory_address_to_id_5: init_lookup_array!(log_size),
                memory_address_to_id_6: init_lookup_array!(log_size),
                memory_address_to_id_7: init_lookup_array!(log_size),
                memory_address_to_id_8: init_lookup_array!(log_size),
                memory_address_to_id_9: init_lookup_array!(log_size),
                memory_address_to_id_10: init_lookup_array!(log_size),
                memory_address_to_id_11: init_lookup_array!(log_size),
                memory_address_to_id_12: init_lookup_array!(log_size),
                memory_address_to_id_13: init_lookup_array!(log_size),
                memory_address_to_id_14: init_lookup_array!(log_size),
                memory_address_to_id_15: init_lookup_array!(log_size),
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                memory_id_to_big_3: init_lookup_array!(log_size),
                memory_id_to_big_4: init_lookup_array!(log_size),
                memory_id_to_big_5: init_lookup_array!(log_size),
                memory_id_to_big_6: init_lookup_array!(log_size),
                memory_id_to_big_7: init_lookup_array!(log_size),
                memory_id_to_big_8: init_lookup_array!(log_size),
                memory_id_to_big_9: init_lookup_array!(log_size),
                memory_id_to_big_10: init_lookup_array!(log_size),
                memory_id_to_big_11: init_lookup_array!(log_size),
                memory_id_to_big_12: init_lookup_array!(log_size),
                memory_id_to_big_13: init_lookup_array!(log_size),
                memory_id_to_big_14: init_lookup_array!(log_size),
                memory_id_to_big_15: init_lookup_array!(log_size),
                range_check_7_2_5_0: init_lookup_array!(log_size),
                range_check_7_2_5_1: init_lookup_array!(log_size),
                range_check_7_2_5_2: init_lookup_array!(log_size),
                range_check_7_2_5_3: init_lookup_array!(log_size),
                range_check_7_2_5_4: init_lookup_array!(log_size),
                range_check_7_2_5_5: init_lookup_array!(log_size),
                range_check_7_2_5_6: init_lookup_array!(log_size),
                range_check_7_2_5_7: init_lookup_array!(log_size),
                range_check_7_2_5_8: init_lookup_array!(log_size),
                range_check_7_2_5_9: init_lookup_array!(log_size),
                range_check_7_2_5_10: init_lookup_array!(log_size),
                range_check_7_2_5_11: init_lookup_array!(log_size),
                range_check_7_2_5_12: init_lookup_array!(log_size),
                range_check_7_2_5_13: init_lookup_array!(log_size),
                range_check_7_2_5_14: init_lookup_array!(log_size),
                range_check_7_2_5_15: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                blake_round_sigma: init_subcomponent_basefield_array!(log_size),
                range_check_7_2_5: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                blake_g: init_subcomponent_uint32_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    // collect lookup_data pointers
    let lookup_blake_g_0_vec = collect_lookup_ptrs!(lookup_data, blake_g_0);
    let lookup_blake_g_1_vec = collect_lookup_ptrs!(lookup_data, blake_g_1);
    let lookup_blake_g_2_vec = collect_lookup_ptrs!(lookup_data, blake_g_2);
    let lookup_blake_g_3_vec = collect_lookup_ptrs!(lookup_data, blake_g_3);
    let lookup_blake_g_4_vec = collect_lookup_ptrs!(lookup_data, blake_g_4);
    let lookup_blake_g_5_vec = collect_lookup_ptrs!(lookup_data, blake_g_5);
    let lookup_blake_g_6_vec = collect_lookup_ptrs!(lookup_data, blake_g_6);
    let lookup_blake_g_7_vec = collect_lookup_ptrs!(lookup_data, blake_g_7);
    let lookup_blake_round_0_vec = collect_lookup_ptrs!(lookup_data, blake_round_0);
    let lookup_blake_round_1_vec = collect_lookup_ptrs!(lookup_data, blake_round_1);
    let lookup_blake_round_sigma_0_vec = collect_lookup_ptrs!(lookup_data, blake_round_sigma_0);
    let lookup_memory_address_to_id_0_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_address_to_id_3_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_3);
    let lookup_memory_address_to_id_4_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_4);
    let lookup_memory_address_to_id_5_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_5);
    let lookup_memory_address_to_id_6_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_6);
    let lookup_memory_address_to_id_7_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_7);
    let lookup_memory_address_to_id_8_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_8);
    let lookup_memory_address_to_id_9_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_9);
    let lookup_memory_address_to_id_10_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_10);
    let lookup_memory_address_to_id_11_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_11);
    let lookup_memory_address_to_id_12_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_12);
    let lookup_memory_address_to_id_13_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_13);
    let lookup_memory_address_to_id_14_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_14);
    let lookup_memory_address_to_id_15_vec =
        collect_lookup_ptrs!(lookup_data, memory_address_to_id_15);
    let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_memory_id_to_big_3_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_3);
    let lookup_memory_id_to_big_4_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_4);
    let lookup_memory_id_to_big_5_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_5);
    let lookup_memory_id_to_big_6_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_6);
    let lookup_memory_id_to_big_7_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_7);
    let lookup_memory_id_to_big_8_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_8);
    let lookup_memory_id_to_big_9_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_9);
    let lookup_memory_id_to_big_10_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_10);
    let lookup_memory_id_to_big_11_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_11);
    let lookup_memory_id_to_big_12_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_12);
    let lookup_memory_id_to_big_13_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_13);
    let lookup_memory_id_to_big_14_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_14);
    let lookup_memory_id_to_big_15_vec = collect_lookup_ptrs!(lookup_data, memory_id_to_big_15);
    let lookup_range_check_7_2_5_0_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_0);
    let lookup_range_check_7_2_5_1_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_1);
    let lookup_range_check_7_2_5_2_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_2);
    let lookup_range_check_7_2_5_3_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_3);
    let lookup_range_check_7_2_5_4_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_4);
    let lookup_range_check_7_2_5_5_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_5);
    let lookup_range_check_7_2_5_6_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_6);
    let lookup_range_check_7_2_5_7_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_7);
    let lookup_range_check_7_2_5_8_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_8);
    let lookup_range_check_7_2_5_9_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_9);
    let lookup_range_check_7_2_5_10_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_10);
    let lookup_range_check_7_2_5_11_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_11);
    let lookup_range_check_7_2_5_12_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_12);
    let lookup_range_check_7_2_5_13_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_13);
    let lookup_range_check_7_2_5_14_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_14);
    let lookup_range_check_7_2_5_15_vec = collect_lookup_ptrs!(lookup_data, range_check_7_2_5_15);

    // collect sub_component_inputs pointers
    let sub_component_inputs_blake_round_sigma_vec =
        collect_sub_input_ptrs!(sub_component_inputs, blake_round_sigma);
    let sub_component_inputs_range_check_7_2_5_vec =
        collect_sub_input_ptrs!(sub_component_inputs, range_check_7_2_5);
    let sub_component_inputs_memory_address_to_id_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_blake_g_vec = collect_sub_input_ptrs!(sub_component_inputs, blake_g);

    let blake_round_input_vec = collect_input_ptrs!(inputs);

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_blake_round_traces(
            traces_vec.as_ptr(),
            lookup_blake_g_0_vec.as_ptr(),
            lookup_blake_g_1_vec.as_ptr(),
            lookup_blake_g_2_vec.as_ptr(),
            lookup_blake_g_3_vec.as_ptr(),
            lookup_blake_g_4_vec.as_ptr(),
            lookup_blake_g_5_vec.as_ptr(),
            lookup_blake_g_6_vec.as_ptr(),
            lookup_blake_g_7_vec.as_ptr(),
            lookup_blake_round_0_vec.as_ptr(),
            lookup_blake_round_1_vec.as_ptr(),
            lookup_blake_round_sigma_0_vec.as_ptr(),
            lookup_memory_address_to_id_0_vec.as_ptr(),
            lookup_memory_address_to_id_1_vec.as_ptr(),
            lookup_memory_address_to_id_2_vec.as_ptr(),
            lookup_memory_address_to_id_3_vec.as_ptr(),
            lookup_memory_address_to_id_4_vec.as_ptr(),
            lookup_memory_address_to_id_5_vec.as_ptr(),
            lookup_memory_address_to_id_6_vec.as_ptr(),
            lookup_memory_address_to_id_7_vec.as_ptr(),
            lookup_memory_address_to_id_8_vec.as_ptr(),
            lookup_memory_address_to_id_9_vec.as_ptr(),
            lookup_memory_address_to_id_10_vec.as_ptr(),
            lookup_memory_address_to_id_11_vec.as_ptr(),
            lookup_memory_address_to_id_12_vec.as_ptr(),
            lookup_memory_address_to_id_13_vec.as_ptr(),
            lookup_memory_address_to_id_14_vec.as_ptr(),
            lookup_memory_address_to_id_15_vec.as_ptr(),
            lookup_memory_id_to_big_0_vec.as_ptr(),
            lookup_memory_id_to_big_1_vec.as_ptr(),
            lookup_memory_id_to_big_2_vec.as_ptr(),
            lookup_memory_id_to_big_3_vec.as_ptr(),
            lookup_memory_id_to_big_4_vec.as_ptr(),
            lookup_memory_id_to_big_5_vec.as_ptr(),
            lookup_memory_id_to_big_6_vec.as_ptr(),
            lookup_memory_id_to_big_7_vec.as_ptr(),
            lookup_memory_id_to_big_8_vec.as_ptr(),
            lookup_memory_id_to_big_9_vec.as_ptr(),
            lookup_memory_id_to_big_10_vec.as_ptr(),
            lookup_memory_id_to_big_11_vec.as_ptr(),
            lookup_memory_id_to_big_12_vec.as_ptr(),
            lookup_memory_id_to_big_13_vec.as_ptr(),
            lookup_memory_id_to_big_14_vec.as_ptr(),
            lookup_memory_id_to_big_15_vec.as_ptr(),
            lookup_range_check_7_2_5_0_vec.as_ptr(),
            lookup_range_check_7_2_5_1_vec.as_ptr(),
            lookup_range_check_7_2_5_2_vec.as_ptr(),
            lookup_range_check_7_2_5_3_vec.as_ptr(),
            lookup_range_check_7_2_5_4_vec.as_ptr(),
            lookup_range_check_7_2_5_5_vec.as_ptr(),
            lookup_range_check_7_2_5_6_vec.as_ptr(),
            lookup_range_check_7_2_5_7_vec.as_ptr(),
            lookup_range_check_7_2_5_8_vec.as_ptr(),
            lookup_range_check_7_2_5_9_vec.as_ptr(),
            lookup_range_check_7_2_5_10_vec.as_ptr(),
            lookup_range_check_7_2_5_11_vec.as_ptr(),
            lookup_range_check_7_2_5_12_vec.as_ptr(),
            lookup_range_check_7_2_5_13_vec.as_ptr(),
            lookup_range_check_7_2_5_14_vec.as_ptr(),
            lookup_range_check_7_2_5_15_vec.as_ptr(),
            sub_component_inputs_blake_round_sigma_vec.as_ptr(),
            sub_component_inputs_range_check_7_2_5_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            sub_component_inputs_blake_g_vec.as_ptr(),
            blake_round_input_vec.as_ptr(),
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,
            n_rows as u32,
            log_size as u32,
        );
    }
    // for i in 0..N_TRACE_COLUMNS {
    //     println!("CUDA trace row {}: {:?}", i, trace.data[i].to_vec());
    // }

    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    blake_g_0: [BaseFieldVec; 20],
    blake_g_1: [BaseFieldVec; 20],
    blake_g_2: [BaseFieldVec; 20],
    blake_g_3: [BaseFieldVec; 20],
    blake_g_4: [BaseFieldVec; 20],
    blake_g_5: [BaseFieldVec; 20],
    blake_g_6: [BaseFieldVec; 20],
    blake_g_7: [BaseFieldVec; 20],
    blake_round_0: [BaseFieldVec; 35],
    blake_round_1: [BaseFieldVec; 35],
    blake_round_sigma_0: [BaseFieldVec; 17],
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    memory_address_to_id_3: [BaseFieldVec; 2],
    memory_address_to_id_4: [BaseFieldVec; 2],
    memory_address_to_id_5: [BaseFieldVec; 2],
    memory_address_to_id_6: [BaseFieldVec; 2],
    memory_address_to_id_7: [BaseFieldVec; 2],
    memory_address_to_id_8: [BaseFieldVec; 2],
    memory_address_to_id_9: [BaseFieldVec; 2],
    memory_address_to_id_10: [BaseFieldVec; 2],
    memory_address_to_id_11: [BaseFieldVec; 2],
    memory_address_to_id_12: [BaseFieldVec; 2],
    memory_address_to_id_13: [BaseFieldVec; 2],
    memory_address_to_id_14: [BaseFieldVec; 2],
    memory_address_to_id_15: [BaseFieldVec; 2],
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    memory_id_to_big_3: [BaseFieldVec; 29],
    memory_id_to_big_4: [BaseFieldVec; 29],
    memory_id_to_big_5: [BaseFieldVec; 29],
    memory_id_to_big_6: [BaseFieldVec; 29],
    memory_id_to_big_7: [BaseFieldVec; 29],
    memory_id_to_big_8: [BaseFieldVec; 29],
    memory_id_to_big_9: [BaseFieldVec; 29],
    memory_id_to_big_10: [BaseFieldVec; 29],
    memory_id_to_big_11: [BaseFieldVec; 29],
    memory_id_to_big_12: [BaseFieldVec; 29],
    memory_id_to_big_13: [BaseFieldVec; 29],
    memory_id_to_big_14: [BaseFieldVec; 29],
    memory_id_to_big_15: [BaseFieldVec; 29],
    range_check_7_2_5_0: [BaseFieldVec; 3],
    range_check_7_2_5_1: [BaseFieldVec; 3],
    range_check_7_2_5_2: [BaseFieldVec; 3],
    range_check_7_2_5_3: [BaseFieldVec; 3],
    range_check_7_2_5_4: [BaseFieldVec; 3],
    range_check_7_2_5_5: [BaseFieldVec; 3],
    range_check_7_2_5_6: [BaseFieldVec; 3],
    range_check_7_2_5_7: [BaseFieldVec; 3],
    range_check_7_2_5_8: [BaseFieldVec; 3],
    range_check_7_2_5_9: [BaseFieldVec; 3],
    range_check_7_2_5_10: [BaseFieldVec; 3],
    range_check_7_2_5_11: [BaseFieldVec; 3],
    range_check_7_2_5_12: [BaseFieldVec; 3],
    range_check_7_2_5_13: [BaseFieldVec; 3],
    range_check_7_2_5_14: [BaseFieldVec; 3],
    range_check_7_2_5_15: [BaseFieldVec; 3],
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
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Allocate interaction trace columns on GPU (4 extensions x 30 columns)
        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| unsafe { Col::<CudaBackend, BaseField>::uninitialized(1 << trace_log_size) })
            .collect_vec();

        // Collect lookup data pointers (already on GPU - no download needed)
        let lookup_blake_g_0_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_0);
        let lookup_blake_g_1_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_1);
        let lookup_blake_g_2_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_2);
        let lookup_blake_g_3_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_3);
        let lookup_blake_g_4_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_4);
        let lookup_blake_g_5_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_5);
        let lookup_blake_g_6_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_6);
        let lookup_blake_g_7_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_7);

        let lookup_blake_round_0_vec = collect_lookup_ptrs!(self.lookup_data, blake_round_0);
        let lookup_blake_round_1_vec = collect_lookup_ptrs!(self.lookup_data, blake_round_1);

        let lookup_blake_round_sigma_0_vec =
            collect_lookup_ptrs!(self.lookup_data, blake_round_sigma_0);

        let lookup_memory_address_to_id_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_address_to_id_3_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_3);
        let lookup_memory_address_to_id_4_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_4);
        let lookup_memory_address_to_id_5_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_5);
        let lookup_memory_address_to_id_6_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_6);
        let lookup_memory_address_to_id_7_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_7);
        let lookup_memory_address_to_id_8_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_8);
        let lookup_memory_address_to_id_9_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_9);
        let lookup_memory_address_to_id_10_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_10);
        let lookup_memory_address_to_id_11_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_11);
        let lookup_memory_address_to_id_12_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_12);
        let lookup_memory_address_to_id_13_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_13);
        let lookup_memory_address_to_id_14_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_14);
        let lookup_memory_address_to_id_15_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_15);

        let lookup_memory_id_to_big_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_memory_id_to_big_3_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_3);
        let lookup_memory_id_to_big_4_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_4);
        let lookup_memory_id_to_big_5_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_5);
        let lookup_memory_id_to_big_6_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_6);
        let lookup_memory_id_to_big_7_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_7);
        let lookup_memory_id_to_big_8_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_8);
        let lookup_memory_id_to_big_9_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_9);
        let lookup_memory_id_to_big_10_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_10);
        let lookup_memory_id_to_big_11_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_11);
        let lookup_memory_id_to_big_12_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_12);
        let lookup_memory_id_to_big_13_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_13);
        let lookup_memory_id_to_big_14_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_14);
        let lookup_memory_id_to_big_15_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_15);

        let lookup_range_check_7_2_5_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_0);
        let lookup_range_check_7_2_5_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_1);
        let lookup_range_check_7_2_5_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_2);
        let lookup_range_check_7_2_5_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_3);
        let lookup_range_check_7_2_5_4_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_4);
        let lookup_range_check_7_2_5_5_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_5);
        let lookup_range_check_7_2_5_6_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_6);
        let lookup_range_check_7_2_5_7_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_7);
        let lookup_range_check_7_2_5_8_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_8);
        let lookup_range_check_7_2_5_9_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_9);
        let lookup_range_check_7_2_5_10_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_10);
        let lookup_range_check_7_2_5_11_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_11);
        let lookup_range_check_7_2_5_12_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_12);
        let lookup_range_check_7_2_5_13_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_13);
        let lookup_range_check_7_2_5_14_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_14);
        let lookup_range_check_7_2_5_15_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_15);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_blake_g = create_modified_lookup_for_cuda(lookup_elements, BLAKE_G_RELATION_ID);
            let mod_blake_round =
                create_modified_lookup_for_cuda(lookup_elements, BLAKE_ROUND_RELATION_ID);
            let mod_blake_round_sigma =
                create_modified_lookup_for_cuda(lookup_elements, BLAKE_ROUND_SIGMA_RELATION_ID);
            let mod_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
            let mod_rc725 = create_modified_lookup_for_cuda(lookup_elements, RC_7_2_5_RELATION_ID);

            bindings_airs::generate_blake_round_interaction_traces(
                &mod_blake_g as *const _ as *mut std::os::raw::c_void,
                &mod_blake_round as *const _ as *mut std::os::raw::c_void,
                &mod_blake_round_sigma as *const _ as *mut std::os::raw::c_void,
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_rc725 as *const _ as *mut std::os::raw::c_void,
                lookup_blake_g_0_vec.as_ptr(),
                lookup_blake_g_1_vec.as_ptr(),
                lookup_blake_g_2_vec.as_ptr(),
                lookup_blake_g_3_vec.as_ptr(),
                lookup_blake_g_4_vec.as_ptr(),
                lookup_blake_g_5_vec.as_ptr(),
                lookup_blake_g_6_vec.as_ptr(),
                lookup_blake_g_7_vec.as_ptr(),
                lookup_blake_round_0_vec.as_ptr(),
                lookup_blake_round_1_vec.as_ptr(),
                lookup_blake_round_sigma_0_vec.as_ptr(),
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_address_to_id_3_vec.as_ptr(),
                lookup_memory_address_to_id_4_vec.as_ptr(),
                lookup_memory_address_to_id_5_vec.as_ptr(),
                lookup_memory_address_to_id_6_vec.as_ptr(),
                lookup_memory_address_to_id_7_vec.as_ptr(),
                lookup_memory_address_to_id_8_vec.as_ptr(),
                lookup_memory_address_to_id_9_vec.as_ptr(),
                lookup_memory_address_to_id_10_vec.as_ptr(),
                lookup_memory_address_to_id_11_vec.as_ptr(),
                lookup_memory_address_to_id_12_vec.as_ptr(),
                lookup_memory_address_to_id_13_vec.as_ptr(),
                lookup_memory_address_to_id_14_vec.as_ptr(),
                lookup_memory_address_to_id_15_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_memory_id_to_big_3_vec.as_ptr(),
                lookup_memory_id_to_big_4_vec.as_ptr(),
                lookup_memory_id_to_big_5_vec.as_ptr(),
                lookup_memory_id_to_big_6_vec.as_ptr(),
                lookup_memory_id_to_big_7_vec.as_ptr(),
                lookup_memory_id_to_big_8_vec.as_ptr(),
                lookup_memory_id_to_big_9_vec.as_ptr(),
                lookup_memory_id_to_big_10_vec.as_ptr(),
                lookup_memory_id_to_big_11_vec.as_ptr(),
                lookup_memory_id_to_big_12_vec.as_ptr(),
                lookup_memory_id_to_big_13_vec.as_ptr(),
                lookup_memory_id_to_big_14_vec.as_ptr(),
                lookup_memory_id_to_big_15_vec.as_ptr(),
                lookup_range_check_7_2_5_0_vec.as_ptr(),
                lookup_range_check_7_2_5_1_vec.as_ptr(),
                lookup_range_check_7_2_5_2_vec.as_ptr(),
                lookup_range_check_7_2_5_3_vec.as_ptr(),
                lookup_range_check_7_2_5_4_vec.as_ptr(),
                lookup_range_check_7_2_5_5_vec.as_ptr(),
                lookup_range_check_7_2_5_6_vec.as_ptr(),
                lookup_range_check_7_2_5_7_vec.as_ptr(),
                lookup_range_check_7_2_5_8_vec.as_ptr(),
                lookup_range_check_7_2_5_9_vec.as_ptr(),
                lookup_range_check_7_2_5_10_vec.as_ptr(),
                lookup_range_check_7_2_5_11_vec.as_ptr(),
                lookup_range_check_7_2_5_12_vec.as_ptr(),
                lookup_range_check_7_2_5_13_vec.as_ptr(),
                lookup_range_check_7_2_5_14_vec.as_ptr(),
                lookup_range_check_7_2_5_15_vec.as_ptr(),
                self.n_rows as u32,
                trace_log_size as u32,
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
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
