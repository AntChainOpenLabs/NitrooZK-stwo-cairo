#![allow(unused_parens)]
use cairo_air::components::blake_round::{Claim, InteractionClaim};
use stwo_cairo_adapter::memory::Memory;
use stwo::stwo_cuda::base_field_vec::Uint32Vec;

use stwo::prover::backend::cuda::CudaBackend;
use crate::witness::components_cuda::blake_round_sigma_cuda;
use crate::witness::components_cuda::memory_address_to_id_cuda;
use crate::witness::components_cuda::blake_g_cuda;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::witness::components_cuda::range_check_7_2_5_cuda;
use crate::witness::components_cuda::memory_id_to_big_cuda;
use itertools::Itertools;
use stwo::stwo_cuda::bindings_airs;
use crate::witness::prelude::*;
use stwo::prover::backend::Col;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

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
        std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size)))
    };
}

macro_rules! init_subcomponent_uint32_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| std::array::from_fn(|_| Uint32Vec::new_uninitialized(1 << $log_size)))
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

macro_rules! collect_input_ptrs {
    ($input:expr) => {{
        let mut v = Vec::new();
        v.push($input.0.device_ptr);
        v.push($input.1.device_ptr);
        for x in $input.2.0.iter() {
            v.push(x.device_ptr);
        }
        v.push($input.2.1.device_ptr);
        v
    }};
}

pub struct CudaClaimGenerator {
    pub packed_inputs: CudaPackedInputType,
    state: BlakeRound,
}
impl CudaClaimGenerator {
    pub fn new(memory: Memory) -> Self {
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
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        blake_g_state: &mut blake_g_cuda::CudaClaimGenerator,
        blake_round_sigma_state: &blake_round_sigma_cuda::CudaClaimGenerator,
        memory_address_to_id_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {

        let log_size = self.packed_inputs.0.size.ilog2();


        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            self.packed_inputs,
            blake_g_state,
            blake_round_sigma_state,
            memory_address_to_id_state,
            memory_id_to_big_state,
            range_check_7_2_5_state,
        );
        blake_round_sigma_state.add_cuda_inputs(&sub_component_inputs.blake_round_sigma);
        range_check_7_2_5_state.add_cuda_inputs(&sub_component_inputs.range_check_7_2_5);
        memory_address_to_id_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);
        blake_g_state.add_cuda_inputs(&sub_component_inputs.blake_g);

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                lookup_data,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, inputs: &[PackedInputType]) {

        // for i in 0..16 {
        // }

        // packed0
        let packed0 = BaseFieldVec::from_vec(
            inputs.iter().flat_map(|x| x.0.to_array()).collect()
        );

        // packed1
        let packed1 = BaseFieldVec::from_vec(
            inputs.iter().flat_map(|x| x.1.to_array()).collect()
        );

        // packed_arr16
        let packed_arr16: [Uint32Vec; 16] = (0..16)
            .map(|i| {
                let elements: Vec<_> = inputs
                    .iter()
                    .flat_map(|input| input.2.0[i].as_array())
                    .collect();
                Uint32Vec::from_vec(unsafe { std::mem::transmute(elements) })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Should have exactly 16 elements");

        // packed2_1
        let packed2_1 = BaseFieldVec::from_vec(
            inputs.iter().flat_map(|x| x.2.1.to_array()).collect()
        );


        // for i in 0..16 {
        // }

        self.packed_inputs = (packed0, packed1, (packed_arr16, packed2_1));
    }

    pub fn add_cuda_inputs(&mut self, inputs: &[CudaPackedInputType]) {
        inputs.iter().for_each(|x| {
            self.packed_inputs.0.extend(&x.0);
            self.packed_inputs.1.extend(&x.1);
            for i in 0..16 {
                self.packed_inputs.2 .0[i].extend(&x.2 .0[i]);
            }
            self.packed_inputs.2 .1.extend(&x.2 .1);
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
    let log_size = inputs.0.size.ilog2();

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
    let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_address_to_id_3_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_3);
    let lookup_memory_address_to_id_4_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_4);
    let lookup_memory_address_to_id_5_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_5);
    let lookup_memory_address_to_id_6_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_6);
    let lookup_memory_address_to_id_7_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_7);
    let lookup_memory_address_to_id_8_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_8);
    let lookup_memory_address_to_id_9_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_9);
    let lookup_memory_address_to_id_10_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_10);
    let lookup_memory_address_to_id_11_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_11);
    let lookup_memory_address_to_id_12_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_12);
    let lookup_memory_address_to_id_13_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_13);
    let lookup_memory_address_to_id_14_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_14);
    let lookup_memory_address_to_id_15_vec = collect_lookup_ptrs!(lookup_data, memory_address_to_id_15);
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
    let sub_component_inputs_blake_round_sigma_vec = collect_sub_input_ptrs!(sub_component_inputs, blake_round_sigma);
    let sub_component_inputs_range_check_7_2_5_vec = collect_sub_input_ptrs!(sub_component_inputs, range_check_7_2_5);
    let sub_component_inputs_memory_address_to_id_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_component_inputs_blake_g_vec = collect_sub_input_ptrs!(sub_component_inputs, blake_g);

    let blake_round_input_vec = collect_input_ptrs!(inputs);

    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state.transposed_big_values
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

            log_size as u32,
        );
    }
    // for i in 0..N_TRACE_COLUMNS {
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
    log_size: u32,
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        blake_g: &relations::BlakeG,
        blake_round: &relations::BlakeRound,
        blake_round_sigma: &relations::BlakeRoundSigma,
        memory_address_to_id: &relations::MemoryAddressToId,
        memory_id_to_big: &relations::MemoryIdToBig,
        range_check_7_2_5: &relations::RangeCheck_7_2_5,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

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

        let lookup_blake_round_sigma_0_vec = collect_lookup_ptrs!(self.lookup_data, blake_round_sigma_0);

        let lookup_memory_address_to_id_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_address_to_id_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_3);
        let lookup_memory_address_to_id_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_4);
        let lookup_memory_address_to_id_5_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_5);
        let lookup_memory_address_to_id_6_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_6);
        let lookup_memory_address_to_id_7_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_7);
        let lookup_memory_address_to_id_8_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_8);
        let lookup_memory_address_to_id_9_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_9);
        let lookup_memory_address_to_id_10_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_10);
        let lookup_memory_address_to_id_11_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_11);
        let lookup_memory_address_to_id_12_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_12);
        let lookup_memory_address_to_id_13_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_13);
        let lookup_memory_address_to_id_14_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_14);
        let lookup_memory_address_to_id_15_vec = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_15);

        let lookup_memory_id_to_big_0_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_memory_id_to_big_3_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_3);
        let lookup_memory_id_to_big_4_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_4);
        let lookup_memory_id_to_big_5_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_5);
        let lookup_memory_id_to_big_6_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_6);
        let lookup_memory_id_to_big_7_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_7);
        let lookup_memory_id_to_big_8_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_8);
        let lookup_memory_id_to_big_9_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_9);
        let lookup_memory_id_to_big_10_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_10);
        let lookup_memory_id_to_big_11_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_11);
        let lookup_memory_id_to_big_12_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_12);
        let lookup_memory_id_to_big_13_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_13);
        let lookup_memory_id_to_big_14_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_14);
        let lookup_memory_id_to_big_15_vec = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_15);

        let lookup_range_check_7_2_5_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_0);
        let lookup_range_check_7_2_5_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_1);
        let lookup_range_check_7_2_5_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_2);
        let lookup_range_check_7_2_5_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_3);
        let lookup_range_check_7_2_5_4_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_4);
        let lookup_range_check_7_2_5_5_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_5);
        let lookup_range_check_7_2_5_6_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_6);
        let lookup_range_check_7_2_5_7_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_7);
        let lookup_range_check_7_2_5_8_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_8);
        let lookup_range_check_7_2_5_9_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_9);
        let lookup_range_check_7_2_5_10_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_10);
        let lookup_range_check_7_2_5_11_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_11);
        let lookup_range_check_7_2_5_12_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_12);
        let lookup_range_check_7_2_5_13_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_13);
        let lookup_range_check_7_2_5_14_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_14);
        let lookup_range_check_7_2_5_15_vec = collect_lookup_ptrs!(self.lookup_data, range_check_7_2_5_15);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let blake_g_ptr = blake_g as *const _ as *mut std::os::raw::c_void;
            let blake_round_ptr = blake_round as *const _ as *mut std::os::raw::c_void;
            let blake_round_sigma_ptr = blake_round_sigma as *const _ as *mut std::os::raw::c_void;
            let memory_address_to_id_ptr = memory_address_to_id as *const _ as *mut std::os::raw::c_void;
            let memory_id_to_big_ptr = memory_id_to_big as *const _ as *mut std::os::raw::c_void;
            let range_check_7_2_5_ptr = range_check_7_2_5 as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_blake_round_interaction_traces(
                blake_g_ptr,
                blake_round_ptr,
                blake_round_sigma_ptr,
                memory_address_to_id_ptr,
                memory_id_to_big_ptr,
                range_check_7_2_5_ptr,

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


                trace_log_size as u32,
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }


        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum =  SecureField::from_m31_array([claimed_sum_vec[0], claimed_sum_vec[1], claimed_sum_vec[2], claimed_sum_vec[3]]);

        let domain = CanonicCoset::new(trace_log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}


#[cfg(test)]
pub mod tests {
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use test_log::test;

    use crate::witness::components_cuda::{blake_g_cuda, blake_round_cuda, blake_round_sigma_cuda,
        memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_7_2_5_cuda};
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use crate::witness::components::blake_round;
    use cairo_air::relations;
    use std::array;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};


    use stwo_constraint_framework::TraceLocationAllocator;
    use stwo_constraint_framework::FrameworkComponent;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::blake_round::Eval;
    use stwo_cairo_common::prover_types::simd::PackedUInt32;

    use stwo_cairo_adapter::memory::MemoryEntry;
    use stwo_cairo_adapter::memory::MemoryConfig;
    use stwo_cairo_adapter::memory::MemoryBuilder;
    use crate::witness::components::blake_g;
    use crate::witness::components::blake_round_sigma;
    use crate::witness::components::memory_address_to_id;
    use crate::witness::components::memory_id_to_big;
    use crate::witness::components::range_check_7_2_5;
    use crate::witness::components::blake_round::PackedInputType;
    use stwo::prover::backend::simd::m31::PackedM31;
    use stwo::core::fields::m31::M31;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::prover::backend::Column;
    use itertools::Itertools;
    use stwo::stwo_cuda::bindings::CudaSecureField;
    use stwo::core::fields::m31::BaseField;

    #[test]
    fn test_blake_round_cpu_ref() {
        const INPUT_ARRAY_LOG:u32 = 10;
        let mut rng = SmallRng::seed_from_u64(0);
        let max_val = 10;
        let a = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));
        let b = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let arr = array::from_fn(|_| {
            let simd = std::simd::u32x16::from_array(array::from_fn(|_| rng.gen_range(1..max_val)));
            PackedUInt32::from_simd(simd)
        });

        let c = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let input: PackedInputType = (a, b, (arr, c));
        let inputs = [input; 1<<INPUT_ARRAY_LOG];

        const N_ENTRIES: u32 = 1<<14;
        let (memory, _) = MemoryBuilder::from_iter(
            MemoryConfig::default(),
            (0..N_ENTRIES).map(|i| MemoryEntry {
                address: i as u64,
                value: [1,0,0,0,0,1,1,2],
            }),
        )
        .build();

        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&memory);
        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&memory);

        let mut blake_round_trace_generator = blake_round::ClaimGenerator::new(memory);
        let mut blake_g_trace_generator = blake_g::ClaimGenerator::new();
        let blake_round_sigma_trace_generator = blake_round_sigma::ClaimGenerator::new();
        let range_check_7_2_5_trace_generator = range_check_7_2_5::ClaimGenerator::new();

        let blake_round_relation = relations::BlakeRound::dummy();
        let blake_g_relation = relations::BlakeG::dummy();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        blake_round_trace_generator.add_packed_inputs(&inputs);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_claim, blake_round_interaction_gen) = blake_round_trace_generator.write_trace(
            &mut mock_tree_builder,
            &mut blake_g_trace_generator,
            &blake_round_sigma_trace_generator,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_7_2_5_trace_generator,
        );

        mock_tree_builder.finalize_interaction();


        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_interaction_claim = blake_round_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_g_relation,
            &blake_round_relation,
            &blake_round_sigma_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_7_2_5_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("blake_round_interaction_claim.claimed_sum: {:?}", blake_round_interaction_claim.claimed_sum);


        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round"),
                claim: blake_round_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
                range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
                blake_round_lookup_elements: relations::BlakeRound::dummy(),
            },
            blake_round_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace)
    }

    #[test]
    fn test_blake_round_trace_gen_by_cpu_and_verify_by_cuda() {
        const INPUT_ARRAY_LOG:u32 = 10;
        let mut rng = SmallRng::seed_from_u64(0);
        let max_val = 10;
        let a = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));
        let b = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let arr = array::from_fn(|_| {
            let simd = std::simd::u32x16::from_array(array::from_fn(|_| rng.gen_range(1..max_val)));
            PackedUInt32::from_simd(simd)
        });

        let c = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let input: PackedInputType = (a, b, (arr, c));
        let inputs = [input; 1<<INPUT_ARRAY_LOG];

        const N_ENTRIES: u32 = 1<<14;
        let (memory, _) = MemoryBuilder::from_iter(
            MemoryConfig::default(),
            (0..N_ENTRIES).map(|i| MemoryEntry {
                address: i as u64,
                value: [1,0,0,0,0,1,1,2],
            }),
        )
        .build();

        let memory_id_to_big_trace_generator = memory_id_to_big::ClaimGenerator::new(&memory);
        let memory_address_to_id_trace_generator = memory_address_to_id::ClaimGenerator::new(&memory);

        let mut blake_round_trace_generator = blake_round::ClaimGenerator::new(memory);
        let mut blake_g_trace_generator = blake_g::ClaimGenerator::new();
        let blake_round_sigma_trace_generator = blake_round_sigma::ClaimGenerator::new();
        let range_check_7_2_5_trace_generator = range_check_7_2_5::ClaimGenerator::new();

        let blake_round_relation = relations::BlakeRound::dummy();
        let blake_g_relation = relations::BlakeG::dummy();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        blake_round_trace_generator.add_packed_inputs(&inputs);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_claim, blake_round_interaction_gen) = blake_round_trace_generator.write_trace(
            &mut mock_tree_builder,
            &mut blake_g_trace_generator,
            &blake_round_sigma_trace_generator,
            &memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_7_2_5_trace_generator,
        );

        mock_tree_builder.finalize_interaction();


        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_interaction_claim = blake_round_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_g_relation,
            &blake_round_relation,
            &blake_round_sigma_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_7_2_5_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("blake_round_interaction_claim.claimed_sum: {:?}", blake_round_interaction_claim.claimed_sum);

        let trace0_vec: Vec<_> = trace[0].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace1_vec: Vec<_> = trace[1].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace2_vec: Vec<_> = trace[2].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();

        let trace0_evaluations_vec = trace0_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();
        let trace1_evaluations_vec = trace1_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let mut trace2_evaluations_vec = vec![];
        if trace.len() != 2 {
            trace2_evaluations_vec = trace2_vec
                .iter()
                .map(|column_evaluations| column_evaluations.device_ptr)
                .collect_vec();
        }

        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round"),
                claim: blake_round_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
                range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
                blake_round_lookup_elements: relations::BlakeRound::dummy(),
            },
            blake_round_interaction_claim.claimed_sum,
        );

        let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;

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
                blake_round_claim.log_size as u32,
                blake_round_claim.log_size as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(blake_round_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << blake_round_claim.log_size)),
                false,
                true,
            );
        }
    }

    #[test]
    fn test_blake_round_trace_gen_by_cuda_and_verify_by_cpu() {
        const INPUT_ARRAY_LOG:u32 = 10;
        let mut rng = SmallRng::seed_from_u64(0);
        let max_val = 10;
        let a = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));
        let b = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let arr = array::from_fn(|_| {
            let simd = std::simd::u32x16::from_array(array::from_fn(|_| rng.gen_range(1..max_val)));
            PackedUInt32::from_simd(simd)
        });

        let c = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let input: PackedInputType = (a, b, (arr, c));
        let inputs = [input; 1<<INPUT_ARRAY_LOG];

        const N_ENTRIES: u32 = 1<<14;
        let (memory, _) = MemoryBuilder::from_iter(
            MemoryConfig::default(),
            (0..N_ENTRIES).map(|i| MemoryEntry {
                address: i as u64,
                value: [1,0,0,0,0,1,1,2],
            }),
        )
        .build();

        let memory_id_to_big_trace_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&memory);
        let mut memory_address_to_id_trace_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&memory);

        let mut blake_round_trace_generator = blake_round_cuda::CudaClaimGenerator::new(memory);
        let mut blake_g_trace_generator = blake_g_cuda::CudaClaimGenerator::new();
        let blake_round_sigma_trace_generator = blake_round_sigma_cuda::CudaClaimGenerator::new();
        let range_check_7_2_5_trace_generator = range_check_7_2_5_cuda::CudaClaimGenerator::new_rc_7_2_5();

        let blake_round_relation = relations::BlakeRound::dummy();
        let blake_g_relation = relations::BlakeG::dummy();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        blake_round_trace_generator.add_packed_inputs(&inputs);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_claim, blake_round_interaction_gen) = blake_round_trace_generator.write_trace(
            &mut mock_tree_builder,
            &mut blake_g_trace_generator,
            &blake_round_sigma_trace_generator,
            &mut memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_7_2_5_trace_generator,
        );

        mock_tree_builder.finalize_interaction();


        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_interaction_claim = blake_round_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_g_relation,
            &blake_round_relation,
            &blake_round_sigma_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_7_2_5_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("blake_round_interaction_claim.claimed_sum: {:?}", blake_round_interaction_claim.claimed_sum);


        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round"),
                claim: blake_round_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
                range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
                blake_round_lookup_elements: relations::BlakeRound::dummy(),
            },
            blake_round_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace)


    }

    #[test]
    fn test_blake_round_trace_gen_by_cuda_and_verify_by_cuda() {
        const INPUT_ARRAY_LOG:u32 = 10;
        let mut rng = SmallRng::seed_from_u64(0);
        let max_val = 10;
        let a = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));
        let b = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let arr = array::from_fn(|_| {
            let simd = std::simd::u32x16::from_array(array::from_fn(|_| rng.gen_range(1..max_val)));
            PackedUInt32::from_simd(simd)
        });

        let c = PackedM31::from_array(array::from_fn(|_| M31(rng.gen_range(1..max_val))));

        let input: PackedInputType = (a, b, (arr, c));
        let inputs = [input; 1<<INPUT_ARRAY_LOG];

        const N_ENTRIES: u32 = 1<<14;
        let (memory, _) = MemoryBuilder::from_iter(
            MemoryConfig::default(),
            (0..N_ENTRIES).map(|i| MemoryEntry {
                address: i as u64,
                value: [1,0,0,0,0,1,1,2],
            }),
        )
        .build();

        let memory_id_to_big_trace_generator = memory_id_to_big_cuda::CudaClaimGenerator::new(&memory);
        let mut memory_address_to_id_trace_generator = memory_address_to_id_cuda::CudaClaimGenerator::new(&memory);

        let mut blake_round_trace_generator = blake_round_cuda::CudaClaimGenerator::new(memory);
        let mut blake_g_trace_generator = blake_g_cuda::CudaClaimGenerator::new();
        let blake_round_sigma_trace_generator = blake_round_sigma_cuda::CudaClaimGenerator::new();
        let range_check_7_2_5_trace_generator = range_check_7_2_5_cuda::CudaClaimGenerator::new_rc_7_2_5();

        let blake_round_relation = relations::BlakeRound::dummy();
        let blake_g_relation = relations::BlakeG::dummy();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_7_2_5_relation = relations::RangeCheck_7_2_5::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(4);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        blake_round_trace_generator.add_packed_inputs(&inputs);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_claim, blake_round_interaction_gen) = blake_round_trace_generator.write_trace(
            &mut mock_tree_builder,
            &mut blake_g_trace_generator,
            &blake_round_sigma_trace_generator,
            &mut memory_address_to_id_trace_generator,
            &memory_id_to_big_trace_generator,
            &range_check_7_2_5_trace_generator,
        );

        mock_tree_builder.finalize_interaction();


        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_interaction_claim = blake_round_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_g_relation,
            &blake_round_relation,
            &blake_round_sigma_relation,
            &memory_address_to_id_relation,
            &memory_id_to_big_relation,
            &range_check_7_2_5_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("blake_round_interaction_claim.claimed_sum: {:?}", blake_round_interaction_claim.claimed_sum);
        let trace0_vec: Vec<_> = trace[0].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace1_vec: Vec<_> = trace[1].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();
        let trace2_vec: Vec<_> = trace[2].clone().into_iter().map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec())).collect();

        let trace0_evaluations_vec = trace0_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();
        let trace1_evaluations_vec = trace1_vec
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let mut trace2_evaluations_vec = vec![];
        if trace.len() != 2 {
            trace2_evaluations_vec = trace2_vec
                .iter()
                .map(|column_evaluations| column_evaluations.device_ptr)
                .collect_vec();
        }

        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);100].to_vec());

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round"),
                claim: blake_round_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
                range_check_7_2_5_lookup_elements: relations::RangeCheck_7_2_5::dummy(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
                blake_round_lookup_elements: relations::BlakeRound::dummy(),
            },
            blake_round_interaction_claim.claimed_sum,
        );

        let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;

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
                blake_round_claim.log_size as u32,
                blake_round_claim.log_size as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(blake_round_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << blake_round_claim.log_size)),
                false,
                true,
            );
        }


    }

}