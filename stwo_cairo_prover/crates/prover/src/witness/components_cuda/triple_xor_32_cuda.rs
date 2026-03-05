#![allow(unused_parens)]
use cairo_air::components::triple_xor_32::{Claim, InteractionClaim, N_TRACE_COLUMNS};
use cairo_air::relations::CommonLookupElements;

use super::verify_bitwise_xor::{vbx_8, vbx_8_b};
use crate::witness::prelude::*;

pub type PackedInputType = [PackedUInt32; 3];
pub type CudaPackedInputType = [Uint32Vec; 3];
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

pub const N_INTERACTION_TRACE_COLUMNS: usize = 5;

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
    pub packed_inputs: CudaPackedInputType,
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
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        verify_bitwise_xor_8_state: &vbx_8::CudaClaimGenerator,
        verify_bitwise_xor_8_b_state: &vbx_8_b::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        assert!(!self.packed_inputs.is_empty());
        let log_size = self.size.ilog2();

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(self.packed_inputs);
        verify_bitwise_xor_8_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
        verify_bitwise_xor_8_b_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8_b);

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                lookup_data,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {
        self.packed_inputs = (0..3)
            .map(|i| {
                let elements: Vec<_> = packed_inputs
                    .iter()
                    .flat_map(|input| input[i].as_array())
                    .collect();
                Uint32Vec::from_vec(unsafe { std::mem::transmute(elements) })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Inputs should have exactly 3 elements");

        // for i in 0..self.packed_inputs.len() {
        //     println!("self.packed_inputs[{}]: {:?}", i, self.packed_inputs[i].to_vec());
        // }
        self.size = self.packed_inputs[0].size as u32;
    }

    pub fn add_cuda_inputs(&mut self, inputs: &[CudaPackedInputType]) {
        // On first call, replace the initial zeroed data instead of extending
        let mut first = self.size <= 1;
        inputs.iter().for_each(|input| {
            if first {
                // Replace initial dummy data
                self.packed_inputs[0] = input[0].clone();
                self.packed_inputs[1] = input[1].clone();
                self.packed_inputs[2] = input[2].clone();
                first = false;
            } else {
                // Extend existing data
                self.packed_inputs[0].extend(&input[0]);
                self.packed_inputs[1].extend(&input[1]);
                self.packed_inputs[2].extend(&input[2]);
            }
        });
        self.size = self.packed_inputs[0].size as u32;
    }
}

struct CudaSubComponentInputs {
    verify_bitwise_xor_8: [vbx_8::CudaPackedInputType; 4],
    verify_bitwise_xor_8_b: [vbx_8_b::CudaPackedInputType; 4],
}

fn write_trace_cuda(
    inputs: [Uint32Vec; 3],
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    // n_rows is the actual row count, NOT inputs.len() which would be 3 (array length)
    let n_rows = inputs[0].size;
    let log_size = n_rows.ilog2();

    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                triple_xor_32_0: init_lookup_array!(log_size),
                verify_bitwise_xor_8_0: init_lookup_array!(log_size),
                verify_bitwise_xor_8_1: init_lookup_array!(log_size),
                verify_bitwise_xor_8_2: init_lookup_array!(log_size),
                verify_bitwise_xor_8_3: init_lookup_array!(log_size),
                verify_bitwise_xor_8_b_0: init_lookup_array!(log_size),
                verify_bitwise_xor_8_b_1: init_lookup_array!(log_size),
                verify_bitwise_xor_8_b_2: init_lookup_array!(log_size),
                verify_bitwise_xor_8_b_3: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_bitwise_xor_8: init_subcomponent_array!(log_size),
                verify_bitwise_xor_8_b: init_subcomponent_array!(log_size),
            },
        )
    };
    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    // collect lookup_data pointers
    let lookup_triple_xor_32_0_vec = collect_lookup_ptrs!(lookup_data, triple_xor_32_0);
    let lookup_verify_bitwise_xor_8_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_0);
    let lookup_verify_bitwise_xor_8_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_1);
    let lookup_verify_bitwise_xor_8_2_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_2);
    let lookup_verify_bitwise_xor_8_3_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_3);
    let lookup_verify_bitwise_xor_8_b_0_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_0);
    let lookup_verify_bitwise_xor_8_b_1_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_1);
    let lookup_verify_bitwise_xor_8_b_2_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_2);
    let lookup_verify_bitwise_xor_8_b_3_vec =
        collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_3);
    // collect sub_component_inputs pointers
    let sub_component_inputs_verify_bitwise_xor_8_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8);
    let sub_component_inputs_verify_bitwise_xor_8_b_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8_b);

    let triple_xor_32_input_vec = collect_input_ptrs!(inputs);

    unsafe {
        bindings_airs::generate_triple_xor_32_traces(
            traces_vec.as_ptr(),
            lookup_triple_xor_32_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_1_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_2_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_3_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_b_0_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_b_1_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_b_2_vec.as_ptr(),
            lookup_verify_bitwise_xor_8_b_3_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_8_vec.as_ptr(),
            sub_component_inputs_verify_bitwise_xor_8_b_vec.as_ptr(),
            triple_xor_32_input_vec.as_ptr(),
            log_size as u32,
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    triple_xor_32_0: [BaseFieldVec; 8],
    verify_bitwise_xor_8_0: [BaseFieldVec; 3],
    verify_bitwise_xor_8_1: [BaseFieldVec; 3],
    verify_bitwise_xor_8_2: [BaseFieldVec; 3],
    verify_bitwise_xor_8_3: [BaseFieldVec; 3],
    verify_bitwise_xor_8_b_0: [BaseFieldVec; 3],
    verify_bitwise_xor_8_b_1: [BaseFieldVec; 3],
    verify_bitwise_xor_8_b_2: [BaseFieldVec; 3],
    verify_bitwise_xor_8_b_3: [BaseFieldVec; 3],
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
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_triple_xor_32_0_vec = collect_lookup_ptrs!(self.lookup_data, triple_xor_32_0);
        let lookup_verify_bitwise_xor_8_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_0);
        let lookup_verify_bitwise_xor_8_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_1);
        let lookup_verify_bitwise_xor_8_2_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_2);
        let lookup_verify_bitwise_xor_8_3_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_3);
        let lookup_verify_bitwise_xor_8_b_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_0);
        let lookup_verify_bitwise_xor_8_b_1_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_1);
        let lookup_verify_bitwise_xor_8_b_2_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_2);
        let lookup_verify_bitwise_xor_8_b_3_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_3);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_txor32 =
                create_modified_lookup_for_cuda(lookup_elements, TRIPLE_XOR_32_RELATION_ID);
            let mod_vbx8 =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_BITWISE_XOR_8_RELATION_ID);
            let mod_vbx8b = create_modified_lookup_for_cuda(
                lookup_elements,
                VERIFY_BITWISE_XOR_8_B_RELATION_ID,
            );

            bindings_airs::generate_triple_xor_32_interaction_traces(
                &mod_txor32 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx8 as *const _ as *mut std::os::raw::c_void,
                &mod_vbx8b as *const _ as *mut std::os::raw::c_void,
                lookup_triple_xor_32_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_2_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_3_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_b_0_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_b_1_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_b_2_vec.as_ptr(),
                lookup_verify_bitwise_xor_8_b_3_vec.as_ptr(),
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
