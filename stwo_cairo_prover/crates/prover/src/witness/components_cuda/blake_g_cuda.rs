#![allow(unused_parens)]

use cairo_air::components::blake_g::{Claim, InteractionClaim, N_TRACE_COLUMNS};

use crate::witness::components_cuda::{
    verify_bitwise_xor_12_cuda, verify_bitwise_xor_4_cuda, verify_bitwise_xor_7_cuda, verify_bitwise_xor_8_cuda,
    verify_bitwise_xor_9_cuda,
};
use crate::witness::prelude::*;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::core::fields::qm31::SecureField;

use stwo::core::fields::m31::BaseField;
use itertools::Itertools;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::stwo_cuda::bindings_airs;
use stwo::prover::backend::Column;
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
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        verify_bitwise_xor_12_state: &verify_bitwise_xor_12_cuda::CudaClaimGenerator,
        verify_bitwise_xor_4_state: &verify_bitwise_xor_4_cuda::CudaClaimGenerator,
        verify_bitwise_xor_7_state: &verify_bitwise_xor_7_cuda::CudaClaimGenerator,
        verify_bitwise_xor_8_state: &verify_bitwise_xor_8_cuda::CudaClaimGenerator,
        verify_bitwise_xor_9_state: &verify_bitwise_xor_9_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        assert!(!self.packed_inputs.is_empty());
        let n_rows: usize = self.packed_inputs.len();
        let log_size = self.size.ilog2();
        println!("blake_g write_trace:  n_rows:{}, trace_log_size: {}", n_rows, log_size);

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            self.packed_inputs,
            n_rows,
        );

        verify_bitwise_xor_8_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
        verify_bitwise_xor_12_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_12);
        verify_bitwise_xor_4_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_4);
        verify_bitwise_xor_7_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_7);
        verify_bitwise_xor_9_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_9);

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                lookup_data,
            },
        )
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {

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
        // }
        self.size = self.packed_inputs[0].size as u32;
     }

    pub fn add_cuda_inputs(&mut self, inputs: &[CudaPackedInputType]) {
        for input in inputs {
            for (input_slice, old_slice) in self.packed_inputs.iter_mut().zip(input.iter()) {
                input_slice.extend(old_slice);
            }
            self.size += input[0].size as u32;
        }
    }

}

struct CudaSubComponentInputs {
    verify_bitwise_xor_8: [verify_bitwise_xor_8_cuda::CudaPackedInputType; 8],
    verify_bitwise_xor_12: [verify_bitwise_xor_12_cuda::CudaPackedInputType; 2],
    verify_bitwise_xor_4: [verify_bitwise_xor_4_cuda::CudaPackedInputType; 2],
    verify_bitwise_xor_7: [verify_bitwise_xor_7_cuda::CudaPackedInputType; 2],
    verify_bitwise_xor_9: [verify_bitwise_xor_9_cuda::CudaPackedInputType; 2],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    inputs:  [Uint32Vec; 6],
    n_rows: usize,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let n_rows = inputs.len();
    let log_size = inputs[0].size.ilog2();
    println!("trace log_size: {}, n_rows:{}", log_size, n_rows);
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
    let lookup_verify_bitwise_xor_12_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_12_0);
    let lookup_verify_bitwise_xor_12_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_12_1);
    let lookup_verify_bitwise_xor_4_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_4_0);
    let lookup_verify_bitwise_xor_4_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_4_1);
    let lookup_verify_bitwise_xor_7_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_7_0);
    let lookup_verify_bitwise_xor_7_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_7_1);
    let lookup_verify_bitwise_xor_8_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_0);
    let lookup_verify_bitwise_xor_8_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_1);
    let lookup_verify_bitwise_xor_8_2_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_2);
    let lookup_verify_bitwise_xor_8_3_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_3);
    let lookup_verify_bitwise_xor_8_4_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_4);
    let lookup_verify_bitwise_xor_8_5_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_5);
    let lookup_verify_bitwise_xor_8_6_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_6);
    let lookup_verify_bitwise_xor_8_7_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_7);
    let lookup_verify_bitwise_xor_9_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_0);
    let lookup_verify_bitwise_xor_9_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_9_1);
    // collect sub_component_inputs pointers
    let sub_component_inputs_verify_bitwise_xor_8_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8);
    let sub_component_inputs_verify_bitwise_xor_12_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_12);
    let sub_component_inputs_verify_bitwise_xor_4_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_4);
    let sub_component_inputs_verify_bitwise_xor_7_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_7);
    let sub_component_inputs_verify_bitwise_xor_9_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_9);

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

            sub_component_inputs_verify_bitwise_xor_8_vec.as_ptr(),
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



pub struct CudaInteractionClaimGenerator {
    log_size: u32,
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        blake_g: &relations::BlakeG,
        verify_bitwise_xor_12: &relations::VerifyBitwiseXor_12,
        verify_bitwise_xor_4: &relations::VerifyBitwiseXor_4,
        verify_bitwise_xor_7: &relations::VerifyBitwiseXor_7,
        verify_bitwise_xor_8: &relations::VerifyBitwiseXor_8,
        verify_bitwise_xor_9: &relations::VerifyBitwiseXor_9,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_blake_g_0_vec = collect_lookup_ptrs!(self.lookup_data, blake_g_0);
        let lookup_verify_bitwise_xor_12_0_vec= collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_12_0);
        let lookup_verify_bitwise_xor_12_1_vec= collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_12_1);
        let lookup_verify_bitwise_xor_4_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_4_0);
        let lookup_verify_bitwise_xor_4_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_4_1);
        let lookup_verify_bitwise_xor_7_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_7_0);
        let lookup_verify_bitwise_xor_7_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_7_1);
        let lookup_verify_bitwise_xor_8_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_0);
        let lookup_verify_bitwise_xor_8_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_1);
        let lookup_verify_bitwise_xor_8_2_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_2);
        let lookup_verify_bitwise_xor_8_3_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_3);
        let lookup_verify_bitwise_xor_8_4_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_4);
        let lookup_verify_bitwise_xor_8_5_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_5);
        let lookup_verify_bitwise_xor_8_6_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_6);
        let lookup_verify_bitwise_xor_8_7_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_7);
        let lookup_verify_bitwise_xor_9_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_0);
        let lookup_verify_bitwise_xor_9_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_9_1);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let blake_g_ptr = blake_g as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_12_ptr = verify_bitwise_xor_12 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_4_ptr = verify_bitwise_xor_4 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_7_ptr = verify_bitwise_xor_7 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_8_ptr = verify_bitwise_xor_8 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_9_ptr = verify_bitwise_xor_9 as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_blake_g_interaction_traces(
                blake_g_ptr,
                verify_bitwise_xor_12_ptr,
                verify_bitwise_xor_4_ptr,
                verify_bitwise_xor_7_ptr,
                verify_bitwise_xor_8_ptr,
                verify_bitwise_xor_9_ptr,

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

        println!("cuda claimed sum: {:?}", claimed_sum);
        InteractionClaim { claimed_sum }
    }
}



#[cfg(test)]
pub mod tests {
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use stwo_cairo_common::prover_types::cpu::M31;
    use stwo::core::pcs::TreeVec;
    use test_log::test;

    use crate::witness::components_cuda::{
        blake_g_cuda,
        verify_bitwise_xor_12_cuda, verify_bitwise_xor_4_cuda, verify_bitwise_xor_7_cuda, verify_bitwise_xor_8_cuda,
        verify_bitwise_xor_9_cuda,
    };
    use crate::witness::components::{
        blake_g,
        verify_bitwise_xor_12, verify_bitwise_xor_4, verify_bitwise_xor_7, verify_bitwise_xor_8,
        verify_bitwise_xor_8_b, verify_bitwise_xor_9,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;

    use cairo_air::relations;
    use std::array;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo_cairo_common::prover_types::simd::PackedUInt32;


    use stwo_constraint_framework::TraceLocationAllocator;
    use stwo_constraint_framework::FrameworkComponent;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::blake_g::Eval;
    use crate::witness::components_cuda::blake_g_cuda::PackedInputType;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

    // Helper function to get environment variable with default value
    fn get_env_var<T: std::fmt::Display + std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .unwrap_or_else(|_| default.to_string())
            .parse()
            .unwrap_or(default)
    }

    #[test]
    fn test_blake_g_cpu_trace_gen_ref() {
        let input_log = get_env_var("INPUT_LOG", 4u32);

        let mut rng = SmallRng::seed_from_u64(123456);
        let packed_inputs: Vec<PackedInputType> = (0..(1<<input_log))
            .map(|_| {
                array::from_fn(|_| {
                    let arr = array::from_fn(|_| rng.gen());
                    let simd = std::simd::u32x16::from_array(arr);
                    PackedUInt32::from_simd(simd)
                })
            })
            .collect();

        for _ in 0..4 {
            let mut blake_g = blake_g::ClaimGenerator::new();
            let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4::ClaimGenerator::new();
            let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7::ClaimGenerator::new();
            let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();
            let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b::ClaimGenerator::new();
            let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9::ClaimGenerator::new();
            let verify_bitwise_xor_12_trace_generator = verify_bitwise_xor_12::ClaimGenerator::new();

            let blake_g_relation = relations::BlakeG::dummy();
            let verify_bitwise_xor_4_relation = relations::VerifyBitwiseXor_4::dummy();
            let verify_bitwise_xor_7_relation = relations::VerifyBitwiseXor_7::dummy();
            let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
            let verify_bitwise_xor_8_b_relation = relations::VerifyBitwiseXor_8_B::dummy();
            let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
            let verify_bitwise_xor_12_relation = relations::VerifyBitwiseXor_12::dummy();

            blake_g.add_packed_inputs(&packed_inputs);

            let mut mock_commitment_scheme = MockCommitmentScheme::default();

            // Preprocessed.
            let preprocessed_trace = testing_preprocessed_tree(input_log);
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
            mock_tree_builder.finalize_interaction();

            // Base trace.
            let start = std::time::Instant::now();
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            let (blake_g_claim, blake_g_interaction_gen) = blake_g.write_trace(
                &mut mock_tree_builder,
                &verify_bitwise_xor_12_trace_generator,
                &verify_bitwise_xor_4_trace_generator,
                &verify_bitwise_xor_7_trace_generator,
                &verify_bitwise_xor_8_trace_generator,
                &verify_bitwise_xor_8_b_trace_generator,
                &verify_bitwise_xor_9_trace_generator,
            );
            println!("CPU log_n:{}, base trace gen time: {:?}", blake_g_claim.log_size, start.elapsed());
            mock_tree_builder.finalize_interaction();

            // Interaction trace.
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            let start = std::time::Instant::now();
            let blake_g_interaction_claim = blake_g_interaction_gen.write_interaction_trace(
                &mut mock_tree_builder,
                &verify_bitwise_xor_8_relation,
                &verify_bitwise_xor_8_b_relation,
                &verify_bitwise_xor_12_relation,
                &verify_bitwise_xor_4_relation,
                &verify_bitwise_xor_7_relation,
                &verify_bitwise_xor_9_relation,
                &blake_g_relation,
            );
            println!("CPU log_n:{}, interaction trace gen time: {:?}", blake_g_claim.log_size, start.elapsed());
            mock_tree_builder.finalize_interaction();
            let trace = mock_commitment_scheme.trace_domain_evaluations();

            // for i in 0..trace[1].len() {
            // }

            // for i in 0..trace[2].len() {
            // }

            let tree_span_provider = &mut TraceLocationAllocator::default();
            let component = FrameworkComponent::new(
                tree_span_provider,
                Eval {
                eval_id: fnv1a_eval_id_gen("blake_g"),
                    claim: blake_g_claim,
                    verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                    verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                    verify_bitwise_xor_12_lookup_elements: relations::VerifyBitwiseXor_12::dummy(),
                    verify_bitwise_xor_4_lookup_elements: relations::VerifyBitwiseXor_4::dummy(),
                    verify_bitwise_xor_7_lookup_elements: relations::VerifyBitwiseXor_7::dummy(),
                    verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                    blake_g_lookup_elements: relations::BlakeG::dummy(),
                },
                blake_g_interaction_claim.claimed_sum,
            );

            let start = std::time::Instant::now();
            assert_component(&component, &trace);
            println!("CPU log_n:{}, evaluate time: {:?}", blake_g_claim.log_size, start.elapsed());
        }
    }

    // try to modify /home/ubuntu/stwo-cairo/stwo/crates/prover/src/stwo_cuda/cuda/witness/gen_blake_g_trace.cu
    #[test]
    fn test_blake_g_trace_gen_by_gpu_and_verify_by_cpu() {
        let mut rng = SmallRng::seed_from_u64(123456);
        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(PackedUInt32::from_simd);

        let mut blake_g_cuda = blake_g_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_12_trace_generator = verify_bitwise_xor_12_cuda::CudaClaimGenerator::new();

        let blake_g_relation = relations::BlakeG::dummy();
        let verify_bitwise_xor_4_relation = relations::VerifyBitwiseXor_4::dummy();
        let verify_bitwise_xor_7_relation = relations::VerifyBitwiseXor_7::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
        let verify_bitwise_xor_12_relation = relations::VerifyBitwiseXor_12::dummy();

        println!("input {:?}", input);
        blake_g_cuda.add_packed_inputs(&[input]);

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        // Base trace.
        let (blake_g_claim, blake_g_interaction_gen) = blake_g_cuda.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_12_trace_generator,
            &verify_bitwise_xor_4_trace_generator,
            &verify_bitwise_xor_7_trace_generator,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_9_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        println!("blake_g_claim: {:?}", blake_g_claim.log_size);
        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_g_interaction_claim = blake_g_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_g_relation,
            &verify_bitwise_xor_12_relation,
            &verify_bitwise_xor_4_relation,
            &verify_bitwise_xor_7_relation,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_9_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        for i in 0..trace[1].len() {
            println!("cuda base traces[{}]: {:?}", i, trace[1][i].to_vec());
        }

        for i in 0..trace[2].len() {
            println!("cuda interaction traces[{}]: {:?}", i, trace[2][i].to_vec());
        }
        println!("blake_g_interaction_claim.claimed_sum: {:?}", blake_g_interaction_claim.claimed_sum);

        let new_traces_1: Vec<Vec<M31>> = trace[1].iter().map(|x| x.to_vec()).collect();
        let new_traces_2: Vec<Vec<M31>> = trace[2].iter().map(|x| x.to_vec()).collect();

        let cuda_trace_to_cpu = TreeVec::new(
            vec![
                vec![],
                new_traces_1.iter().collect(),
                new_traces_2.iter().collect(),
            ],
        );

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_g"),
                claim: blake_g_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                verify_bitwise_xor_12_lookup_elements: relations::VerifyBitwiseXor_12::dummy(),
                verify_bitwise_xor_4_lookup_elements: relations::VerifyBitwiseXor_4::dummy(),
                verify_bitwise_xor_7_lookup_elements: relations::VerifyBitwiseXor_7::dummy(),
                verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
            },
            blake_g_interaction_claim.claimed_sum,
        );

        assert_component(&component, &cuda_trace_to_cpu)


    }

    #[test]
    fn stress_test_blake_g_trace_gen_by_gpu_and_verify_by_cpu() {
        let input_log = get_env_var("INPUT_LOG", 8u32);

        for _ in 0..2 {
            let mut rng = SmallRng::from_entropy();
            let packed_inputs: Vec<PackedInputType> = (0..(1<<input_log))
                .map(|_| {
                    array::from_fn(|_| {
                        let arr = array::from_fn(|_| rng.gen());
                        let simd = std::simd::u32x16::from_array(arr);
                        PackedUInt32::from_simd(simd)
                    })
                })
                .collect();

            let mut blake_g_cuda = blake_g_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_12_trace_generator = verify_bitwise_xor_12_cuda::CudaClaimGenerator::new();

            let blake_g_relation = relations::BlakeG::dummy();
            let verify_bitwise_xor_4_relation = relations::VerifyBitwiseXor_4::dummy();
            let verify_bitwise_xor_7_relation = relations::VerifyBitwiseXor_7::dummy();
            let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
            let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
            let verify_bitwise_xor_12_relation = relations::VerifyBitwiseXor_12::dummy();

            blake_g_cuda.add_packed_inputs(&packed_inputs);

            let mut mock_commitment_scheme = MockCommitmentScheme::default();

            // Preprocessed.
            mock_commitment_scheme.tree_builder().finalize_interaction();
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

            // Base trace.
            let start = std::time::Instant::now();
            let (blake_g_claim, blake_g_interaction_gen) = blake_g_cuda.write_trace(
                &mut mock_tree_builder,
                &verify_bitwise_xor_12_trace_generator,
                &verify_bitwise_xor_4_trace_generator,
                &verify_bitwise_xor_7_trace_generator,
                &verify_bitwise_xor_8_trace_generator,
                &verify_bitwise_xor_9_trace_generator,
            );
            println!("blake_g_cuda log_n:{}, write_trace time: {:?}", blake_g_claim.log_size, start.elapsed());
            mock_tree_builder.finalize_interaction();

            // Interaction trace.
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            let start = std::time::Instant::now();
            let blake_g_interaction_claim = blake_g_interaction_gen.write_interaction_trace(
                &mut mock_tree_builder,
                &blake_g_relation,
                &verify_bitwise_xor_12_relation,
                &verify_bitwise_xor_4_relation,
                &verify_bitwise_xor_7_relation,
                &verify_bitwise_xor_8_relation,
                &verify_bitwise_xor_9_relation,
            );
            println!("blake_g_interaction_gen log_n:{} write_interaction_trace time: {:?}", blake_g_claim.log_size, start.elapsed());
            mock_tree_builder.finalize_interaction();
            let trace = mock_commitment_scheme.trace_domain_evaluations();

            let new_traces_1: Vec<Vec<M31>> = trace[1].iter().map(|x| x.to_vec()).collect();
            let new_traces_2: Vec<Vec<M31>> = trace[2].iter().map(|x| x.to_vec()).collect();

            let cuda_trace_to_cpu = TreeVec::new(
                vec![
                    vec![],
                    new_traces_1.iter().collect(),
                    new_traces_2.iter().collect(),
                ],
            );

            let tree_span_provider = &mut TraceLocationAllocator::default();
            let component = FrameworkComponent::new(
                tree_span_provider,
                Eval {
                eval_id: fnv1a_eval_id_gen("blake_g"),
                    claim: blake_g_claim,
                    verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                    verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                    verify_bitwise_xor_12_lookup_elements: relations::VerifyBitwiseXor_12::dummy(),
                    verify_bitwise_xor_4_lookup_elements: relations::VerifyBitwiseXor_4::dummy(),
                    verify_bitwise_xor_7_lookup_elements: relations::VerifyBitwiseXor_7::dummy(),
                    verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                    blake_g_lookup_elements: relations::BlakeG::dummy(),
                },
                blake_g_interaction_claim.claimed_sum,
            );
            let start = std::time::Instant::now();
            assert_component(&component, &cuda_trace_to_cpu);
            println!("blake_g_interaction_gen log_n:{}, evaluate time: {:?}", blake_g_claim.log_size, start.elapsed());

        }

    }

    // try to modify /home/ubuntu/stwo-cairo/stwo/crates/prover/src/stwo_cuda/cuda/constraints/evaluate_xor_rot_32R16.cuh
    #[test]
    fn blake_g_trace_gen_by_cpu_and_verify_by_cuda() {
        use itertools::Itertools;
        use stwo::prover::backend::Column;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::core::fields::m31::BaseField;

        // let log_size = 4;
        let mut rng = SmallRng::seed_from_u64(123456);
        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(PackedUInt32::from_simd);

        let mut blake_g = blake_g::ClaimGenerator::new();
        let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4::ClaimGenerator::new();
        let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7::ClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b::ClaimGenerator::new();
        let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9::ClaimGenerator::new();
        let verify_bitwise_xor_12_trace_generator = verify_bitwise_xor_12::ClaimGenerator::new();

        let blake_g_relation = relations::BlakeG::dummy();
        let verify_bitwise_xor_4_relation = relations::VerifyBitwiseXor_4::dummy();
        let verify_bitwise_xor_7_relation = relations::VerifyBitwiseXor_7::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_8_b_relation = relations::VerifyBitwiseXor_8_B::dummy();
        let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
        let verify_bitwise_xor_12_relation = relations::VerifyBitwiseXor_12::dummy();


        println!("input {:?}", input);
        blake_g.add_packed_inputs(&[input]);

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        // Base trace.
        let (blake_g_claim, blake_g_interaction_gen) = blake_g.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_12_trace_generator,
            &verify_bitwise_xor_4_trace_generator,
            &verify_bitwise_xor_7_trace_generator,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_8_b_trace_generator,
            &verify_bitwise_xor_9_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_g_interaction_claim = blake_g_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_8_b_relation,
            &verify_bitwise_xor_12_relation,
            &verify_bitwise_xor_4_relation,
            &verify_bitwise_xor_7_relation,
            &verify_bitwise_xor_9_relation,
            &blake_g_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // for i in 0..trace[0].len() {
        // }

        // for i in 0..trace[1].len() {
        // }

        // for i in 0..trace[2].len() {
        // }

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
                eval_id: fnv1a_eval_id_gen("blake_g"),
                claim: blake_g_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                verify_bitwise_xor_12_lookup_elements: relations::VerifyBitwiseXor_12::dummy(),
                verify_bitwise_xor_4_lookup_elements: relations::VerifyBitwiseXor_4::dummy(),
                verify_bitwise_xor_7_lookup_elements: relations::VerifyBitwiseXor_7::dummy(),
                verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
            },
            blake_g_interaction_claim.claimed_sum,
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
                blake_g_claim.log_size as u32,
                blake_g_claim.log_size as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(blake_g_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << blake_g_claim.log_size)),
                false,
                true,
            );
        }

    }


    // try to modify /home/ubuntu/stwo-cairo/stwo/crates/prover/src/stwo_cuda/cuda/witness/gen_blake_g_trace.cu
    #[test]
    fn blake_g_trace_gen_by_cuda_and_verify_by_cuda() {
        use itertools::Itertools;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::core::fields::m31::BaseField;
        use stwo::prover::backend::Column;

        let mut rng = SmallRng::seed_from_u64(123456);
        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(PackedUInt32::from_simd);

        let mut blake_g_cuda = blake_g_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_12_trace_generator = verify_bitwise_xor_12_cuda::CudaClaimGenerator::new();

        let blake_g_relation = relations::BlakeG::dummy();
        let verify_bitwise_xor_4_relation = relations::VerifyBitwiseXor_4::dummy();
        let verify_bitwise_xor_7_relation = relations::VerifyBitwiseXor_7::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
        let verify_bitwise_xor_12_relation = relations::VerifyBitwiseXor_12::dummy();

        println!("input {:?}", input);
        blake_g_cuda.add_packed_inputs(&[input]);

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        // Base trace.
        let (blake_g_claim, blake_g_interaction_gen) = blake_g_cuda.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_12_trace_generator,
            &verify_bitwise_xor_4_trace_generator,
            &verify_bitwise_xor_7_trace_generator,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_9_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_g_interaction_claim = blake_g_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_g_relation,
            &verify_bitwise_xor_12_relation,
            &verify_bitwise_xor_4_relation,
            &verify_bitwise_xor_7_relation,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_9_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // for i in 0..trace[0].len() {
        // }

        // for i in 0..trace[1].len() {
        // }

        // for i in 0..trace[2].len() {
        // }
        println!("cuda blake_g_interaction_claim.claimed_sum: {:?}", blake_g_interaction_claim.claimed_sum);

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
                eval_id: fnv1a_eval_id_gen("blake_g"),
                claim: blake_g_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                verify_bitwise_xor_12_lookup_elements: relations::VerifyBitwiseXor_12::dummy(),
                verify_bitwise_xor_4_lookup_elements: relations::VerifyBitwiseXor_4::dummy(),
                verify_bitwise_xor_7_lookup_elements: relations::VerifyBitwiseXor_7::dummy(),
                verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                blake_g_lookup_elements: relations::BlakeG::dummy(),
            },
            blake_g_interaction_claim.claimed_sum,
        );

        //     "component logup_counts: {:?}, sum:{:?}",
        //     component.info.logup_counts,
        //     component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>()
        // );

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
                blake_g_claim.log_size as u32,
                blake_g_claim.log_size as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(blake_g_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << blake_g_claim.log_size)),
                false,
                true,
            );
        }

    }


    #[test]
    fn stree_test_blake_g_trace_gen_by_cuda_and_verify_by_cuda() {
        use itertools::Itertools;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::core::fields::m31::BaseField;
        use stwo::prover::backend::Column;
        let input_log = get_env_var("INPUT_LOG", 8u32);

        for _ in 0..1000 {
            let mut rng = SmallRng::from_entropy();
            let packed_inputs: Vec<PackedInputType> = (0..(1<<input_log))
                .map(|_| {
                    array::from_fn(|_| {
                        let arr = array::from_fn(|_| rng.gen());
                        let simd = std::simd::u32x16::from_array(arr);
                        PackedUInt32::from_simd(simd)
                    })
                })
                .collect();

            let mut blake_g_cuda = blake_g_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_4_trace_generator = verify_bitwise_xor_4_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_7_trace_generator = verify_bitwise_xor_7_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_9_trace_generator = verify_bitwise_xor_9_cuda::CudaClaimGenerator::new();
            let verify_bitwise_xor_12_trace_generator = verify_bitwise_xor_12_cuda::CudaClaimGenerator::new();

            let blake_g_relation = relations::BlakeG::dummy();
            let verify_bitwise_xor_4_relation = relations::VerifyBitwiseXor_4::dummy();
            let verify_bitwise_xor_7_relation = relations::VerifyBitwiseXor_7::dummy();
            let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
            let verify_bitwise_xor_9_relation = relations::VerifyBitwiseXor_9::dummy();
            let verify_bitwise_xor_12_relation = relations::VerifyBitwiseXor_12::dummy();

            blake_g_cuda.add_packed_inputs(&packed_inputs);

            let mut mock_commitment_scheme = MockCommitmentScheme::default();

            // Preprocessed.
            let preprocessed_trace = testing_preprocessed_tree(input_log);
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
            mock_tree_builder.finalize_interaction();


            // Base trace.
            let start = std::time::Instant::now();
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            let (blake_g_claim, blake_g_interaction_gen) = blake_g_cuda.write_trace(
                &mut mock_tree_builder,
                &verify_bitwise_xor_12_trace_generator,
                &verify_bitwise_xor_4_trace_generator,
                &verify_bitwise_xor_7_trace_generator,
                &verify_bitwise_xor_8_trace_generator,
                &verify_bitwise_xor_9_trace_generator,
            );
            println!("blake_g_cuda log_n:{} write_trace time: {:?}", blake_g_claim.log_size, start.elapsed());
            mock_tree_builder.finalize_interaction();

            // Interaction trace.
            let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
            let start = std::time::Instant::now();
            let blake_g_interaction_claim = blake_g_interaction_gen.write_interaction_trace(
                &mut mock_tree_builder,
                &blake_g_relation,
                &verify_bitwise_xor_12_relation,
                &verify_bitwise_xor_4_relation,
                &verify_bitwise_xor_7_relation,
                &verify_bitwise_xor_8_relation,
                &verify_bitwise_xor_9_relation,
            );
            println!("blake_g_interaction_gen log_n:{}, write_interaction_trace time: {:?}", blake_g_claim.log_size, start.elapsed());
            mock_tree_builder.finalize_interaction();
            let trace = mock_commitment_scheme.trace_domain_evaluations();

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

            let len = 1 << (input_log + 4);
            let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); len]);
            let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); len]);

            let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); len]);
            let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); len]);
            let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); len]);
            let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); len]);

            let tree_span_provider = &mut TraceLocationAllocator::default();
            let component = FrameworkComponent::new(
                tree_span_provider,
                Eval {
                eval_id: fnv1a_eval_id_gen("blake_g"),
                    claim: blake_g_claim,
                    verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                    verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                    verify_bitwise_xor_12_lookup_elements: relations::VerifyBitwiseXor_12::dummy(),
                    verify_bitwise_xor_4_lookup_elements: relations::VerifyBitwiseXor_4::dummy(),
                    verify_bitwise_xor_7_lookup_elements: relations::VerifyBitwiseXor_7::dummy(),
                    verify_bitwise_xor_9_lookup_elements: relations::VerifyBitwiseXor_9::dummy(),
                    blake_g_lookup_elements: relations::BlakeG::dummy(),
                },
                blake_g_interaction_claim.claimed_sum,
            );

            let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;

            let start = std::time::Instant::now();
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
                    blake_g_claim.log_size as u32,
                    blake_g_claim.log_size as u32,
                    component.info.n_constraints as u32,
                    component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                    eval_ptr,
                    CudaSecureField::from(blake_g_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << blake_g_claim.log_size)),
                    false,
                    true,
                );
            }
            println!("blake_g_cuda log_n:{} evaluate time: {:?}", blake_g_claim.log_size, start.elapsed());

        }

    }
}

