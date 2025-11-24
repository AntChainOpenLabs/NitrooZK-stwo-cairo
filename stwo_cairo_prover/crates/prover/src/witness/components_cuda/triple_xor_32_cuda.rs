#![allow(unused_parens)]
use cairo_air::components::triple_xor_32::{Claim, InteractionClaim, N_TRACE_COLUMNS};

use crate::witness::components_cuda::verify_bitwise_xor_8_cuda;
use crate::witness::components_cuda::verify_bitwise_xor_8_b_cuda;
use crate::witness::prelude::*;

pub type PackedInputType = [PackedUInt32; 3];
pub type CudaPackedInputType = [Uint32Vec; 3];
use stwo::stwo_cuda::base_field_vec::Uint32Vec;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::prover::backend::cuda::CudaBackend;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::bindings_airs;
use stwo::core::fields::qm31::SecureField;

pub const N_INTERACTION_TRACE_COLUMNS: usize = 5;

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
        verify_bitwise_xor_8_state: &verify_bitwise_xor_8_cuda::CudaClaimGenerator,
        verify_bitwise_xor_8_b_state: &verify_bitwise_xor_8_b_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        assert!(!self.packed_inputs.is_empty());
        let log_size = self.size.ilog2();

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            self.packed_inputs,
        );
        verify_bitwise_xor_8_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8);
        verify_bitwise_xor_8_b_state.add_cuda_inputs(&sub_component_inputs.verify_bitwise_xor_8_b);

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
        inputs.iter().for_each(|input| {
            self.packed_inputs[0].extend(&input[0]);
            self.packed_inputs[1].extend(&input[1]);
            self.packed_inputs[2].extend(&input[2]);
        });
    }
}

struct CudaSubComponentInputs {
    verify_bitwise_xor_8: [verify_bitwise_xor_8_cuda::CudaPackedInputType; 4],
    verify_bitwise_xor_8_b: [verify_bitwise_xor_8_b_cuda::CudaPackedInputType; 4],
}

fn write_trace_cuda(
    inputs:  [Uint32Vec; 3],
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
    let lookup_verify_bitwise_xor_8_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_0);
    let lookup_verify_bitwise_xor_8_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_1);
    let lookup_verify_bitwise_xor_8_2_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_2);
    let lookup_verify_bitwise_xor_8_3_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_3);
    let lookup_verify_bitwise_xor_8_b_0_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_0);
    let lookup_verify_bitwise_xor_8_b_1_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_1);
    let lookup_verify_bitwise_xor_8_b_2_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_2);
    let lookup_verify_bitwise_xor_8_b_3_vec = collect_lookup_ptrs!(lookup_data, verify_bitwise_xor_8_b_3);
    // collect sub_component_inputs pointers
    let sub_component_inputs_verify_bitwise_xor_8_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8);
    let sub_component_inputs_verify_bitwise_xor_8_b_vec = collect_sub_input_ptrs!(sub_component_inputs, verify_bitwise_xor_8_b);

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
        verify_bitwise_xor_8: &relations::VerifyBitwiseXor_8,
        verify_bitwise_xor_8_b: &relations::VerifyBitwiseXor_8_B,
        triple_xor_32: &relations::TripleXor32,
    ) -> InteractionClaim {
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_triple_xor_32_0_vec = collect_lookup_ptrs!(self.lookup_data, triple_xor_32_0);
        let lookup_verify_bitwise_xor_8_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_0);
        let lookup_verify_bitwise_xor_8_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_1);
        let lookup_verify_bitwise_xor_8_2_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_2);
        let lookup_verify_bitwise_xor_8_3_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_3);
        let lookup_verify_bitwise_xor_8_b_0_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_0);
        let lookup_verify_bitwise_xor_8_b_1_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_1);
        let lookup_verify_bitwise_xor_8_b_2_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_2);
        let lookup_verify_bitwise_xor_8_b_3_vec = collect_lookup_ptrs!(self.lookup_data, verify_bitwise_xor_8_b_3);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();


        unsafe {
            let triple_xor_32_ptr = triple_xor_32 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_8_ptr = verify_bitwise_xor_8 as *const _ as *mut std::os::raw::c_void;
            let verify_bitwise_xor_8_b_ptr = verify_bitwise_xor_8_b as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_triple_xor_32_interaction_traces(
                triple_xor_32_ptr,
                verify_bitwise_xor_8_ptr,
                verify_bitwise_xor_8_b_ptr,

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
    use test_log::test;

    use crate::witness::components::triple_xor_32;
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;

    use cairo_air::relations;
    use std::array;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::CudaClaimGenerator;
    use super::BaseFieldVec;
    use crate::witness::prelude::*;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::pcs::TreeVec;
    use itertools::Itertools;
    use stwo_constraint_framework::TraceLocationAllocator;
    use stwo_constraint_framework::FrameworkComponent;
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::triple_xor_32::Eval;
    use stwo_cairo_common::prover_types::simd::PackedUInt32;
    use crate::witness::components::verify_bitwise_xor_8;
    use crate::witness::components::verify_bitwise_xor_8_b;

    #[test]
    fn test_triple_xor_32_cpu_ref() {
        let mut rng = SmallRng::seed_from_u64(0);

        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(|x| PackedUInt32::from_simd(x));

        let mut triple_xor_32_trace_generator = triple_xor_32::ClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b::ClaimGenerator::new();
        let triple_xor_32_relation = relations::TripleXor32::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_8_b_relation = relations::VerifyBitwiseXor_8_B::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        println!("input {:?}", input);
        triple_xor_32_trace_generator.add_packed_inputs(&[input]);

        // Base trace.
        let (triple_xor_32_claim, triple_xor_32_interaction_gen) = triple_xor_32_trace_generator.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_8_b_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        println!("triple_xor_32_claim log size: {:?}", triple_xor_32_claim.log_sizes());

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let triple_xor_32_interaction_claim = triple_xor_32_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_8_b_relation,
            &triple_xor_32_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("triple_xor_32_interaction_claim.claimed_sum: {:?}", triple_xor_32_interaction_claim.claimed_sum);
        // println!("trace: {:?}", trace);


        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("triple_xor_32"),
                claim: triple_xor_32_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
            },
            triple_xor_32_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace)

    }

    #[test]
    fn test_triple_xor_32_trace_gen_by_cpu_and_verify_by_cuda() {
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::prover::backend::Column;

        let mut rng = SmallRng::seed_from_u64(0);

        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(|x|  PackedUInt32::from_simd(x) );

        let mut triple_xor_32_trace_generator = triple_xor_32::ClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8::ClaimGenerator::new();
        let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b::ClaimGenerator::new();
        let triple_xor_32_relation = relations::TripleXor32::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_8_b_relation = relations::VerifyBitwiseXor_8_B::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        println!("input {:?}", input);
        triple_xor_32_trace_generator.add_packed_inputs(&[input]);

        // Base trace.
        let (triple_xor_32_claim, triple_xor_32_interaction_gen) = triple_xor_32_trace_generator.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_8_b_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        println!("triple_xor_32_claim log size: {:?}", triple_xor_32_claim.log_sizes());

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let triple_xor_32_interaction_claim = triple_xor_32_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_8_b_relation,
            &triple_xor_32_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("triple_xor_32_interaction_claim.claimed_sum: {:?}", triple_xor_32_interaction_claim.claimed_sum);
        // println!("trace: {:?}", trace);
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

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("triple_xor_32"),
                claim: triple_xor_32_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
            },
            triple_xor_32_interaction_claim.claimed_sum,
        );

        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
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
                triple_xor_32_claim.log_size as u32,
                triple_xor_32_claim.log_size as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(triple_xor_32_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << triple_xor_32_claim.log_size)),
                false,
                true,  // use_assert_evaluator = true for tests
            );
        }
        println!("triple_xor_32 log_n:{} evaluate time: {:?}", triple_xor_32_claim.log_size, start.elapsed());
    }


    // try to modify crates/prover/src/stwo_cuda/cuda/witness/gen_triple_xor_32_trace.cu
    #[test]
    fn test_triple_xor_32_trace_gen_by_gpu_and_verify_by_cpu() {
        use crate::witness::components_cuda::verify_bitwise_xor_8_cuda;
        use crate::witness::components_cuda::verify_bitwise_xor_8_b_cuda;

        let mut rng = SmallRng::seed_from_u64(123456);
        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(PackedUInt32::from_simd);

        let mut triple_xor_32_cuda = CudaClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b_cuda::CudaClaimGenerator::new();

        let triple_xor_32_relation = relations::TripleXor32::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_8_b_relation = relations::VerifyBitwiseXor_8_B::dummy();

        println!("input {:?}", input);
        triple_xor_32_cuda.add_packed_inputs(&[input]);

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        // Base trace.
        let (triple_xor_32_claim, triple_xor_32_interaction_gen) = triple_xor_32_cuda.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_8_b_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        println!("triple_xor_32_claim: {:?}", triple_xor_32_claim.log_size);
        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let triple_xor_32_interaction_claim = triple_xor_32_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_8_b_relation,
            &triple_xor_32_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        for i in 0..trace[1].len() {
            println!("cuda base traces[{}]: {:?}", i, trace[1][i].to_vec());
        }

        for i in 0..trace[2].len() {
            println!("cuda interaction traces[{}]: {:?}", i, trace[2][i].to_vec());
        }
        println!("triple_xor_32_interaction_claim.claimed_sum: {:?}", triple_xor_32_interaction_claim.claimed_sum);

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
                eval_id: fnv1a_eval_id_gen("triple_xor_32"),
                claim: triple_xor_32_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
            },
            triple_xor_32_interaction_claim.claimed_sum,
        );

        assert_component(&component, &cuda_trace_to_cpu)

        // println!("triple_xor_32_interaction_claim.claimed_sum: {:?}", triple_xor_32_interaction_claim.claimed_sum);

    }

    // try to modify /home/ubuntu/stwo-cairo/stwo/crates/prover/src/stwo_cuda/cuda/witness/gen_triple_xor_32_trace.cu
    // TODO: This test requires access to private fields component.eval and component.info
    // which are not accessible from this crate. Need to refactor or add public accessors.
    // #[cfg(disabled)]  // Disabled: requires access to private fields
    #[test]
    fn triple_xor_32_trace_gen_by_cuda_and_verify_by_cuda() {
        use crate::witness::components_cuda::verify_bitwise_xor_8_cuda;
        use crate::witness::components_cuda::verify_bitwise_xor_8_b_cuda;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::prover::backend::Column;

        let mut rng = SmallRng::seed_from_u64(123456);
        let input = array::from_fn(|_| array::from_fn(|_| rng.gen()))
            .map(std::simd::u32x16::from_array)
            .map(PackedUInt32::from_simd);

        let mut triple_xor_32_cuda = CudaClaimGenerator::new();
        let verify_bitwise_xor_8_trace_generator = verify_bitwise_xor_8_cuda::CudaClaimGenerator::new();
        let verify_bitwise_xor_8_b_trace_generator = verify_bitwise_xor_8_b_cuda::CudaClaimGenerator::new();

        let triple_xor_32_relation = relations::TripleXor32::dummy();
        let verify_bitwise_xor_8_relation = relations::VerifyBitwiseXor_8::dummy();
        let verify_bitwise_xor_8_b_relation = relations::VerifyBitwiseXor_8_B::dummy();

        println!("input {:?}", input);
        triple_xor_32_cuda.add_packed_inputs(&[input]);

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        mock_commitment_scheme.tree_builder().finalize_interaction();
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();

        // Base trace.
        let (triple_xor_32_claim, triple_xor_32_interaction_gen) = triple_xor_32_cuda.write_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_trace_generator,
            &verify_bitwise_xor_8_b_trace_generator,
        );
        mock_tree_builder.finalize_interaction();

        // println!("triple_xor_32_claim: {:?}", triple_xor_32_claim.log_siz);
        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let triple_xor_32_interaction_claim = triple_xor_32_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &verify_bitwise_xor_8_relation,
            &verify_bitwise_xor_8_b_relation,
            &triple_xor_32_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("cuda triple_xor_32_interaction_claim.claimed_sum: {:?}", triple_xor_32_interaction_claim.claimed_sum);

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
                eval_id: fnv1a_eval_id_gen("triple_xor_32"),
                claim: triple_xor_32_claim,
                verify_bitwise_xor_8_lookup_elements: relations::VerifyBitwiseXor_8::dummy(),
                verify_bitwise_xor_8_b_lookup_elements: relations::VerifyBitwiseXor_8_B::dummy(),
                triple_xor_32_lookup_elements: relations::TripleXor32::dummy(),
            },
            triple_xor_32_interaction_claim.claimed_sum,
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
                triple_xor_32_claim.log_size as u32,
                triple_xor_32_claim.log_size as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(triple_xor_32_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << triple_xor_32_claim.log_size)),
                false,
                true,  // use_assert_evaluator = true for tests
            );
        }

        // println!("triple_xor_32_interaction_claim.claimed_sum: {:?}", triple_xor_32_interaction_claim.claimed_sum);
    }

}
