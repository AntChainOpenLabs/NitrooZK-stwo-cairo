#![allow(unused_parens)]
#![allow(dead_code)]

use cairo_air::components::blake_round_sigma::{Claim, InteractionClaim, LOG_SIZE};
use itertools::Itertools;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

use crate::witness::prelude::*;

pub type InputType = [M31; 1];
pub type PackedInputType = [PackedM31; 1];
pub type CudaPackedInputType = [BaseFieldVec; 1];

use stwo::stwo_cuda::bindings_airs;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::Col;
use stwo::core::fields::qm31::SecureField;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 1;

pub struct CudaClaimGenerator {
    pub mults: Vec<BaseFieldVec>,
}
impl CudaClaimGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            mults: (0..1)
            .map(|_| BaseFieldVec::new_zeroes(1 << LOG_SIZE))
            .collect_vec(),
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let mults = self.mults;

        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let trace = mults.clone()
            .into_iter()
            .map(|col| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, col));
        let lookup_data = CudaLookupData { mults };

        tree_builder.extend_evals(trace);

        (Claim {}, CudaInteractionClaimGenerator { lookup_data })
    }

    pub fn add_packed_inputs(&mut self, packed_inputs: &[PackedInputType]) {
        let cuda_inputs: [BaseFieldVec; 1]= (0..1)
            .map(|i| {
                let elements: Vec<_> = packed_inputs
                    .iter()
                    .flat_map(|input| input[i].to_array())
                    .collect();
                BaseFieldVec::from_vec(unsafe { std::mem::transmute(elements) })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Inputs should have exactly 3 elements");

        let cuda_input_vec = cuda_inputs
            .iter()
            .map(|row| row.device_ptr)
            .collect_vec();
        let mults_vec = self.mults
            .iter()
            .map(|row| row.device_ptr)
            .collect_vec();
        unsafe {
            bindings_airs::blake_round_sigma_mults_init(
                cuda_input_vec.as_ptr(),
                1,
                (1 << LOG_SIZE) as u32,
                mults_vec.as_ptr(),
                1,
                LOG_SIZE,
            );
        }
    }

    pub fn add_cuda_inputs(&self, cuda_inputs: &[CudaPackedInputType]) {
        // println!("cuda_inputs verify_bitwise_xor_4_cuda inputs cols: {}, rows:{}, mults log_size:{}", cuda_inputs.len(), cuda_inputs[0][0].size, LOG_SIZE);
        let inputs_vec = cuda_inputs.iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();

        let mults_vec = self.mults
            .iter()
            .map(|row| row.device_ptr)
            .collect_vec();
        unsafe {
            bindings_airs::blake_round_sigma_mults_init(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                mults_vec.as_ptr(),
                1,
                LOG_SIZE,
            );
        }
    }
}

struct CudaLookupData {
    mults: Vec<BaseFieldVec>,
}

pub struct CudaInteractionClaimGenerator {
    lookup_data: CudaLookupData,
}
impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        blake_round_sigma: &relations::BlakeRoundSigma,
    ) -> InteractionClaim {
        let trace_log_size = self.lookup_data.mults[0].size.ilog2();

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0.. 4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        let lookup_blake_round_sigma_vec = self.lookup_data.mults
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            let blake_round_sigma_ptr = blake_round_sigma as *const _ as *mut std::os::raw::c_void;

            bindings_airs::generate_blake_round_sigma_interaction_traces(
                blake_round_sigma_ptr,

                lookup_blake_round_sigma_vec.as_ptr(),

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

        println!("cuda blake_round_sigma claimed sum: {:?}", claimed_sum);
        InteractionClaim { claimed_sum }
    }
}


#[cfg(test)]
pub mod tests {
    use test_log::test;

    use crate::witness::components::blake_round_sigma;
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;

    use cairo_air::relations;
    use std::array;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};


    use stwo_constraint_framework::TraceLocationAllocator;
    use stwo_constraint_framework::FrameworkComponent;
    use crate::debug_tools::assert_constraints::assert_component;
    use cairo_air::components::blake_round_sigma::Eval;
    use stwo::prover::backend::simd::m31::PackedM31;
    use cairo_air::components::blake_round_sigma::LOG_SIZE;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

    use crate::witness::components_cuda::blake_round_sigma_cuda;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use itertools::Itertools;
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use stwo::prover::backend::Column;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::m31::BaseField;
    use stwo::stwo_cuda::bindings::CudaSecureField;


    #[test]
    fn test_blake_round_sigma_gen_ref() {
        let mut rng = SmallRng::seed_from_u64(0);

        let packed_input = array::from_fn(|_| {
            let input = array::from_fn(|_| rng.gen_range(0..(1 << LOG_SIZE)));
            let simd = std::simd::u32x16::from_array(input);
            unsafe { PackedM31::from_simd_unchecked(simd) }
        });

        let blake_round_sigma_trace_generator = blake_round_sigma::ClaimGenerator::new();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(LOG_SIZE);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        println!("input {:?}", packed_input);
        blake_round_sigma_trace_generator.add_packed_inputs(&[packed_input]);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_sigma_claim, blake_round_sigma_interaction_gen) =
            blake_round_sigma_trace_generator.write_trace(
                &mut mock_tree_builder,
            );
        mock_tree_builder.finalize_interaction();

        println!("blake_round_sigma_claim log size: {:?}", blake_round_sigma_claim.log_sizes());

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_sigma_interaction_claim = blake_round_sigma_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_round_sigma_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        // println!("blake_round_sigma_interaction_claim.claimed_sum: {:?}", blake_round_sigma_interaction_claim.claimed_sum);
        // for i in 0..trace[0].len() {
        //     println!("cuda precompute traces[{}]: {:?}", i, trace[0][i].to_vec());
        // }

        // for i in 0..trace[1].len() {
        //     println!("cuda base traces[{}]: {:?}", i, trace[1][i].to_vec());
        // }

        // for i in 0..trace[2].len() {
        //     println!("cuda interaction traces[{}]: {:?}", i, trace[2][i].to_vec());
        // }


        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round_sigma"),
                claim: blake_round_sigma_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
            },
            blake_round_sigma_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace)

    }

    #[test]
    fn test_blake_round_sigma_trace_gen_by_gpu_and_verify_by_cpu() {
        let mut rng = SmallRng::seed_from_u64(0);

        let packed_input = array::from_fn(|_| {
            let input = array::from_fn(|_| rng.gen_range(0..(1 << LOG_SIZE)));
            let simd = std::simd::u32x16::from_array(input);
            unsafe { PackedM31::from_simd_unchecked(simd) }
        });

        let mut blake_round_sigma_trace_generator = blake_round_sigma_cuda::CudaClaimGenerator::new();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(LOG_SIZE);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        println!("input {:?}", packed_input);
        blake_round_sigma_trace_generator.add_packed_inputs(&[packed_input]);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_sigma_claim, blake_round_sigma_interaction_gen) =
            blake_round_sigma_trace_generator.write_trace(
                &mut mock_tree_builder,
            );
        mock_tree_builder.finalize_interaction();

        println!("blake_round_sigma_claim log size: {:?}", blake_round_sigma_claim.log_sizes());

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_sigma_interaction_claim = blake_round_sigma_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_round_sigma_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();


        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round_sigma"),
                claim: blake_round_sigma_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
            },
            blake_round_sigma_interaction_claim.claimed_sum,
        );

        assert_component(&component, &trace)

    }

    #[test]
    fn test_blake_round_sigma_trace_gen_by_cpu_and_verify_by_cuda() {
        let mut rng = SmallRng::seed_from_u64(0);

        let packed_input = array::from_fn(|_| {
            let input = array::from_fn(|_| rng.gen_range(0..(1 << LOG_SIZE)));
            let simd = std::simd::u32x16::from_array(input);
            unsafe { PackedM31::from_simd_unchecked(simd) }
        });

        let blake_round_sigma_trace_generator = blake_round_sigma::ClaimGenerator::new();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(LOG_SIZE);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        println!("input {:?}", packed_input);
        blake_round_sigma_trace_generator.add_packed_inputs(&[packed_input]);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_sigma_claim, blake_round_sigma_interaction_gen) =
            blake_round_sigma_trace_generator.write_trace(
                &mut mock_tree_builder,
            );
        mock_tree_builder.finalize_interaction();

        println!("blake_round_sigma_claim log size: {:?}", blake_round_sigma_claim.log_sizes());

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_sigma_interaction_claim = blake_round_sigma_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_round_sigma_relation,
        );
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

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round_sigma"),
                claim: blake_round_sigma_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
            },
            blake_round_sigma_interaction_claim.claimed_sum,
        );


        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;

        println!("evaluating constraints on domain, trace0_evaluations_vec.len:{}, trace1_evaluations_vec.len:{}, trace2_evaluations_vec.len:{}", trace0_evaluations_vec.len(), trace1_evaluations_vec.len(), trace2_evaluations_vec.len());
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
                LOG_SIZE as u32,
                LOG_SIZE as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(blake_round_sigma_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << LOG_SIZE)),
                false,
                true,
            );
        }
        println!("blake_round_sigma log_n:{} evaluate time: {:?}", LOG_SIZE, start.elapsed());

    }

    #[test]
    fn test_blake_round_sigma_trace_gen_by_cuda_and_verify_by_cuda() {
        let mut rng = SmallRng::seed_from_u64(111);

        let packed_input = array::from_fn(|_| {
            let input = array::from_fn(|_| rng.gen_range(0..(1 << LOG_SIZE)));
            let simd = std::simd::u32x16::from_array(input);
            unsafe { PackedM31::from_simd_unchecked(simd) }
        });

        let mut blake_round_sigma_trace_generator = blake_round_sigma_cuda::CudaClaimGenerator::new();
        let blake_round_sigma_relation = relations::BlakeRoundSigma::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed.
        let preprocessed_trace = testing_preprocessed_tree(LOG_SIZE);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        println!("input {:?}", packed_input);
        blake_round_sigma_trace_generator.add_packed_inputs(&[packed_input]);

        // Base trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (blake_round_sigma_claim, blake_round_sigma_interaction_gen) =
            blake_round_sigma_trace_generator.write_trace(
                &mut mock_tree_builder,
            );
        mock_tree_builder.finalize_interaction();

        println!("blake_round_sigma_claim log size: {:?}", blake_round_sigma_claim.log_sizes());

        // Interaction trace.
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let blake_round_sigma_interaction_claim = blake_round_sigma_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &blake_round_sigma_relation,
        );
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

        let tree_span_provider = &mut TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("blake_round_sigma"),
                claim: blake_round_sigma_claim,
                blake_round_sigma_lookup_elements: relations::BlakeRoundSigma::dummy(),
            },
            blake_round_sigma_interaction_claim.claimed_sum,
        );


        let mock_random_coeff_powers = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_gpu_denom_inv = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec([M31::from_u32_unchecked(0);(1<<10)].to_vec());
        let eval_ptr = &component.eval as *const _ as *mut std::os::raw::c_void;

        println!("evaluating constraints on domain, trace0_evaluations_vec.len:{}, trace1_evaluations_vec.len:{}, trace2_evaluations_vec.len:{}", trace0_evaluations_vec.len(), trace1_evaluations_vec.len(), trace2_evaluations_vec.len());
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
                LOG_SIZE as u32,
                LOG_SIZE as u32,
                component.info.n_constraints as u32,
                component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(blake_round_sigma_interaction_claim.claimed_sum / BaseField::from_u32_unchecked(1 << LOG_SIZE)),
                false,
                true,
            );
        }
        println!("blake_round_sigma log_n:{} evaluate time: {:?}", LOG_SIZE, start.elapsed());

    }



}
