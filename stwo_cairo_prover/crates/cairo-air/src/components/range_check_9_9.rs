// AIR version 54d95c0d
use crate::components::prelude::*;

pub const N_TRACE_COLUMNS: usize = 1;
pub const LOG_SIZE: u32 = 18;
pub const RELATION_USES_PER_ROW: [RelationUse; 0] = [];

#[repr(C)]
pub struct Eval {
    pub eval_id: u32,
    pub claim: Claim,
    pub range_check_9_9_lookup_elements: relations::RangeCheck_9_9,
}

#[derive(Copy, Clone, Serialize, Deserialize, CairoSerialize, CairoDeserialize)]
#[repr(C)]
pub struct Claim {
    pub log_size: u32,
}
impl Claim {
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let trace_log_sizes = vec![self.log_size; N_TRACE_COLUMNS];
        let interaction_log_sizes = vec![self.log_size; SECURE_EXTENSION_DEGREE];
        TreeVec::new(vec![vec![], trace_log_sizes, interaction_log_sizes])
    }

    pub fn mix_into(&self, _channel: &mut impl Channel) {}
}

#[derive(Copy, Clone, Serialize, Deserialize, CairoSerialize, CairoDeserialize)]
pub struct InteractionClaim {
    pub claimed_sum: SecureField,
}
impl InteractionClaim {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.claimed_sum]);
    }
}

pub type Component = FrameworkComponent<Eval>;

impl FrameworkEval for Eval {
    fn log_size(&self) -> u32 {
        LOG_SIZE
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }

    #[allow(unused_parens)]
    #[allow(clippy::double_parens)]
    #[allow(non_snake_case)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let rangecheck_9_9_0 = eval.get_preprocessed_column((RangeCheck::new([9, 9], 0)).id());
        let rangecheck_9_9_1 = eval.get_preprocessed_column((RangeCheck::new([9, 9], 1)).id());
        let multiplicity = eval.next_trace_mask();

        eval.add_to_relation(RelationEntry::new(
            &self.range_check_9_9_lookup_elements,
            -E::EF::from(multiplicity),
            &[rangecheck_9_9_0.clone(), rangecheck_9_9_1.clone()],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo::core::fields::qm31::QM31;
    use stwo_constraint_framework::expr::ExprEvaluator;
    use stwo_constraint_framework::{fnv1a_eval_id_gen, Relation};

    use super::*;
    use crate::components::constraints_regression_test_values::RANGE_CHECK_9_9;

    #[test]
    fn range_check_9_9_constraints_regression() {
        let mut rng = SmallRng::seed_from_u64(0);
        let eval = Eval {
            eval_id: fnv1a_eval_id_gen("range_check_9_9"),
            claim: Claim { log_size: LOG_SIZE },
            range_check_9_9_lookup_elements: relations::RangeCheck_9_9::dummy(),
        };
        let expr_eval = eval.evaluate(ExprEvaluator::new());
        let assignment = expr_eval.random_assignment();

        let mut sum = QM31::zero();
        for c in expr_eval.constraints {
            sum += c.assign(&assignment) * rng.gen::<QM31>();
        }

        assert_eq!(sum, RANGE_CHECK_9_9);
    }

    #[test]
    fn range_check_9_9_cuda_constraints() {
        // This test validates the CUDA kernel implementation for range_check_9_9.
        // It generates test trace data, calls the CUDA kernel, and verifies the results
        // match CPU-computed reference quotients.

        use itertools::Itertools;
        use stwo::core::channel::Blake2sChannel;
        use stwo::core::fields::m31::BaseField;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::backend::cuda::CudaBackend;
        use stwo::prover::backend::{Col, Column};
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;
        use stwo_constraint_framework::{
            assert_constraints_on_polys_cuda, fnv1a_eval_id_gen, FrameworkEval,
            LogupTraceGenerator,
        };

        const TEST_LOG_SIZE: u32 = 10; // 2^10 = 1024 rows for testing

        println!("Testing CUDA range_check_9_9 constraints for log_size={}", TEST_LOG_SIZE);

        // Generate preprocessed trace (2 columns)
        let domain_size = 1 << TEST_LOG_SIZE;
        let domain = CanonicCoset::new(TEST_LOG_SIZE).circle_domain();

        let preprocessed_0_cpu: Vec<BaseField> = (0..domain_size)
            .map(|i| BaseField::from_u32_unchecked((i % 512) as u32))
            .collect();
        let preprocessed_1_cpu: Vec<BaseField> = (0..domain_size)
            .map(|i| BaseField::from_u32_unchecked(((i / 512) % 512) as u32))
            .collect();

        let trace0 = vec![
            CircleEvaluation::new(
                domain,
                Col::<CudaBackend, BaseField>::from_iter(preprocessed_0_cpu.clone()),
            ),
            CircleEvaluation::new(
                domain,
                Col::<CudaBackend, BaseField>::from_iter(preprocessed_1_cpu.clone()),
            ),
        ];

        // Generate trace column (multiplicity)
        let multiplicity_cpu: Vec<BaseField> = (0..domain_size)
            .map(|i| BaseField::from_u32_unchecked(((i % 100) + 1) as u32))
            .collect();

        let trace1 = vec![CircleEvaluation::new(
            domain,
            Col::<CudaBackend, BaseField>::from_iter(multiplicity_cpu.clone()),
        )];

        // Draw lookup elements from channel (not dummy) to match prover behavior
        let mut channel = Blake2sChannel::default();
        let lookup_elements = relations::RangeCheck_9_9::draw(&mut channel);

        // Generate proper interaction trace on CPU first, then transfer to CUDA
        use stwo::prover::backend::simd::SimdBackend;
        use stwo::prover::backend::simd::column::BaseColumn;
        use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
        use stwo::prover::backend::simd::qm31::PackedSecureField;
        use num_traits::One;

        let cpu_trace0: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = vec![
            CircleEvaluation::new(domain, BaseColumn::from_cpu(preprocessed_0_cpu)),
            CircleEvaluation::new(domain, BaseColumn::from_cpu(preprocessed_1_cpu)),
        ];
        let cpu_trace1: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = vec![
            CircleEvaluation::new(domain, BaseColumn::from_cpu(multiplicity_cpu))
        ];

        // Generate CPU interaction trace properly
        let mut logup_gen = LogupTraceGenerator::new(TEST_LOG_SIZE);
        let mut col_gen = logup_gen.new_col();

        for vec_row in 0..(1 << (TEST_LOG_SIZE - LOG_N_LANES)) {
            let v0_packed: PackedBaseField = cpu_trace0[0].values.data[vec_row];
            let v1_packed: PackedBaseField = cpu_trace0[1].values.data[vec_row];
            let mult_packed: PackedBaseField = cpu_trace1[0].values.data[vec_row];

            let denom: PackedSecureField = lookup_elements.combine(&[v0_packed, v1_packed]);
            let numer: PackedSecureField = PackedSecureField::one() * mult_packed;

            col_gen.write_frac(vec_row, -numer, denom);
        }
        col_gen.finalize_col();

        let (cpu_trace2, claimed_sum) = logup_gen.finalize_last();

        // Transfer interaction trace to CUDA
        let trace2 = cpu_trace2
            .into_iter()
            .map(|cpu_col| {
                CircleEvaluation::new(
                    domain,
                    Col::<CudaBackend, BaseField>::from_iter(cpu_col.values.to_cpu()),
                )
            })
            .collect_vec();

        // Build trace tree
        let cuda_traces = TreeVec::new(vec![trace0, trace1, trace2]);
        let cuda_trace_polys = cuda_traces
            .map(|trace| {
                trace.into_iter().map(|c: CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>| c.interpolate()).collect_vec()
            });

        // Create CUDA-compatible eval structure using the native Eval type
        let range_check_eval = Eval {
            eval_id: fnv1a_eval_id_gen("range_check_9_9"),
            claim: Claim { log_size: TEST_LOG_SIZE },
            range_check_9_9_lookup_elements: lookup_elements.clone(),
        };

        // Get number of constraints
        // range_check_9_9 has 1 logup constraint (from finalize_logup_in_pairs)
        let n_constraints = 1;
        let logup_counts = 1; // One logup entry

        // Verify constraints using CUDA kernel with CPU reference verification
        let eval_log_size = range_check_eval.max_constraint_log_degree_bound();

        assert_constraints_on_polys_cuda(
            &cuda_trace_polys,
            CanonicCoset::new(TEST_LOG_SIZE),
            eval_log_size,
            &range_check_eval,
            n_constraints,
            claimed_sum,
            logup_counts,
        );

        println!("✓ CUDA range_check_9_9 constraints verified successfully");
        println!("  Test log_size: {}", TEST_LOG_SIZE);
        println!("  Domain size: {}", domain_size);
        println!("  Constraints: {}", n_constraints);
        println!("  Logup entries: {}", logup_counts);
        println!("  Claimed sum: {:?}", claimed_sum);
    }
}
