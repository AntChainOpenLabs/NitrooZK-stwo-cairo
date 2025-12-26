// CUDA witness generation for range_check_builtin_bits_96 component
// This handles 96-bit range check operations

#![allow(unused_parens)]
use cairo_air::components::range_check_builtin_bits_96::{Claim, InteractionClaim};

use crate::witness::prelude::*;

use super::super::{memory_address_to_id_cuda, memory_id_to_big_cuda};
use crate::witness::components::{memory_address_to_id, memory_id_to_big, range_check_6};
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use stwo_air_utils::trace::component_trace::CudaComponentTrace;
use stwo::core::fields::m31::BaseField;
pub const N_TRACE_COLUMNS: usize = 12;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 2;

use itertools::Itertools;
use stwo::prover::backend::{Col, Column};
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

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

pub struct CudaClaimGenerator {
    pub log_size: u32,
    pub segment_start: u32,
}

impl CudaClaimGenerator {
    pub fn new(log_size: u32, segment_start: u32) -> Self {
        Self {
            log_size,
            segment_start,
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        // Also pass SIMD generators for multiplicity tracking (needed for final memory traces)
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        range_check_6_simd_state: &range_check_6::ClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        let n_rows = 1usize << log_size;

        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            n_rows,
            log_size,
            self.segment_start,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
        );

        // Add to CUDA generators
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // Add to SIMD generators for final trace generation
        // Copy GPU data to CPU and add to SIMD generators
        // Skip zero values (padding sentinel) and out-of-bounds addresses
        let memory_size = memory_address_to_id_simd_state.memory_size();
        for input_arr in &sub_component_inputs.memory_address_to_id {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for addr in cpu_data.iter().take(n_rows) {
                // Skip zero (padding sentinel) and addresses exceeding memory bounds
                if addr.0 != 0 && addr.0 <= memory_size {
                    memory_address_to_id_simd_state.add_input(addr);
                }
            }
        }
        for input_arr in &sub_component_inputs.memory_id_to_big {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for id in cpu_data.iter().take(n_rows) {
                // Skip zero (padding sentinel) - memory_id_to_big handles bounds internally
                if id.0 != 0 {
                    memory_id_to_big_simd_state.add_input(id);
                }
            }
        }
        // Add range_check_6 inputs
        for input_arr in &sub_component_inputs.range_check_6 {
            let cpu_data: Vec<M31> = input_arr[0].to_vec();
            for val in cpu_data.iter().take(n_rows) {
                range_check_6_simd_state.add_input(&[*val]);
            }
        }

        tree_builder.extend_evals(trace.to_evals());

        (
            Claim { log_size, range_check96_builtin_segment_start: self.segment_start },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

pub struct CudaSubComponentInputs {
    // 1 memory_address_to_id lookup
    pub memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 1],
    // 1 memory_id_to_big lookup
    pub memory_id_to_big: [memory_address_to_id_cuda::CudaPackedInputType; 1],
    // 1 range_check_6 lookup (1 element)
    pub range_check_6: [[BaseFieldVec; 1]; 1],
}

pub struct CudaLookupData {
    // 1 memory_address_to_id lookup (2 elements)
    pub memory_address_to_id_0: [BaseFieldVec; 2],
    // 1 memory_id_to_big lookup (29 elements)
    pub memory_id_to_big_0: [BaseFieldVec; 29],
    // 1 range_check_6 lookup (1 element)
    pub range_check_6_0: [BaseFieldVec; 1],
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    n_rows: usize,
    log_size: u32,
    segment_start: u32,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                // 1 memory_address_to_id lookup (2 elements)
                memory_address_to_id_0: init_lookup_array!(log_size),
                // 1 memory_id_to_big lookup (29 elements)
                memory_id_to_big_0: init_lookup_array!(log_size),
                // 1 range_check_6 lookup (1 element)
                range_check_6_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
                range_check_6: std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << log_size))),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect lookup data pointers
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_range_check_6_0 = collect_lookup_ptrs!(lookup_data, range_check_6_0);

    // Collect sub-component input pointers
    let sub_inputs_memory_address_to_id = collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_inputs_memory_id_to_big = collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);
    let sub_inputs_range_check_6: Vec<_> = sub_component_inputs.range_check_6
        .iter()
        .flat_map(|row| row.iter().map(|x| x.device_ptr))
        .collect();

    // Collect memory_id_to_big transposed_big_values pointers
    let memory_id_to_big_transposed_big_values_vec: Vec<_> = memory_id_to_big_state.transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect();

    unsafe {
        bindings_airs::generate_range_check_builtin_bits_96_traces(
            traces_vec.as_ptr(),
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_range_check_6_0.as_ptr(),
            sub_inputs_memory_address_to_id.as_ptr(),
            sub_inputs_memory_id_to_big.as_ptr(),
            sub_inputs_range_check_6.as_ptr(),
            segment_start,
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,
            n_rows as u32,
            log_size,
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

pub struct CudaInteractionClaimGenerator {
    pub n_rows: usize,
    pub log_size: u32,
    pub lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id: &cairo_air::relations::MemoryAddressToId,
        range_check_6: &cairo_air::relations::RangeCheck_6,
        memory_id_to_big: &cairo_air::relations::MemoryIdToBig,
    ) -> InteractionClaim {
        // Allocate interaction trace columns (2 logical columns = 8 M31 columns)
        let interaction_trace: Vec<BaseFieldVec> = (0..N_INTERACTION_TRACE_COLUMNS * 4)
            .map(|_| unsafe { BaseFieldVec::uninitialized(self.n_rows) })
            .collect();
        let interaction_trace_ptrs: Vec<_> = interaction_trace.iter().map(|c| c.device_ptr).collect();

        // Collect lookup data pointers
        let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_range_check_6_0 = collect_lookup_ptrs!(self.lookup_data, range_check_6_0);

        let mut claimed_sum = [0u32; 4];

        unsafe {
            bindings_airs::generate_range_check_builtin_bits_96_interaction_traces(
                memory_address_to_id as *const _ as *mut std::os::raw::c_void,
                memory_id_to_big as *const _ as *mut std::os::raw::c_void,
                range_check_6 as *const _ as *mut std::os::raw::c_void,
                lookup_memory_address_to_id_0.as_ptr(),
                lookup_memory_id_to_big_0.as_ptr(),
                lookup_range_check_6_0.as_ptr(),
                self.n_rows as u32,
                self.log_size,
                interaction_trace_ptrs.as_ptr(),
                claimed_sum.as_mut_ptr(),
            );
        }

        // Convert interaction trace to CircleEvaluations
        let domain = stwo::core::poly::circle::CanonicCoset::new(self.log_size).circle_domain();
        let evals: Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> = interaction_trace
            .into_iter()
            .map(|col| CircleEvaluation::new(domain, Col::<CudaBackend, BaseField>::from(col)))
            .collect();

        tree_builder.extend_evals(evals);

        // Convert claimed_sum from u32 array to SecureField
        let claimed_sum = SecureField::from_m31_array(std::array::from_fn(|i| {
            M31::from_u32_unchecked(claimed_sum[i])
        }));

        InteractionClaim { claimed_sum }
    }
}

#[cfg(test)]
pub mod tests {
    use test_log::test;
    use dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_adapter::utils::{run_program_and_adapter, ProgramType};
    use stwo::core::fields::m31::M31;
    use stwo_constraint_framework::fnv1a_eval_id_gen;
    use crate::witness::components::{
        memory_address_to_id, memory_id_to_big,
        range_check_6,
        range_check_builtin_bits_96,
    };
    use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
    use super::*;
    use crate::witness::components_cuda::{memory_address_to_id_cuda, memory_id_to_big_cuda};

    #[test]
    fn test_range_check_bits_96_builtin_cpu_ref() {
        use cairo_air::relations;
        use cairo_air::components::range_check_builtin_bits_96::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use crate::debug_tools::assert_constraints::assert_component;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_range_check_bits_96_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for range_check_bits_96 builtin segment
        use stwo_cairo_adapter::builtins::RANGE_CHECK_MEMORY_CELLS;

        let range_check96_segment = input.builtin_segments.range_check_bits_96.as_ref()
            .expect("Expected range_check_bits_96 builtin segment");

        let segment_length = range_check96_segment.stop_ptr - range_check96_segment.begin_addr;
        let n_instances = segment_length / RANGE_CHECK_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = range_check96_segment.begin_addr as u32;

        println!("range_check_bits_96_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_6_state = range_check_6::ClaimGenerator::new();

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_state.get_id(addr);
            memory_address_to_id_state.add_input(&addr);
            memory_id_to_big_state.add_input(&id);
        }

        // Create range_check_bits_96_builtin claim generator
        let range_check96_claim_gen = range_check_builtin_bits_96::ClaimGenerator::new(log_size, segment_start);

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_6_relation = relations::RangeCheck_6::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (range_check96_claim, range_check96_interaction_gen) = range_check96_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &range_check_6_state,
        );

        mock_tree_builder.finalize_interaction();

        println!("range_check_bits_96_builtin_claim log_size: {:?}", range_check96_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check96_interaction_claim = range_check96_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &range_check_6_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("range_check_bits_96_builtin_interaction_claim.claimed_sum: {:?}", range_check96_interaction_claim.claimed_sum);

        // Create component and verify with assert_component
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let range_check96_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("range_check_builtin_bits_96"),
                claim: range_check96_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_6_lookup_elements: relations::RangeCheck_6::dummy(),
            },
            range_check96_interaction_claim.claimed_sum,
        );

        assert_component(&range_check96_component, &trace)
    }

    #[test]
    fn test_range_check_bits_96_builtin_trace_gen_by_cpu_and_verify_by_cuda() {
        use cairo_air::relations;
        use cairo_air::components::range_check_builtin_bits_96::{Component, Eval};
        use stwo::core::fields::m31::{M31, BaseField};
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings::CudaSecureField;
        use stwo::prover::backend::Column;
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_range_check_bits_96_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for range_check_bits_96 builtin segment
        use stwo_cairo_adapter::builtins::RANGE_CHECK_MEMORY_CELLS;

        let range_check96_segment = input.builtin_segments.range_check_bits_96.as_ref()
            .expect("Expected range_check_bits_96 builtin segment");

        let segment_length = range_check96_segment.stop_ptr - range_check96_segment.begin_addr;
        let n_instances = segment_length / RANGE_CHECK_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = range_check96_segment.begin_addr as u32;

        println!("range_check_bits_96_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize sub-component generators
        let memory_address_to_id_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_6_state = range_check_6::ClaimGenerator::new();

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_state.get_id(addr);
            memory_address_to_id_state.add_input(&addr);
            memory_id_to_big_state.add_input(&id);
        }

        // Create range_check_bits_96_builtin claim generator
        let range_check96_claim_gen = range_check_builtin_bits_96::ClaimGenerator::new(log_size, segment_start);

        // Create relations
        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_6_relation = relations::RangeCheck_6::dummy();

        // Create mock commitment scheme
        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace - use testing_preprocessed_tree for GPU compatibility
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (range_check96_claim, range_check96_interaction_gen) = range_check96_claim_gen.write_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_state,
            &memory_id_to_big_state,
            &range_check_6_state,
        );

        mock_tree_builder.finalize_interaction();

        println!("range_check_bits_96_builtin_claim log_size: {:?}", range_check96_claim.log_size);

        // Interaction trace
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check96_interaction_claim = range_check96_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &range_check_6_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("range_check_bits_96_builtin_interaction_claim.claimed_sum: {:?}", range_check96_interaction_claim.claimed_sum);

        // Convert trace to CUDA format
        // trace[0] is preprocessed trace (Seq column)
        // trace[1] is base trace
        // trace[2] is interaction trace
        let trace0_vec: Vec<_> = if !trace[0].is_empty() {
            trace[0].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace1_vec: Vec<_> = if trace.len() > 1 && !trace[1].is_empty() {
            trace[1].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace2_vec: Vec<_> = if trace.len() > 2 && !trace[2].is_empty() {
            trace[2].clone().into_iter()
                .map(|eval| BaseFieldVec::from_vec(eval.to_cpu().to_vec()))
                .collect()
        } else {
            vec![]
        };

        let trace0_evaluations_vec: Vec<_> = trace0_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();
        let trace1_evaluations_vec: Vec<_> = trace1_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();
        let trace2_evaluations_vec: Vec<_> = trace2_vec.iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect();

        // Create component first to get actual constraint count
        let tree_span_provider_temp = &mut TraceLocationAllocator::default();
        let range_check96_component_temp = Component::new(
            tree_span_provider_temp,
            Eval {
                eval_id: fnv1a_eval_id_gen("range_check_builtin_bits_96"),
                claim: range_check96_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_6_lookup_elements: relations::RangeCheck_6::dummy(),
            },
            range_check96_interaction_claim.claimed_sum,
        );

        println!("range_check_bits_96_builtin n_constraints: {}", range_check96_component_temp.info.n_constraints);

        // Create mock CUDA buffers with correct size
        let domain_size = 1 << range_check96_claim.log_size;
        let n_constraints = range_check96_component_temp.info.n_constraints.max(500);

        let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
        let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        let mock_accum_col_columns_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
        let mock_accum_col_columns_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

        // Use the component we already created
        let range_check96_component = range_check96_component_temp;

        // Call CUDA evaluator
        let eval_ptr = &range_check96_component.eval as *const _ as *mut std::os::raw::c_void;
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
                range_check96_claim.log_size as u32,
                range_check96_claim.log_size as u32,
                range_check96_component.info.n_constraints as u32,
                range_check96_component.info.logup_counts.iter().map(|(_, &count)| count).sum::<usize>() as u32,
                eval_ptr,
                CudaSecureField::from(
                    range_check96_interaction_claim.claimed_sum
                        / BaseField::from_u32_unchecked(1 << range_check96_claim.log_size)
                ),
                false, // should_accumulate
                true,  // use_assert_evaluator
            );
        }

        println!("range_check_bits_96_builtin CUDA evaluator test completed successfully!");
    }

    /// Compare CPU and CUDA traces column by column to identify discrepancies.
    #[test]
    fn test_range_check_bits_96_builtin_compare_cpu_vs_cuda_traces() {
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use stwo::prover::backend::Column;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_range_check_bits_96_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for range_check_bits_96 builtin segment
        use stwo_cairo_adapter::builtins::RANGE_CHECK_MEMORY_CELLS;

        let range_check96_segment = input.builtin_segments.range_check_bits_96.as_ref()
            .expect("Expected range_check_bits_96 builtin segment");

        let segment_length = range_check96_segment.stop_ptr - range_check96_segment.begin_addr;
        let n_instances = segment_length / RANGE_CHECK_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = range_check96_segment.begin_addr as u32;

        println!("range_check_bits_96_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // ============ CPU trace generation ============
        let memory_address_to_id_state_cpu = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_state_cpu = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_6_state_cpu = range_check_6::ClaimGenerator::new();

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_state_cpu.get_id(addr);
            memory_address_to_id_state_cpu.add_input(&addr);
            memory_id_to_big_state_cpu.add_input(&id);
        }

        let range_check96_claim_gen_cpu = range_check_builtin_bits_96::ClaimGenerator::new(log_size, segment_start);

        let mut mock_commitment_scheme_cpu = MockCommitmentScheme::default();
        let preprocessed_trace_cpu = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        mock_tree_builder_cpu.extend_evals(preprocessed_trace_cpu.gen_trace());
        mock_tree_builder_cpu.finalize_interaction();

        let mut mock_tree_builder_cpu = mock_commitment_scheme_cpu.tree_builder();
        let (cpu_claim, _cpu_interaction_gen) = range_check96_claim_gen_cpu.write_trace(
            &mut mock_tree_builder_cpu,
            &memory_address_to_id_state_cpu,
            &memory_id_to_big_state_cpu,
            &range_check_6_state_cpu,
        );
        mock_tree_builder_cpu.finalize_interaction();
        let cpu_trace = mock_commitment_scheme_cpu.trace_domain_evaluations();

        // ============ CUDA trace generation ============
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_6_simd_state = range_check_6::ClaimGenerator::new();

        for addr in input.public_memory_addresses.iter().copied().map(M31::from_u32_unchecked) {
            let id = memory_address_to_id_cuda_state.get_id(addr);
            memory_address_to_id_cuda_state.add_cuda_input(&addr);
            memory_id_to_big_cuda_state.add_cuda_input(&id);
        }

        let range_check96_cuda_gen = CudaClaimGenerator::new(log_size, segment_start);

        let mut mock_commitment_scheme_cuda = MockCommitmentScheme::default();
        let preprocessed_trace_cuda = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        mock_tree_builder_cuda.extend_evals(preprocessed_trace_cuda.gen_trace());
        mock_tree_builder_cuda.finalize_interaction();

        let mut mock_tree_builder_cuda = mock_commitment_scheme_cuda.tree_builder();
        let (cuda_claim, _cuda_interaction_gen) = range_check96_cuda_gen.write_trace(
            &mut mock_tree_builder_cuda,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
            &range_check_6_simd_state,
        );
        mock_tree_builder_cuda.finalize_interaction();
        let cuda_trace = mock_commitment_scheme_cuda.trace_domain_evaluations();

        // ============ Compare traces ============
        println!("CPU log_size: {:?}, CUDA log_size: {:?}", cpu_claim.log_size, cuda_claim.log_size);
        assert_eq!(cpu_claim.log_size, cuda_claim.log_size);

        let trace_size = 1usize << cpu_claim.log_size;

        // Compare base trace columns (Tree 1)
        let cpu_base_trace = &cpu_trace[1];
        let cuda_base_trace = &cuda_trace[1];

        println!("CPU base trace columns: {:?}", cpu_base_trace.len());
        println!("CUDA base trace columns: {:?}", cuda_base_trace.len());

        let mut different_cols = Vec::new();
        for col in 0..std::cmp::min(cpu_base_trace.len(), cuda_base_trace.len()) {
            let cpu_col = &cpu_base_trace[col];
            let cuda_col = &cuda_base_trace[col];
            let cpu_vals: Vec<M31> = cpu_col.to_cpu().to_vec();
            let cuda_vals: Vec<M31> = cuda_col.to_cpu().to_vec();

            let mut diff_rows = Vec::new();
            for row in 0..trace_size {
                if cpu_vals[row] != cuda_vals[row] {
                    diff_rows.push((row, cpu_vals[row], cuda_vals[row]));
                }
            }

            if !diff_rows.is_empty() {
                different_cols.push(col);
                println!("\n=== Column {} has {} differences ===", col, diff_rows.len());
                for (row, cpu_val, cuda_val) in diff_rows.iter().take(5) {
                    println!("  Row {:3}: CPU={:10}, CUDA={:10}", row, cpu_val.0, cuda_val.0);
                }
                if diff_rows.len() > 5 {
                    println!("  ... and {} more differences", diff_rows.len() - 5);
                }
            }
        }

        if different_cols.is_empty() {
            println!("\nAll {} base trace columns match!", cpu_base_trace.len());
        } else {
            println!("\n=== SUMMARY: {} columns with differences: {:?} ===", different_cols.len(), different_cols);
        }

        assert!(different_cols.is_empty(), "Found differences in {} columns", different_cols.len());
    }

    /// Generate trace using CUDA and verify with CPU constraint evaluator.
    #[test]
    fn test_range_check_bits_96_builtin_trace_gen_by_cuda_and_verify_by_cpu() {
        use cairo_air::relations;
        use cairo_air::components::range_check_builtin_bits_96::{Component, Eval};
        use stwo_constraint_framework::TraceLocationAllocator;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
        use crate::debug_tools::assert_constraints::assert_component;

        // Load test program
        let compiled_program = get_compiled_cairo_program_path("test_prove_verify_range_check_bits_96_builtin");
        let input = run_program_and_adapter(&compiled_program, ProgramType::Json, None);

        // Check for range_check_bits_96 builtin segment
        use stwo_cairo_adapter::builtins::RANGE_CHECK_MEMORY_CELLS;

        let range_check96_segment = input.builtin_segments.range_check_bits_96.as_ref()
            .expect("Expected range_check_bits_96 builtin segment");

        let segment_length = range_check96_segment.stop_ptr - range_check96_segment.begin_addr;
        let n_instances = segment_length / RANGE_CHECK_MEMORY_CELLS;
        let log_size = n_instances.ilog2();
        let segment_start = range_check96_segment.begin_addr as u32;

        println!("range_check_bits_96_builtin log_size: {}, segment_start: {}", log_size, segment_start);

        // Initialize CUDA generators
        let mut memory_address_to_id_cuda_state = memory_address_to_id_cuda::CudaClaimGenerator::new(&input.memory);
        let memory_id_to_big_cuda_state = memory_id_to_big_cuda::CudaClaimGenerator::new(&input.memory);

        // SIMD generators for multiplicity tracking
        let memory_address_to_id_simd_state = memory_address_to_id::ClaimGenerator::new(&input.memory);
        let memory_id_to_big_simd_state = memory_id_to_big::ClaimGenerator::new(&input.memory);
        let range_check_6_simd_state = range_check_6::ClaimGenerator::new();

        // Yield public memory
        for addr in input
            .public_memory_addresses
            .iter()
            .copied()
            .map(M31::from_u32_unchecked)
        {
            let id = memory_address_to_id_cuda_state.get_id(addr);
            memory_address_to_id_cuda_state.add_cuda_input(&addr);
            memory_id_to_big_cuda_state.add_cuda_input(&id);
        }

        // Create CUDA claim generator
        let range_check96_cuda_gen = CudaClaimGenerator::new(log_size, segment_start);

        let memory_address_to_id_relation = relations::MemoryAddressToId::dummy();
        let memory_id_to_big_relation = relations::MemoryIdToBig::dummy();
        let range_check_6_relation = relations::RangeCheck_6::dummy();

        let mut mock_commitment_scheme = MockCommitmentScheme::default();

        // Preprocessed trace
        let preprocessed_trace = testing_preprocessed_tree(log_size);
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        mock_tree_builder.extend_evals(preprocessed_trace.gen_trace());
        mock_tree_builder.finalize_interaction();

        // Base trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let (range_check96_claim, range_check96_interaction_gen) = range_check96_cuda_gen.write_trace(
            &mut mock_tree_builder,
            &mut memory_address_to_id_cuda_state,
            &memory_id_to_big_cuda_state,
            &memory_address_to_id_simd_state,
            &memory_id_to_big_simd_state,
            &range_check_6_simd_state,
        );
        mock_tree_builder.finalize_interaction();

        println!("range_check_bits_96_builtin_claim log_size: {:?}", range_check96_claim.log_size);

        // Interaction trace (CUDA)
        let mut mock_tree_builder = mock_commitment_scheme.tree_builder();
        let range_check96_interaction_claim = range_check96_interaction_gen.write_interaction_trace(
            &mut mock_tree_builder,
            &memory_address_to_id_relation,
            &range_check_6_relation,
            &memory_id_to_big_relation,
        );
        mock_tree_builder.finalize_interaction();
        let trace = mock_commitment_scheme.trace_domain_evaluations();

        println!("range_check_bits_96_builtin_interaction_claim.claimed_sum: {:?}", range_check96_interaction_claim.claimed_sum);

        // Verify with CPU constraint evaluator
        let tree_span_provider = &mut TraceLocationAllocator::default();
        let range_check96_component = Component::new(
            tree_span_provider,
            Eval {
                eval_id: fnv1a_eval_id_gen("range_check_builtin_bits_96"),
                claim: range_check96_claim.clone(),
                memory_address_to_id_lookup_elements: relations::MemoryAddressToId::dummy(),
                memory_id_to_big_lookup_elements: relations::MemoryIdToBig::dummy(),
                range_check_6_lookup_elements: relations::RangeCheck_6::dummy(),
            },
            range_check96_interaction_claim.claimed_sum,
        );

        assert_component(&range_check96_component, &trace);
        println!("range_check_bits_96_builtin CUDA trace gen + CPU verify test completed successfully!");
    }
}
