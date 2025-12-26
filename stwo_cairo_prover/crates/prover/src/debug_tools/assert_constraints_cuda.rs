use std::ops::Deref;

use cairo_air::air::{CairoComponents, CairoInteractionElements};
use cairo_air::builtins_air::BuiltinComponents;
use cairo_air::opcodes_air::OpcodeComponents;
use cairo_air::range_checks_air::RangeChecksComponents;
use itertools::Itertools;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::pcs::TreeVec;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings::CudaSecureField;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

use crate::debug_tools::assert_constraints::assert_component;
use crate::debug_tools::mock_tree_builder::MockCommitmentScheme;
use crate::witness::cairo::CairoClaimGenerator;

/// Asserts that a single component's constraints are satisfied using CUDA evaluation.
///
/// This function:
/// 1. First validates using CPU (assert_component)
/// 2. Then validates using CUDA evaluator
pub fn assert_cuda_component<E: FrameworkEval + Sync>(
    component: &FrameworkComponent<E>,
    trace: &TreeVec<Vec<&Vec<M31>>>,
) {
    // First assert with CPU
    assert_component(component, trace);

    // Extract component trace
    let mut component_trace = trace
        .sub_tree(component.trace_locations())
        .map(|tree| tree.into_iter().cloned().collect_vec());
    component_trace[PREPROCESSED_TRACE_IDX] = component
        .preproccessed_column_indices()
        .iter()
        .map(|idx| trace[PREPROCESSED_TRACE_IDX][*idx])
        .collect();

    let log_size = component.log_size();
    let domain_size = 1 << log_size;

    // Convert trace to CUDA format
    let trace0_vec: Vec<BaseFieldVec> = if !component_trace[0].is_empty() {
        component_trace[0]
            .iter()
            .map(|col| BaseFieldVec::from_vec(col.to_vec()))
            .collect()
    } else {
        vec![]
    };

    let trace1_vec: Vec<BaseFieldVec> = if component_trace.len() > 1 && !component_trace[1].is_empty() {
        component_trace[1]
            .iter()
            .map(|col| BaseFieldVec::from_vec(col.to_vec()))
            .collect()
    } else {
        vec![]
    };

    let trace2_vec: Vec<BaseFieldVec> = if component_trace.len() > 2 && !component_trace[2].is_empty() {
        component_trace[2]
            .iter()
            .map(|col| BaseFieldVec::from_vec(col.to_vec()))
            .collect()
    } else {
        vec![]
    };

    let trace0_ptrs: Vec<_> = trace0_vec.iter().map(|v| v.device_ptr).collect();
    let trace1_ptrs: Vec<_> = trace1_vec.iter().map(|v| v.device_ptr).collect();
    let trace2_ptrs: Vec<_> = trace2_vec.iter().map(|v| v.device_ptr).collect();

    // Create mock CUDA buffers
    let n_constraints = component.info.n_constraints.max(1);
    let mock_random_coeff_powers = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); n_constraints]);
    let mock_gpu_denom_inv = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
    let mock_accum_col_0 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
    let mock_accum_col_1 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
    let mock_accum_col_2 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);
    let mock_accum_col_3 = BaseFieldVec::from_vec(vec![M31::from_u32_unchecked(0); domain_size]);

    let eval_ptr = component.deref() as *const E as *mut std::os::raw::c_void;
    let logup_count: usize = component.info.logup_counts.iter().map(|(_, &count)| count).sum();

    let claimed_sum_normalized = component.claimed_sum()
        / BaseField::from_u32_unchecked(1 << log_size);

    unsafe {
        stwo::stwo_cuda::bindings::evaluate_constraint_quotients_on_domain(
            mock_accum_col_0.device_ptr,
            mock_accum_col_1.device_ptr,
            mock_accum_col_2.device_ptr,
            mock_accum_col_3.device_ptr,
            trace0_ptrs.as_ptr(),
            trace0_ptrs.len() as u32,
            trace1_ptrs.as_ptr(),
            trace1_ptrs.len() as u32,
            trace2_ptrs.as_ptr(),
            trace2_ptrs.len() as u32,
            mock_random_coeff_powers.device_ptr,
            mock_gpu_denom_inv.device_ptr,
            log_size,
            log_size,
            n_constraints as u32,
            logup_count as u32,
            eval_ptr,
            CudaSecureField::from(claimed_sum_normalized),
            false, // should_accumulate
            true,  // use_assert_evaluator
        );
    }
}

fn assert_cuda_many<E: FrameworkEval + Sync>(
    components: &[FrameworkComponent<E>],
    trace: &TreeVec<Vec<&Vec<M31>>>,
) {
    components.iter().for_each(|x| assert_cuda_component(x, trace));
}

/// Asserts all Cairo AIR constraints are satisfied using CUDA evaluation.
fn assert_cairo_cuda_components(trace: TreeVec<Vec<&Vec<M31>>>, cairo_components: &CairoComponents) {
    let CairoComponents {
        opcodes,
        verify_instruction,
        blake_context,
        builtins,
        pedersen_context,
        poseidon_context,
        memory_address_to_id,
        memory_id_to_value,
        range_checks,
        verify_bitwise_xor_4,
        verify_bitwise_xor_7,
        verify_bitwise_xor_8,
        verify_bitwise_xor_8_b,
        verify_bitwise_xor_9,
    } = cairo_components;

    let OpcodeComponents {
        add,
        add_small,
        add_ap,
        assert_eq,
        assert_eq_imm,
        assert_eq_double_deref,
        blake,
        call,
        call_rel_imm,
        generic,
        jnz,
        jnz_taken,
        jump,
        jump_double_deref,
        jump_rel,
        jump_rel_imm,
        mul,
        mul_small,
        qm31,
        ret,
    } = opcodes;

    let RangeChecksComponents {
        rc_6,
        rc_8,
        rc_11,
        rc_12,
        rc_18,
        rc_18_b,
        rc_19,
        rc_19_b,
        rc_19_c,
        rc_19_d,
        rc_19_e,
        rc_19_f,
        rc_19_g,
        rc_19_h,
        rc_4_3,
        rc_4_4,
        rc_5_4,
        rc_9_9,
        rc_9_9_b,
        rc_9_9_c,
        rc_9_9_d,
        rc_9_9_e,
        rc_9_9_f,
        rc_9_9_g,
        rc_9_9_h,
        rc_7_2_5,
        rc_3_6_6_3,
        rc_4_4_4_4,
        rc_3_3_3_3_3,
    } = range_checks;

    // Opcodes
    println!("++++++Testing opcodes with CUDA...");
    assert_cuda_many(add, &trace);
    assert_cuda_many(add_small, &trace);
    assert_cuda_many(add_ap, &trace);
    assert_cuda_many(assert_eq, &trace);
    assert_cuda_many(assert_eq_imm, &trace);
    assert_cuda_many(assert_eq_double_deref, &trace);
    assert_cuda_many(blake, &trace);
    assert_cuda_many(call, &trace);
    assert_cuda_many(call_rel_imm, &trace);
    assert_cuda_many(generic, &trace);
    assert_cuda_many(jnz, &trace);
    assert_cuda_many(jnz_taken, &trace);
    assert_cuda_many(jump, &trace);
    assert_cuda_many(jump_double_deref, &trace);
    assert_cuda_many(jump_rel, &trace);
    assert_cuda_many(jump_rel_imm, &trace);
    assert_cuda_many(mul, &trace);
    assert_cuda_many(mul_small, &trace);
    assert_cuda_many(qm31, &trace);
    assert_cuda_many(ret, &trace);

    // Verify instruction
    println!("++++++Testing verify_instruction with CUDA...");
    assert_cuda_component(verify_instruction, &trace);

    // Range checks
    println!("++++++Testing range checks with CUDA...");
    assert_cuda_component(rc_6, &trace);
    assert_cuda_component(rc_8, &trace);
    assert_cuda_component(rc_11, &trace);
    assert_cuda_component(rc_12, &trace);
    assert_cuda_component(rc_18, &trace);
    assert_cuda_component(rc_18_b, &trace);
    assert_cuda_component(rc_19, &trace);
    assert_cuda_component(rc_19_b, &trace);
    assert_cuda_component(rc_19_c, &trace);
    assert_cuda_component(rc_19_d, &trace);
    assert_cuda_component(rc_19_e, &trace);
    assert_cuda_component(rc_19_f, &trace);
    assert_cuda_component(rc_19_g, &trace);
    assert_cuda_component(rc_19_h, &trace);
    assert_cuda_component(rc_4_3, &trace);
    assert_cuda_component(rc_4_4, &trace);
    assert_cuda_component(rc_5_4, &trace);
    assert_cuda_component(rc_9_9, &trace);
    assert_cuda_component(rc_9_9_b, &trace);
    assert_cuda_component(rc_9_9_c, &trace);
    assert_cuda_component(rc_9_9_d, &trace);
    assert_cuda_component(rc_9_9_e, &trace);
    assert_cuda_component(rc_9_9_f, &trace);
    assert_cuda_component(rc_9_9_g, &trace);
    assert_cuda_component(rc_9_9_h, &trace);
    assert_cuda_component(rc_7_2_5, &trace);
    assert_cuda_component(rc_3_6_6_3, &trace);
    assert_cuda_component(rc_4_4_4_4, &trace);
    assert_cuda_component(rc_3_3_3_3_3, &trace);

    // Verify bitwise xor
    println!("++++++Testing verify_bitwise_xor with CUDA...");
    assert_cuda_component(verify_bitwise_xor_4, &trace);
    assert_cuda_component(verify_bitwise_xor_7, &trace);
    assert_cuda_component(verify_bitwise_xor_8, &trace);
    assert_cuda_component(verify_bitwise_xor_8_b, &trace);
    assert_cuda_component(verify_bitwise_xor_9, &trace);

    // Memory components
    println!("++++++Testing memory components with CUDA...");
    assert_cuda_component(memory_address_to_id, &trace);
    for component in &memory_id_to_value.0 {
        assert_cuda_component(component, &trace);
    }
    assert_cuda_component(&memory_id_to_value.1, &trace);

    // Blake context
    if let Some(cairo_air::blake::air::Components {
        blake_round,
        blake_g,
        blake_sigma,
        triple_xor_32,
        verify_bitwise_xor_12,
    }) = &blake_context.components
    {
        println!("++++++Testing blake components with CUDA...");
        assert_cuda_component(blake_round, &trace);
        assert_cuda_component(blake_g, &trace);
        assert_cuda_component(blake_sigma, &trace);
        assert_cuda_component(triple_xor_32, &trace);
        assert_cuda_component(verify_bitwise_xor_12, &trace);
    }

    // Builtins
    println!("++++++Testing builtins with CUDA...");
    let BuiltinComponents {
        add_mod_builtin,
        bitwise_builtin,
        pedersen_builtin,
        poseidon_builtin,
        mul_mod_builtin,
        range_check_96_builtin,
        range_check_128_builtin,
    } = builtins;

    if let Some(add_mod) = add_mod_builtin {
        assert_cuda_component(add_mod, &trace);
    }
    if let Some(mul_mod) = mul_mod_builtin {
        assert_cuda_component(mul_mod, &trace);
    }
    if let Some(bitwise) = bitwise_builtin {
        assert_cuda_component(bitwise, &trace);
    }
    if let Some(pedersen) = pedersen_builtin {
        assert_cuda_component(pedersen, &trace);
    }
    if let Some(poseidon) = poseidon_builtin {
        assert_cuda_component(poseidon, &trace);
    }
    if let Some(rc_96) = range_check_96_builtin {
        assert_cuda_component(rc_96, &trace);
    }
    if let Some(rc_128) = range_check_128_builtin {
        assert_cuda_component(rc_128, &trace);
    }

    // Pedersen context
    if let Some(cairo_air::pedersen::air::Components {
        partial_ec_mul,
        pedersen_points_table,
    }) = &pedersen_context.components
    {
        println!("++++++Testing pedersen context components with CUDA...");
        // Test simpler component first
        assert_cuda_component(pedersen_points_table, &trace);
        // Test partial_ec_mul - includes EC point addition and PartialEcMul relations
        assert_cuda_component(partial_ec_mul, &trace);
    }

    // Poseidon context
    if let Some(cairo_air::poseidon::air::Components {
        poseidon_3_partial_rounds_chain,
        poseidon_full_round_chain,
        cube_252,
        poseidon_round_keys,
        range_check_felt_252_width_27,
    }) = &poseidon_context.components
    {
        println!("++++++Testing poseidon context components with CUDA...");
        assert_cuda_component(poseidon_round_keys, &trace);
        assert_cuda_component(range_check_felt_252_width_27, &trace);
        assert_cuda_component(poseidon_full_round_chain, &trace);
        assert_cuda_component(poseidon_3_partial_rounds_chain, &trace);
        assert_cuda_component(cube_252, &trace);
    }
}

/// Main entry point to assert all Cairo constraints using CUDA evaluation.
pub fn assert_cairo_cuda_constraints(input: ProverInput, preprocessed_trace: PreProcessedTrace) {
    let mut commitment_scheme = MockCommitmentScheme::default();

    // Preprocessed trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(preprocessed_trace.gen_trace());
    tree_builder.finalize_interaction();

    // Base trace.
    let cairo_claim_generator = CairoClaimGenerator::new(input);
    let mut tree_builder = commitment_scheme.tree_builder();
    let (claim, interaction_generator) = cairo_claim_generator.write_trace(&mut tree_builder);
    tree_builder.finalize_interaction();

    // Interaction trace.
    let mut dummy_channel = Blake2sChannel::default();
    let interaction_elements = CairoInteractionElements::draw(&mut dummy_channel);
    let mut tree_builder = commitment_scheme.tree_builder();
    let interaction_claim =
        interaction_generator.write_interaction_trace(&mut tree_builder, &interaction_elements);
    tree_builder.finalize_interaction();

    let components = CairoComponents::new(
        &claim,
        &interaction_elements,
        &interaction_claim,
        &preprocessed_trace.ids(),
    );

    assert_cairo_cuda_components(commitment_scheme.trace_domain_evaluations(), &components);
}
