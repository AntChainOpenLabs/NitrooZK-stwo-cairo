//! CUDA opcodes claim generator.
//!
//! This module provides `OpcodesCudaClaimGenerator` which uses CUDA for some opcode trace generation
//! while other opcodes use SIMD mode. It completely replaces `OpcodesClaimGenerator`.

use cairo_air::air::CairoInteractionElements;
use cairo_air::opcodes_air::{OpcodeClaim, OpcodeInteractionClaim};
use stwo::prover::backend::cuda::CudaBackend;
use stwo_cairo_adapter::opcodes::StateTransitions;

use super::blake_context_cuda::BlakeContextCudaClaimGenerator;
use super::components_cuda::{
    add_ap_opcode_cuda, add_opcode_cuda, add_opcode_small_cuda, assert_eq_opcode_cuda,
    assert_eq_opcode_double_deref_cuda, assert_eq_opcode_imm_cuda,
    blake_compress_opcode_cuda, call_opcode_cuda, call_opcode_rel_imm_cuda,
    generic_opcode_cuda, jnz_opcode_cuda, jnz_opcode_taken_cuda,
    jump_opcode_cuda, jump_opcode_double_deref_cuda, jump_opcode_rel_cuda, jump_opcode_rel_imm_cuda,
    mul_opcode_cuda, mul_opcode_small_cuda, qm_31_add_mul_opcode_cuda, ret_opcode_cuda,
    memory_address_to_id_cuda, memory_id_to_big_cuda,
    range_check_11_cuda, range_check_18_cuda, range_check_4_4_4_4_cuda, range_check_7_2_5_cuda,
    verify_instruction_cuda, verify_bitwise_xor_8_cuda,
    blake_round_cuda, triple_xor_32_cuda,
};
use super::range_checks_cuda::RangeChecksCudaClaimGenerator;
use crate::witness::components::{
    memory_address_to_id, memory_id_to_big, verify_bitwise_xor_8, verify_instruction,
};
use crate::witness::utils::TreeBuilder;

pub struct OpcodesCudaClaimGenerator {
    // CUDA opcodes
    add: Vec<add_opcode_cuda::CudaClaimGenerator>,
    add_small: Vec<add_opcode_small_cuda::CudaClaimGenerator>,
    add_ap: Vec<add_ap_opcode_cuda::CudaClaimGenerator>,
    assert_eq: Vec<assert_eq_opcode_cuda::CudaClaimGenerator>,
    assert_eq_imm: Vec<assert_eq_opcode_imm_cuda::CudaClaimGenerator>,
    assert_eq_double_deref: Vec<assert_eq_opcode_double_deref_cuda::CudaClaimGenerator>,
    blake: Vec<blake_compress_opcode_cuda::CudaClaimGenerator>,
    call: Vec<call_opcode_cuda::CudaClaimGenerator>,
    call_rel_imm: Vec<call_opcode_rel_imm_cuda::CudaClaimGenerator>,
    generic: Vec<generic_opcode_cuda::CudaClaimGenerator>,
    jnz: Vec<jnz_opcode_cuda::CudaClaimGenerator>,
    jnz_taken: Vec<jnz_opcode_taken_cuda::CudaClaimGenerator>,
    jump: Vec<jump_opcode_cuda::CudaClaimGenerator>,
    jump_double_deref: Vec<jump_opcode_double_deref_cuda::CudaClaimGenerator>,
    jump_rel: Vec<jump_opcode_rel_cuda::CudaClaimGenerator>,
    jump_rel_imm: Vec<jump_opcode_rel_imm_cuda::CudaClaimGenerator>,
    mul: Vec<mul_opcode_cuda::CudaClaimGenerator>,
    mul_small: Vec<mul_opcode_small_cuda::CudaClaimGenerator>,
    qm31: Vec<qm_31_add_mul_opcode_cuda::CudaClaimGenerator>,
    ret: Vec<ret_opcode_cuda::CudaClaimGenerator>,
}

impl OpcodesCudaClaimGenerator {
    pub fn new(input: StateTransitions) -> Self {
        let mut add = vec![];
        let mut add_small = vec![];
        let mut add_ap = vec![];
        let mut assert_eq = vec![];
        let mut assert_eq_imm = vec![];
        let mut assert_eq_double_deref = vec![];
        let mut blake = vec![];
        let mut call = vec![];
        let mut call_rel_imm = vec![];
        let mut generic = vec![];
        let mut jnz = vec![];
        let mut jnz_taken = vec![];
        let mut jump = vec![];
        let mut jump_double_deref = vec![];
        let mut jump_rel = vec![];
        let mut jump_rel_imm = vec![];
        let mut mul = vec![];
        let mut mul_small = vec![];
        let mut qm31 = vec![];
        let mut ret = vec![];

        // CUDA opcodes
        if !input.casm_states_by_opcode.add_opcode.is_empty() {
            add.push(add_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.add_opcode,
            ));
        }
        if !input.casm_states_by_opcode.add_opcode_small.is_empty() {
            add_small.push(add_opcode_small_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.add_opcode_small,
            ));
        }
        if !input.casm_states_by_opcode.add_ap_opcode.is_empty() {
            add_ap.push(add_ap_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.add_ap_opcode,
            ));
        }
        if !input.casm_states_by_opcode.assert_eq_opcode.is_empty() {
            assert_eq.push(assert_eq_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.assert_eq_opcode,
            ));
        }
        if !input.casm_states_by_opcode.assert_eq_opcode_imm.is_empty() {
            assert_eq_imm.push(assert_eq_opcode_imm_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.assert_eq_opcode_imm,
            ));
        }
        if !input
            .casm_states_by_opcode
            .assert_eq_opcode_double_deref
            .is_empty()
        {
            assert_eq_double_deref.push(
                assert_eq_opcode_double_deref_cuda::CudaClaimGenerator::new(
                    input.casm_states_by_opcode.assert_eq_opcode_double_deref,
                ),
            );
        }

        // CUDA opcodes (remaining)
        if !input.casm_states_by_opcode.blake_compress_opcode.is_empty() {
            blake.push(blake_compress_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.blake_compress_opcode,
            ));
        }
        if !input.casm_states_by_opcode.call_opcode.is_empty() {
            call.push(call_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.call_opcode,
            ));
        }
        if !input.casm_states_by_opcode.call_opcode_rel_imm.is_empty() {
            call_rel_imm.push(call_opcode_rel_imm_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.call_opcode_rel_imm,
            ));
        }
        if !input.casm_states_by_opcode.generic_opcode.is_empty() {
            generic.push(generic_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.generic_opcode,
            ));
        }
        if !input.casm_states_by_opcode.jnz_opcode.is_empty() {
            jnz.push(jnz_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.jnz_opcode,
            ));
        }
        if !input.casm_states_by_opcode.jnz_opcode_taken.is_empty() {
            jnz_taken.push(jnz_opcode_taken_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.jnz_opcode_taken,
            ));
        }
        if !input.casm_states_by_opcode.jump_opcode.is_empty() {
            jump.push(jump_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.jump_opcode,
            ));
        }
        if !input
            .casm_states_by_opcode
            .jump_opcode_double_deref
            .is_empty()
        {
            jump_double_deref.push(jump_opcode_double_deref_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.jump_opcode_double_deref,
            ));
        }
        if !input.casm_states_by_opcode.jump_opcode_rel.is_empty() {
            jump_rel.push(jump_opcode_rel_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.jump_opcode_rel,
            ));
        }
        if !input.casm_states_by_opcode.jump_opcode_rel_imm.is_empty() {
            jump_rel_imm.push(jump_opcode_rel_imm_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.jump_opcode_rel_imm,
            ));
        }
        if !input.casm_states_by_opcode.mul_opcode.is_empty() {
            mul.push(mul_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.mul_opcode,
            ));
        }
        if !input.casm_states_by_opcode.mul_opcode_small.is_empty() {
            mul_small.push(mul_opcode_small_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.mul_opcode_small,
            ));
        }
        if !input.casm_states_by_opcode.qm_31_add_mul_opcode.is_empty() {
            qm31.push(qm_31_add_mul_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.qm_31_add_mul_opcode,
            ));
        }
        if !input.casm_states_by_opcode.ret_opcode.is_empty() {
            ret.push(ret_opcode_cuda::CudaClaimGenerator::new(
                input.casm_states_by_opcode.ret_opcode,
            ));
        }

        Self {
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
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda: &memory_id_to_big_cuda::CudaClaimGenerator,
        range_check_11_cuda: &range_check_11_cuda::CudaClaimGenerator,
        range_check_18_cuda: &range_check_18_cuda::CudaClaimGenerator,
        range_check_4_4_4_4_cuda: &range_check_4_4_4_4_cuda::CudaClaimGenerator,
        range_check_7_2_5_cuda: &range_check_7_2_5_cuda::CudaClaimGenerator,
        verify_instruction_cuda: &verify_instruction_cuda::CudaClaimGenerator,
        verify_bitwise_xor_8_cuda: &verify_bitwise_xor_8_cuda::CudaClaimGenerator,
        blake_round_cuda: &mut blake_round_cuda::CudaClaimGenerator,
        triple_xor_32_cuda: &mut triple_xor_32_cuda::CudaClaimGenerator,
        // SIMD generators for multiplicity tracking
        blake_context_trace_generator: &mut BlakeContextCudaClaimGenerator,
        memory_address_to_id_trace_generator: &memory_address_to_id::ClaimGenerator,
        memory_id_to_value_trace_generator: &memory_id_to_big::ClaimGenerator,
        verify_instruction_trace_generator: &verify_instruction::ClaimGenerator,
        range_checks_trace_generator: &RangeChecksCudaClaimGenerator,
        verify_bitwise_xor_8_trace_generator: &verify_bitwise_xor_8::ClaimGenerator,
    ) -> (OpcodeClaim, OpcodesCudaInteractionClaimGenerator) {
        // ==== CUDA opcodes ====
        let (add_claims, add_interaction_gens): (Vec<_>, Vec<_>) = self
            .add
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (add_small_claims, add_small_interaction_gens): (Vec<_>, Vec<_>) = self
            .add_small
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (add_ap_claims, add_ap_interaction_gens): (Vec<_>, Vec<_>) = self
            .add_ap
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    range_check_11_cuda,
                    range_check_18_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                    &range_checks_trace_generator.rc_11_simd_trace_generator,
                    &range_checks_trace_generator.rc_18_simd_trace_generator,
                )
            })
            .unzip();

        let (assert_eq_claims, assert_eq_interaction_gens): (Vec<_>, Vec<_>) = self
            .assert_eq
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (assert_eq_imm_claims, assert_eq_imm_interaction_gens): (Vec<_>, Vec<_>) = self
            .assert_eq_imm
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (assert_eq_double_deref_claims, assert_eq_double_deref_interaction_gens): (
            Vec<_>,
            Vec<_>,
        ) = self
            .assert_eq_double_deref
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        // ==== CUDA opcodes (remaining) ====
        let (blake_claims, blake_interaction_gens): (Vec<_>, Vec<_>) = self
            .blake
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    blake_round_cuda,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    range_check_7_2_5_cuda,
                    triple_xor_32_cuda,
                    verify_bitwise_xor_8_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                    &mut blake_context_trace_generator.blake_round,
                    &mut blake_context_trace_generator.triple_xor_32,
                    verify_bitwise_xor_8_trace_generator,
                    &range_checks_trace_generator.rc_7_2_5_simd_trace_generator,
                )
            })
            .unzip();

        let (call_claims, call_interaction_gens): (Vec<_>, Vec<_>) = self
            .call
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (call_rel_imm_claims, call_rel_imm_interaction_gens): (Vec<_>, Vec<_>) = self
            .call_rel_imm
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (generic_claims, generic_interaction_gens): (Vec<_>, Vec<_>) = self
            .generic
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                    &range_checks_trace_generator.rc_9_9_trace_generator,
                    &range_checks_trace_generator.rc_9_9_b_trace_generator,
                    &range_checks_trace_generator.rc_9_9_c_trace_generator,
                    &range_checks_trace_generator.rc_9_9_d_trace_generator,
                    &range_checks_trace_generator.rc_9_9_e_trace_generator,
                    &range_checks_trace_generator.rc_9_9_f_trace_generator,
                    &range_checks_trace_generator.rc_9_9_g_trace_generator,
                    &range_checks_trace_generator.rc_9_9_h_trace_generator,
                    &range_checks_trace_generator.rc_19_trace_generator,
                    &range_checks_trace_generator.rc_19_b_trace_generator,
                    &range_checks_trace_generator.rc_19_c_trace_generator,
                    &range_checks_trace_generator.rc_19_d_trace_generator,
                    &range_checks_trace_generator.rc_19_e_trace_generator,
                    &range_checks_trace_generator.rc_19_f_trace_generator,
                    &range_checks_trace_generator.rc_19_g_trace_generator,
                    &range_checks_trace_generator.rc_19_h_trace_generator,
                    &range_checks_trace_generator.rc_18_simd_trace_generator,
                    &range_checks_trace_generator.rc_11_simd_trace_generator,
                )
            })
            .unzip();

        let (jnz_claims, jnz_interaction_gens): (Vec<_>, Vec<_>) = self
            .jnz
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (jnz_taken_claims, jnz_taken_interaction_gens): (Vec<_>, Vec<_>) = self
            .jnz_taken
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (jump_claims, jump_interaction_gens): (Vec<_>, Vec<_>) = self
            .jump
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (jump_double_deref_claims, jump_double_deref_interaction_gens): (Vec<_>, Vec<_>) = self
            .jump_double_deref
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (jump_rel_claims, jump_rel_interaction_gens): (Vec<_>, Vec<_>) = self
            .jump_rel
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (jump_rel_imm_claims, jump_rel_imm_interaction_gens): (Vec<_>, Vec<_>) = self
            .jump_rel_imm
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        let (mul_claims, mul_interaction_gens): (Vec<_>, Vec<_>) = self
            .mul
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                    &range_checks_trace_generator.rc_19_trace_generator,
                    &range_checks_trace_generator.rc_19_b_trace_generator,
                    &range_checks_trace_generator.rc_19_c_trace_generator,
                    &range_checks_trace_generator.rc_19_d_trace_generator,
                    &range_checks_trace_generator.rc_19_e_trace_generator,
                    &range_checks_trace_generator.rc_19_f_trace_generator,
                    &range_checks_trace_generator.rc_19_g_trace_generator,
                    &range_checks_trace_generator.rc_19_h_trace_generator,
                )
            })
            .unzip();

        let (mul_small_claims, mul_small_interaction_gens): (Vec<_>, Vec<_>) = self
            .mul_small
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    range_check_11_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                    &range_checks_trace_generator.rc_11_simd_trace_generator,
                )
            })
            .unzip();

        let (qm31_claims, qm31_interaction_gens): (Vec<_>, Vec<_>) = self
            .qm31
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    range_check_4_4_4_4_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                    &range_checks_trace_generator.rc_4_4_4_4_trace_generator,
                )
            })
            .unzip();

        let (ret_claims, ret_interaction_gens): (Vec<_>, Vec<_>) = self
            .ret
            .into_iter()
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda,
                    memory_id_to_big_cuda,
                    verify_instruction_cuda,
                    memory_address_to_id_trace_generator,
                    memory_id_to_value_trace_generator,
                    verify_instruction_trace_generator,
                )
            })
            .unzip();

        (
            OpcodeClaim {
                add: add_claims,
                add_small: add_small_claims,
                add_ap: add_ap_claims,
                assert_eq: assert_eq_claims,
                assert_eq_imm: assert_eq_imm_claims,
                assert_eq_double_deref: assert_eq_double_deref_claims,
                blake: blake_claims,
                call: call_claims,
                call_rel_imm: call_rel_imm_claims,
                generic: generic_claims,
                jnz: jnz_claims,
                jnz_taken: jnz_taken_claims,
                jump: jump_claims,
                jump_double_deref: jump_double_deref_claims,
                jump_rel: jump_rel_claims,
                jump_rel_imm: jump_rel_imm_claims,
                mul: mul_claims,
                mul_small: mul_small_claims,
                qm31: qm31_claims,
                ret: ret_claims,
            },
            OpcodesCudaInteractionClaimGenerator {
                add: add_interaction_gens,
                add_small: add_small_interaction_gens,
                add_ap: add_ap_interaction_gens,
                assert_eq: assert_eq_interaction_gens,
                assert_eq_imm: assert_eq_imm_interaction_gens,
                assert_eq_double_deref: assert_eq_double_deref_interaction_gens,
                blake: blake_interaction_gens,
                call: call_interaction_gens,
                call_rel_imm: call_rel_imm_interaction_gens,
                generic: generic_interaction_gens,
                jnz: jnz_interaction_gens,
                jnz_taken: jnz_taken_interaction_gens,
                jump: jump_interaction_gens,
                jump_double_deref: jump_double_deref_interaction_gens,
                jump_rel: jump_rel_interaction_gens,
                jump_rel_imm: jump_rel_imm_interaction_gens,
                mul: mul_interaction_gens,
                mul_small: mul_small_interaction_gens,
                qm31: qm31_interaction_gens,
                ret: ret_interaction_gens,
            },
        )
    }
}

pub struct OpcodesCudaInteractionClaimGenerator {
    // CUDA opcodes
    add: Vec<add_opcode_cuda::CudaInteractionClaimGenerator>,
    add_small: Vec<add_opcode_small_cuda::CudaInteractionClaimGenerator>,
    add_ap: Vec<add_ap_opcode_cuda::CudaInteractionClaimGenerator>,
    assert_eq: Vec<assert_eq_opcode_cuda::CudaInteractionClaimGenerator>,
    assert_eq_imm: Vec<assert_eq_opcode_imm_cuda::CudaInteractionClaimGenerator>,
    assert_eq_double_deref: Vec<assert_eq_opcode_double_deref_cuda::CudaInteractionClaimGenerator>,
    blake: Vec<blake_compress_opcode_cuda::CudaInteractionClaimGenerator>,
    call: Vec<call_opcode_cuda::CudaInteractionClaimGenerator>,
    call_rel_imm: Vec<call_opcode_rel_imm_cuda::CudaInteractionClaimGenerator>,
    generic: Vec<generic_opcode_cuda::CudaInteractionClaimGenerator>,
    jnz: Vec<jnz_opcode_cuda::CudaInteractionClaimGenerator>,
    jnz_taken: Vec<jnz_opcode_taken_cuda::CudaInteractionClaimGenerator>,
    jump: Vec<jump_opcode_cuda::CudaInteractionClaimGenerator>,
    jump_double_deref: Vec<jump_opcode_double_deref_cuda::CudaInteractionClaimGenerator>,
    jump_rel: Vec<jump_opcode_rel_cuda::CudaInteractionClaimGenerator>,
    jump_rel_imm: Vec<jump_opcode_rel_imm_cuda::CudaInteractionClaimGenerator>,
    mul: Vec<mul_opcode_cuda::CudaInteractionClaimGenerator>,
    mul_small: Vec<mul_opcode_small_cuda::CudaInteractionClaimGenerator>,
    qm31: Vec<qm_31_add_mul_opcode_cuda::CudaInteractionClaimGenerator>,
    ret: Vec<ret_opcode_cuda::CudaInteractionClaimGenerator>,
}

impl OpcodesCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        interaction_elements: &CairoInteractionElements,
    ) -> OpcodeInteractionClaim {
        // ==== CUDA opcodes ====
        let add_interaction_claims: Vec<_> = self
            .add
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let add_small_interaction_claims: Vec<_> = self
            .add_small
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                    &interaction_elements.verify_instruction,
                )
            })
            .collect();

        let add_ap_interaction_claims: Vec<_> = self
            .add_ap
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.range_checks.rc_18,
                    &interaction_elements.range_checks.rc_11,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let assert_eq_interaction_claims: Vec<_> = self
            .assert_eq
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let assert_eq_imm_interaction_claims: Vec<_> = self
            .assert_eq_imm
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.opcodes,
                    &interaction_elements.verify_instruction,
                )
            })
            .collect();

        let assert_eq_double_deref_interaction_claims: Vec<_> = self
            .assert_eq_double_deref
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                    &interaction_elements.verify_instruction,
                )
            })
            .collect();

        // ==== CUDA opcodes (remaining) ====
        let blake_interaction_claims: Vec<_> = self
            .blake
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.range_checks.rc_7_2_5,
                    &interaction_elements.verify_bitwise_xor_8,
                    &interaction_elements.blake_round,
                    &interaction_elements.triple_xor_32,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let call_interaction_claims: Vec<_> = self
            .call
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let call_rel_imm_interaction_claims: Vec<_> = self
            .call_rel_imm
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let generic_interaction_claims: Vec<_> = self
            .generic
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.range_checks.rc_9_9,
                    &interaction_elements.range_checks.rc_9_9_b,
                    &interaction_elements.range_checks.rc_9_9_c,
                    &interaction_elements.range_checks.rc_9_9_d,
                    &interaction_elements.range_checks.rc_9_9_e,
                    &interaction_elements.range_checks.rc_9_9_f,
                    &interaction_elements.range_checks.rc_9_9_g,
                    &interaction_elements.range_checks.rc_9_9_h,
                    &interaction_elements.range_checks.rc_19_h,
                    &interaction_elements.range_checks.rc_19,
                    &interaction_elements.range_checks.rc_19_b,
                    &interaction_elements.range_checks.rc_19_c,
                    &interaction_elements.range_checks.rc_19_d,
                    &interaction_elements.range_checks.rc_19_e,
                    &interaction_elements.range_checks.rc_19_f,
                    &interaction_elements.range_checks.rc_19_g,
                    &interaction_elements.range_checks.rc_18,
                    &interaction_elements.range_checks.rc_11,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let jnz_interaction_claims: Vec<_> = self
            .jnz
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let jnz_taken_interaction_claims: Vec<_> = self
            .jnz_taken
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let jump_interaction_claims: Vec<_> = self
            .jump
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let jump_double_deref_interaction_claims: Vec<_> = self
            .jump_double_deref
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let jump_rel_interaction_claims: Vec<_> = self
            .jump_rel
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let jump_rel_imm_interaction_claims: Vec<_> = self
            .jump_rel_imm
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let mul_interaction_claims: Vec<_> = self
            .mul
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.range_checks.rc_19_h,
                    &interaction_elements.range_checks.rc_19,
                    &interaction_elements.range_checks.rc_19_b,
                    &interaction_elements.range_checks.rc_19_c,
                    &interaction_elements.range_checks.rc_19_d,
                    &interaction_elements.range_checks.rc_19_e,
                    &interaction_elements.range_checks.rc_19_f,
                    &interaction_elements.range_checks.rc_19_g,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let mul_small_interaction_claims: Vec<_> = self
            .mul_small
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.range_checks.rc_11,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let qm31_interaction_claims: Vec<_> = self
            .qm31
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.range_checks.rc_4_4_4_4,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        let ret_interaction_claims: Vec<_> = self
            .ret
            .into_iter()
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.verify_instruction,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                    &interaction_elements.opcodes,
                )
            })
            .collect();

        OpcodeInteractionClaim {
            add: add_interaction_claims,
            add_small: add_small_interaction_claims,
            add_ap: add_ap_interaction_claims,
            assert_eq: assert_eq_interaction_claims,
            assert_eq_imm: assert_eq_imm_interaction_claims,
            assert_eq_double_deref: assert_eq_double_deref_interaction_claims,
            blake: blake_interaction_claims,
            call: call_interaction_claims,
            call_rel_imm: call_rel_imm_interaction_claims,
            generic: generic_interaction_claims,
            jnz: jnz_interaction_claims,
            jnz_taken: jnz_taken_interaction_claims,
            jump: jump_interaction_claims,
            jump_double_deref: jump_double_deref_interaction_claims,
            jump_rel: jump_rel_interaction_claims,
            jump_rel_imm: jump_rel_imm_interaction_claims,
            mul: mul_interaction_claims,
            mul_small: mul_small_interaction_claims,
            qm31: qm31_interaction_claims,
            ret: ret_interaction_claims,
        }
    }
}
