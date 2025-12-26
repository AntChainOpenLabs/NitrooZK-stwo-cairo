//! CUDA-accelerated builtins claim generator.
//!
//! This module provides `BuiltinsCudaClaimGenerator` which uses CUDA for builtin trace generation.

use cairo_air::air::CairoInteractionElements;
use cairo_air::builtins_air::{BuiltinsClaim, BuiltinsInteractionClaim};
use stwo::prover::backend::cuda::CudaBackend;
use stwo_cairo_adapter::builtins::{
    BuiltinSegments, ADD_MOD_MEMORY_CELLS, BITWISE_MEMORY_CELLS, MUL_MOD_MEMORY_CELLS,
    PEDERSEN_MEMORY_CELLS, POSEIDON_MEMORY_CELLS, RANGE_CHECK_MEMORY_CELLS,
};

// CUDA builtin implementations
use super::components_cuda::builtins::{
    add_mod_builtin_cuda, bitwise_builtin_cuda, mul_mod_builtin_cuda,
    pedersen_builtin_cuda, poseidon_builtin_cuda,
    range_check_bits_96_builtin_cuda, range_check_bits_128_builtin_cuda,
};
use super::components_cuda::pedersen_cuda::PedersenContextCudaClaimGenerator;
use super::components_cuda::poseidon_cuda::PoseidonContextCudaClaimGenerator;
use super::components_cuda::{memory_address_to_id_cuda, memory_id_to_big_cuda};

// SIMD components for multiplicity tracking
use crate::witness::components::{
    memory_address_to_id, memory_id_to_big,
    range_check_6,
};
use crate::witness::utils::TreeBuilder;

pub struct BuiltinsCudaClaimGenerator {
    add_mod_builtin_trace_generator: Option<add_mod_builtin_cuda::CudaClaimGenerator>,
    bitwise_builtin_trace_generator: Option<bitwise_builtin_cuda::CudaClaimGenerator>,
    mul_mod_builtin_trace_generator: Option<mul_mod_builtin_cuda::CudaClaimGenerator>,
    pedersen_builtin_trace_generator: Option<pedersen_builtin_cuda::CudaClaimGenerator>,
    poseidon_builtin_trace_generator: Option<poseidon_builtin_cuda::CudaClaimGenerator>,
    range_check_96_builtin_trace_generator: Option<range_check_bits_96_builtin_cuda::CudaClaimGenerator>,
    range_check_128_builtin_trace_generator: Option<range_check_bits_128_builtin_cuda::CudaClaimGenerator>,
}

impl BuiltinsCudaClaimGenerator {
    pub fn new(builtin_segments: BuiltinSegments) -> Self {
        let add_mod_builtin_trace_generator = builtin_segments.add_mod.map(|segment| {
            let segment_length = segment.stop_ptr - segment.begin_addr;
            assert!(
                segment_length.is_multiple_of(ADD_MOD_MEMORY_CELLS),
                "add mod segment length is not a multiple of it's cells_per_instance"
            );
            let n_instances = segment_length / ADD_MOD_MEMORY_CELLS;
            assert!(
                n_instances.is_power_of_two(),
                "add mod instances number is not a power of two"
            );
            add_mod_builtin_cuda::CudaClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32)
        });
        let bitwise_builtin_trace_generator = builtin_segments.bitwise.map(|segment| {
            let segment_length = segment.stop_ptr - segment.begin_addr;
            assert!(
                segment_length.is_multiple_of(BITWISE_MEMORY_CELLS),
                "bitwise segment length is not a multiple of it's cells_per_instance"
            );
            let n_instances = segment_length / BITWISE_MEMORY_CELLS;
            assert!(
                n_instances.is_power_of_two(),
                "bitwise instances number is not a power of two"
            );
            bitwise_builtin_cuda::CudaClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32)
        });
        let mul_mod_builtin_trace_generator = builtin_segments.mul_mod.map(|segment| {
            let segment_length = segment.stop_ptr - segment.begin_addr;
            assert!(
                segment_length.is_multiple_of(MUL_MOD_MEMORY_CELLS),
                "mul mod segment length is not a multiple of it's cells_per_instance"
            );
            let n_instances = segment_length / MUL_MOD_MEMORY_CELLS;
            assert!(
                n_instances.is_power_of_two(),
                "mul mod instances number is not a power of two"
            );
            mul_mod_builtin_cuda::CudaClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32)
        });
        let pedersen_builtin_trace_generator = builtin_segments.pedersen.map(|segment| {
            let segment_length = segment.stop_ptr - segment.begin_addr;
            assert!(
                segment_length.is_multiple_of(PEDERSEN_MEMORY_CELLS),
                "pedersen segment length is not a multiple of it's cells_per_instance"
            );
            let n_instances = segment_length / PEDERSEN_MEMORY_CELLS;
            assert!(
                n_instances.is_power_of_two(),
                "pedersen instances number is not a power of two"
            );
            pedersen_builtin_cuda::CudaClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32)
        });
        let poseidon_builtin_trace_generator = builtin_segments.poseidon.map(|segment| {
            let segment_length = segment.stop_ptr - segment.begin_addr;
            assert!(
                segment_length.is_multiple_of(POSEIDON_MEMORY_CELLS),
                "poseidon segment length is not a multiple of it's cells_per_instance"
            );
            let n_instances = segment_length / POSEIDON_MEMORY_CELLS;
            assert!(
                n_instances.is_power_of_two(),
                "poseidon instances number is not a power of two"
            );
            poseidon_builtin_cuda::CudaClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32)
        });
        let range_check_96_builtin_trace_generator =
            builtin_segments.range_check_bits_96.map(|segment| {
                let segment_length = segment.stop_ptr - segment.begin_addr;
                let n_instances = segment_length / RANGE_CHECK_MEMORY_CELLS;
                assert!(
                    n_instances.is_power_of_two(),
                    "range_check_bits_96 instances number is not a power of two"
                );
                range_check_bits_96_builtin_cuda::CudaClaimGenerator::new(
                    n_instances.ilog2(),
                    segment.begin_addr as u32,
                )
            });
        let range_check_128_builtin_trace_generator =
            builtin_segments.range_check_bits_128.map(|segment| {
                let segment_length = segment.stop_ptr - segment.begin_addr;
                let n_instances = segment_length / RANGE_CHECK_MEMORY_CELLS;
                assert!(
                    n_instances.is_power_of_two(),
                    "range_check_bits_128 instances number is not a power of two"
                );
                range_check_bits_128_builtin_cuda::CudaClaimGenerator::new(
                    n_instances.ilog2(),
                    segment.begin_addr as u32,
                )
            });
        Self {
            add_mod_builtin_trace_generator,
            bitwise_builtin_trace_generator,
            mul_mod_builtin_trace_generator,
            pedersen_builtin_trace_generator,
            poseidon_builtin_trace_generator,
            range_check_96_builtin_trace_generator,
            range_check_128_builtin_trace_generator,
        }
    }

    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        // CUDA generators
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        // SIMD generators for multiplicity tracking
        memory_address_to_id_simd_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_simd_state: &memory_id_to_big::ClaimGenerator,
        // pedersen dependencies
        pedersen_context_trace_generator: &mut PedersenContextCudaClaimGenerator,
        // poseidon dependencies (not used in CUDA version - poseidon_builtin_cuda handles internally)
        _poseidon_context_trace_generator: &mut PoseidonContextCudaClaimGenerator,
        // range check dependencies
        range_check_6_trace_generator: &range_check_6::ClaimGenerator,
    ) -> (BuiltinsClaim, BuiltinsCudaInteractionClaimGenerator) {
        let (add_mod_builtin_claim, add_mod_builtin_interaction_gen) = self
            .add_mod_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                )
            })
            .unzip();

        let (bitwise_builtin_claim, bitwise_builtin_interaction_gen) = self
            .bitwise_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                )
            })
            .unzip();

        let (mul_mod_builtin_claim, mul_mod_builtin_interaction_gen) = self
            .mul_mod_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                )
            })
            .unzip();

        let (pedersen_builtin_claim, pedersen_builtin_interaction_gen) = self
            .pedersen_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    &mut pedersen_context_trace_generator.partial_ec_mul_trace_generator,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                )
            })
            .unzip();

        let (poseidon_builtin_claim, poseidon_builtin_interaction_gen) = self
            .poseidon_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                )
            })
            .unzip();

        let (range_check_96_builtin_claim, range_check_96_builtin_interaction_gen) = self
            .range_check_96_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                    range_check_6_trace_generator,
                )
            })
            .unzip();

        let (range_check_128_builtin_claim, range_check_128_builtin_interaction_gen) = self
            .range_check_128_builtin_trace_generator
            .map(|gen| {
                gen.write_trace(
                    tree_builder,
                    memory_address_to_id_cuda_state,
                    memory_id_to_big_cuda_state,
                    memory_address_to_id_simd_state,
                    memory_id_to_big_simd_state,
                )
            })
            .unzip();

        (
            BuiltinsClaim {
                add_mod_builtin: add_mod_builtin_claim,
                bitwise_builtin: bitwise_builtin_claim,
                mul_mod_builtin: mul_mod_builtin_claim,
                pedersen_builtin: pedersen_builtin_claim,
                poseidon_builtin: poseidon_builtin_claim,
                range_check_96_builtin: range_check_96_builtin_claim,
                range_check_128_builtin: range_check_128_builtin_claim,
            },
            BuiltinsCudaInteractionClaimGenerator {
                add_mod_builtin_interaction_gen,
                bitwise_builtin_interaction_gen,
                mul_mod_builtin_interaction_gen,
                pedersen_builtin_interaction_gen,
                poseidon_builtin_interaction_gen,
                range_check_96_builtin_interaction_gen,
                range_check_128_builtin_interaction_gen,
            },
        )
    }
}

pub struct BuiltinsCudaInteractionClaimGenerator {
    add_mod_builtin_interaction_gen: Option<add_mod_builtin_cuda::CudaInteractionClaimGenerator>,
    bitwise_builtin_interaction_gen: Option<bitwise_builtin_cuda::CudaInteractionClaimGenerator>,
    mul_mod_builtin_interaction_gen: Option<mul_mod_builtin_cuda::CudaInteractionClaimGenerator>,
    pedersen_builtin_interaction_gen: Option<pedersen_builtin_cuda::CudaInteractionClaimGenerator>,
    poseidon_builtin_interaction_gen: Option<poseidon_builtin_cuda::CudaInteractionClaimGenerator>,
    range_check_96_builtin_interaction_gen:
        Option<range_check_bits_96_builtin_cuda::CudaInteractionClaimGenerator>,
    range_check_128_builtin_interaction_gen:
        Option<range_check_bits_128_builtin_cuda::CudaInteractionClaimGenerator>,
}

impl BuiltinsCudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        interaction_elements: &CairoInteractionElements,
    ) -> BuiltinsInteractionClaim {
        let add_mod_builtin_interaction_claim =
            self.add_mod_builtin_interaction_gen
                .map(|gen| {
                    gen.write_interaction_trace(
                        tree_builder,
                        &interaction_elements.memory_address_to_id,
                        &interaction_elements.memory_id_to_value,
                    )
                });

        let bitwise_builtin_interaction_claim =
            self.bitwise_builtin_interaction_gen
                .map(|gen| {
                    gen.write_interaction_trace(
                        tree_builder,
                        &interaction_elements.memory_address_to_id,
                        &interaction_elements.memory_id_to_value,
                        &interaction_elements.verify_bitwise_xor_9,
                        &interaction_elements.verify_bitwise_xor_8,
                    )
                });

        let mul_mod_builtin_interaction_claim =
            self.mul_mod_builtin_interaction_gen
                .map(|gen| {
                    gen.write_interaction_trace(
                        tree_builder,
                        &interaction_elements.memory_address_to_id,
                        &interaction_elements.memory_id_to_value,
                        &interaction_elements.range_checks.rc_12,
                        &interaction_elements.range_checks.rc_3_6_6_3,
                        &interaction_elements.range_checks.rc_18,
                    )
                });

        let pedersen_builtin_interaction_claim =
            self.pedersen_builtin_interaction_gen
                .map(|gen| {
                    gen.write_interaction_trace(
                        tree_builder,
                        &interaction_elements.memory_address_to_id,
                        &interaction_elements.memory_id_to_value,
                        &interaction_elements.partial_ec_mul,
                        &interaction_elements.range_checks.rc_5_4,
                        &interaction_elements.range_checks.rc_8,
                    )
                });

        let poseidon_builtin_interaction_claim =
            self.poseidon_builtin_interaction_gen
                .map(|gen| {
                    gen.write_interaction_trace(
                        tree_builder,
                        &interaction_elements.memory_address_to_id,
                        &interaction_elements.memory_id_to_value,
                        &interaction_elements.poseidon_full_round_chain,
                        &interaction_elements.range_check_felt_252_width_27,
                        &interaction_elements.cube_252,
                        &interaction_elements.range_checks.rc_3_3_3_3_3,
                        &interaction_elements.range_checks.rc_4_4_4_4,
                        &interaction_elements.range_checks.rc_4_4,
                        &interaction_elements.poseidon_3_partial_rounds_chain,
                    )
                });

        let range_check_96_builtin_interaction_claim = self
            .range_check_96_builtin_interaction_gen
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.range_checks.rc_6,
                    &interaction_elements.memory_id_to_value,
                )
            });

        let range_check_128_builtin_interaction_claim = self
            .range_check_128_builtin_interaction_gen
            .map(|gen| {
                gen.write_interaction_trace(
                    tree_builder,
                    &interaction_elements.memory_address_to_id,
                    &interaction_elements.memory_id_to_value,
                )
            });

        BuiltinsInteractionClaim {
            add_mod_builtin: add_mod_builtin_interaction_claim,
            bitwise_builtin: bitwise_builtin_interaction_claim,
            mul_mod_builtin: mul_mod_builtin_interaction_claim,
            pedersen_builtin: pedersen_builtin_interaction_claim,
            poseidon_builtin: poseidon_builtin_interaction_claim,
            range_check_96_builtin: range_check_96_builtin_interaction_claim,
            range_check_128_builtin: range_check_128_builtin_interaction_claim,
        }
    }
}
