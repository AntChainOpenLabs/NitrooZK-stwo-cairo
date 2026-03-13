// Opcodes
pub mod opcodes;
pub use opcodes::*;

// Builtins
pub mod builtins;
pub use builtins::*;

// Blake sub-components
pub mod blake_g_cuda;
pub mod blake_round_cuda;
pub mod blake_round_sigma_cuda;

// Verify bitwise components
pub mod verify_bitwise_xor;
pub use verify_bitwise_xor::{
    vbx_12 as vbx_12_cuda, vbx_4 as vbx_4_cuda, vbx_7 as vbx_7_cuda, vbx_8 as vbx_8_cuda,
    vbx_8_b as vbx_8_b_cuda, vbx_9 as vbx_9_cuda,
};

pub mod triple_xor_32_cuda;
pub mod verify_instruction_cuda;

// Memory components
pub mod memory_address_to_id_cuda;
pub mod memory_id_to_big_cuda;

// Range check components
pub mod range_check;

// Backwards-compatible re-exports for range check modules
pub use range_check::{
    rc_11 as range_check_11_cuda, rc_12 as range_check_12_cuda, rc_18 as range_check_18_cuda,
    rc_20 as range_check_20_cuda, rc_3_3_3_3_3 as range_check_3_3_3_3_3_cuda,
    rc_3_6_6_3 as range_check_3_6_6_3_cuda, rc_4_3 as range_check_4_3_cuda,
    rc_4_4 as range_check_4_4_cuda, rc_4_4_4_4 as range_check_4_4_4_4_cuda,
    rc_6 as range_check_6_cuda, rc_7_2_5 as range_check_7_2_5_cuda, rc_8 as range_check_8_cuda,
    rc_9_9 as range_check_9_9_cuda,
};

// Shared CUDA lookup element helper
pub mod cuda_lookup_helper;

// Other components
pub mod cube_252_cuda;
pub mod partial_ec_mul_cuda;
pub mod partial_ec_mul_wb9_cuda;
pub mod pedersen_aggregator_cuda;
pub mod pedersen_aggregator_wb9_cuda;
pub mod pedersen_cuda;
pub mod pedersen_points_table_cuda;
pub mod pedersen_points_table_wb9_cuda;
pub mod pedersen_wb9_cuda;
pub mod poseidon_3_partial_rounds_chain_cuda;
pub mod poseidon_aggregator_cuda;
pub mod poseidon_aggregator_native_cuda;
pub mod poseidon_cuda;
pub mod poseidon_full_round_chain_cuda;
pub mod range_check_252_width_27_cuda;
