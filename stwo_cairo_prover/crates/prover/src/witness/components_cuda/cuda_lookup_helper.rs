use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::logup::LookupElements;

// Relation constants for all relations used in CUDA interaction trace kernels.
// These are the M31 hash values that uniquely identify each relation in the AIR.
pub const MEMORY_ADDRESS_TO_ID_RELATION_ID: M31 = M31(1444891767);
pub const MEMORY_ID_TO_BIG_RELATION_ID: M31 = M31(1662111297);
pub const OPCODES_RELATION_ID: M31 = M31(428564188);
pub const VERIFY_INSTRUCTION_RELATION_ID: M31 = M31(1719106205);

// Range check relation constants (also used by non-range-check components as sub-lookups)
pub const RC_4_3_RELATION_ID: M31 = M31(1567323731);
pub const RC_7_2_5_RELATION_ID: M31 = M31(371240602);
pub const RC_11_RELATION_ID: M31 = M31(991608089);
pub const RC_18_RELATION_ID: M31 = M31(1109051422);
pub const RC_4_4_4_4_RELATION_ID: M31 = M31(1027333874);

// rc_9_9 variants (8 relations)
pub const RC_9_9_RELATION_IDS: [M31; 8] = [
    M31(517791011),
    M31(1897792095),
    M31(1881014476),
    M31(1864236857),
    M31(1847459238),
    M31(1830681619),
    M31(1813904000),
    M31(2065568285),
];

// range_check_19 (CUDA naming) / range_check_20 (now architecture) variants (8 relations)
pub const RC_19_RELATION_ID: M31 = M31(1410849886);
pub const RC_19_B_RELATION_ID: M31 = M31(514232941);
pub const RC_19_C_RELATION_ID: M31 = M31(531010560);
pub const RC_19_D_RELATION_ID: M31 = M31(480677703);
pub const RC_19_E_RELATION_ID: M31 = M31(497455322);
pub const RC_19_F_RELATION_ID: M31 = M31(447122465);
pub const RC_19_G_RELATION_ID: M31 = M31(463900084);
pub const RC_19_H_RELATION_ID: M31 = M31(682009131);

// Builtin-specific relation constants
pub const RANGE_CHECK_6_RELATION_ID: M31 = M31(1185356339);
pub const RANGE_CHECK_12_RELATION_ID: M31 = M31(941275232);
pub const RANGE_CHECK_3_6_6_3_RELATION_ID: M31 = M31(1005786011);
pub const VERIFY_BITWISE_XOR_9_RELATION_ID: M31 = M31(95781001);

// Poseidon-related relation constants
pub const CUBE_252_RELATION_ID: M31 = M31(1987997202);
pub const POSEIDON_FULL_ROUND_CHAIN_RELATION_ID: M31 = M31(1480369132);
pub const POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID: M31 = M31(1343313504);
pub const POSEIDON_ROUND_KEYS_RELATION_ID: M31 = M31(1024310512);
pub const RANGE_CHECK_252_WIDTH_27_RELATION_ID: M31 = M31(1090315331);
pub const RANGE_CHECK_3_3_3_3_3_RELATION_ID: M31 = M31(502259093);
pub const RANGE_CHECK_4_4_RELATION_ID: M31 = M31(1651211826);
pub const RANGE_CHECK_5_4_RELATION_ID: M31 = M31(1735099921);
pub const RANGE_CHECK_8_RELATION_ID: M31 = M31(1420243005);
pub const PARTIAL_EC_MUL_RELATION_ID: M31 = M31(1621226978);
pub const PEDERSEN_POINTS_TABLE_RELATION_ID: M31 = M31(1444721856);
pub const RC_18_B_RELATION_ID: M31 = M31(1424798916);

// Blake-related relation constants
pub const BLAKE_G_RELATION_ID: M31 = M31(1139985212);
pub const BLAKE_ROUND_RELATION_ID: M31 = M31(40528774);
pub const BLAKE_ROUND_SIGMA_RELATION_ID: M31 = M31(1805967942);
pub const TRIPLE_XOR_32_RELATION_ID: M31 = M31(990559919);
pub const VERIFY_BITWISE_XOR_4_RELATION_ID: M31 = M31(45448144);
pub const VERIFY_BITWISE_XOR_7_RELATION_ID: M31 = M31(62225763);
pub const VERIFY_BITWISE_XOR_8_RELATION_ID: M31 = M31(112558620);
pub const VERIFY_BITWISE_XOR_8_B_RELATION_ID: M31 = M31(521092554);
pub const VERIFY_BITWISE_XOR_12_RELATION_ID: M31 = M31(648362599);

/// Creates modified lookup elements for a CUDA kernel that uses `LookupElementsBasic<N>`.
///
/// CUDA interaction trace kernels compute: `alpha_powers[0]*val0 + ... - z`
/// But the correct "now" architecture computation is:
///   `alpha_powers[0]*REL_CONST + alpha_powers[1]*val0 + ... - z`
///
/// This function creates a modified `LookupElements<128>` where:
///   - z_modified = z - alpha_powers[0]*REL_CONST
///   - alpha_powers are shifted by 1 (alpha_powers[i] = original[i+1])
///
/// The kernel then computes:
///   `shifted_alpha_powers[0]*val0 + ... - z_modified`
///   = `alpha_powers[1]*val0 + ... - (z - alpha_powers[0]*REL_CONST)`
///   = `alpha_powers[0]*REL_CONST + alpha_powers[1]*val0 + ... - z`  (correct!)
///
/// The returned struct has the same memory layout as `CommonLookupElements` and can be
/// safely cast to `*mut c_void` for passing to any CUDA kernel's lookup element parameter.
pub fn create_modified_lookup_for_cuda(
    lookup_elements: &CommonLookupElements,
    relation_constant: M31,
) -> LookupElements<128> {
    create_modified_lookup_for_cuda_with_offset(lookup_elements, relation_constant, M31(0))
}

/// Like `create_modified_lookup_for_cuda`, but also compensates for a value offset difference.
///
/// When a CUDA kernel stores `val + old_offset` but the AIR expects `val + new_offset`,
/// pass `value_offset_correction = new_offset - old_offset` here.
///
/// The modified z becomes:
///   `z_modified = z - alpha_powers[0]*REL_CONST - alpha_powers[1]*offset_correction`
///
/// Then the kernel computes:
///   `alpha_powers[1]*(val + old_offset) - z_modified`
///   = `alpha_powers[0]*REL_CONST + alpha_powers[1]*(val + old_offset + offset_correction) - z`
///   = `alpha_powers[0]*REL_CONST + alpha_powers[1]*(val + new_offset) - z`  (correct!)
pub fn create_modified_lookup_for_cuda_with_offset(
    lookup_elements: &CommonLookupElements,
    relation_constant: M31,
    value_offset_correction: M31,
) -> LookupElements<128> {
    let common =
        unsafe { &*(lookup_elements as *const CommonLookupElements as *const LookupElements<128>) };
    LookupElements {
        z: common.z
            - common.alpha_powers[0] * SecureField::from(relation_constant)
            - common.alpha_powers[1] * SecureField::from(value_offset_correction),
        alpha: common.alpha,
        alpha_powers: std::array::from_fn(|i| {
            if i < 127 {
                common.alpha_powers[i + 1]
            } else {
                common.alpha_powers[127] * common.alpha
            }
        }),
    }
}
