/// FNV-1a hash function to generate unique eval IDs for components
pub const fn fnv1a_eval_id_gen(tag: &str) -> u32 {
    const FNV_OFFSET_BASIS: u32 = 2166136261;
    const FNV_PRIME: u32 = 16777619;

    let bytes = tag.as_bytes();
    let mut hash = FNV_OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_eval_id_gen() {
        // Test that different tags generate different IDs
        let id1 = fnv1a_eval_id_gen("add_opcode_small");
        let id2 = fnv1a_eval_id_gen("add_ap_opcode");
        assert_ne!(id1, id2);

        // Test that same tag generates same ID
        let id3 = fnv1a_eval_id_gen("add_opcode_small");
        assert_eq!(id1, id3);
    }
}
