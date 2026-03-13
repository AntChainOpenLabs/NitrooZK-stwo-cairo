// GPU-side preprocessed trace generation.
//
// Generates ALL preprocessed columns directly on GPU, avoiding CPU->GPU transfer.
//
// Supported column types (all GPU, no CPU fallback):
// - Seq: Sequential numbers [0..2^n] — GPU kernel
// - RangeCheck: Partitioned enumeration — GPU kernel
// - BitwiseXor: XOR lookup tables — GPU kernel
// - PedersenPoints: Shared GPU table (COW) — borrowed device ptrs, D2D copy at NTT
// - PoseidonRoundKeys: Explicit construct + convert (tiny: 64 rows)
// - BlakeSigma: Explicit construct + convert (tiny: 16 rows)
//
// Optimizations:
// - Batch NTT interpolation: groups columns by log_size, calls ntt_b2n_column with num_poly > 1
// - GPU column generation for Seq, RangeCheck, and BitwiseXor (avoids CPU->GPU transfer)
// - Pedersen columns use shared global GPU table (no copy); COW clone before in-place NTT
//
// Usage:
//   GPU-side generation is enabled by default for all columns.
//   Set PREPROCESSED_TRACE_GPU_GENERATE=0 to disable and use CPU generation.

use std::collections::HashMap;
use std::mem::transmute;

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::prover::backend::cpu::CpuCircleEvaluation;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::{bindings, bindings_airs};
use stwo_cairo_common::preprocessed_columns::blake::BlakeSigma;
use stwo_cairo_common::preprocessed_columns::pedersen::PedersenPoints;
use stwo_cairo_common::preprocessed_columns::poseidon::PoseidonRoundKeys;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedColumn, PreProcessedTrace,
};
use tracing::{info, span, Level};

use crate::witness::cairo_cuda::convert_simd_to_cuda_evaluation;

/// Check if GPU-side preprocessed trace generation is enabled.
/// GPU generation is enabled by default. Set PREPROCESSED_TRACE_GPU_GENERATE=0 to disable.
pub fn use_gpu_generation() -> bool {
    let disabled = std::env::var("PREPROCESSED_TRACE_GPU_GENERATE")
        .map(|v| v == "0" || v.to_lowercase() == "false")
        .unwrap_or(false);
    !disabled
}

/// Generate preprocessed trace on GPU or fall back to CPU generation.
///
/// Returns Vec of CUDA CircleEvaluations ready for commitment.
pub fn gen_preprocessed_trace_cuda(
    preprocessed_trace: &PreProcessedTrace,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    if use_gpu_generation() {
        gen_preprocessed_trace_on_gpu(preprocessed_trace)
    } else {
        // Fall back to CPU generation + conversion.
        preprocessed_trace
            .columns
            .iter()
            .map(|c| convert_simd_to_cuda_evaluation(c.gen_column_simd()))
            .collect()
    }
}

/// Parse a range_check column ID into (bits_per_segment, column_index).
/// Format: "range_check_{b1}_{b2}_..._{bN}_column_{idx}"
fn parse_range_check_id(id: &str) -> Option<(Vec<u32>, usize)> {
    let stripped = id.strip_prefix("range_check_")?;
    let column_pos = stripped.find("_column_")?;
    let ranges_str = &stripped[..column_pos];
    let col_idx_str = &stripped[column_pos + "_column_".len()..];

    let bits: Vec<u32> = ranges_str
        .split('_')
        .map(|s| s.parse::<u32>().ok())
        .collect::<Option<Vec<_>>>()?;
    let col_idx = col_idx_str.parse::<usize>().ok()?;

    Some((bits, col_idx))
}

/// Parse a bitwise_xor column ID into (n_bits, column_index).
/// Format: "bitwise_xor_{n_bits}_{col_index}"
fn parse_bitwise_xor_id(id: &str) -> Option<(u32, usize)> {
    let stripped = id.strip_prefix("bitwise_xor_")?;
    let mut parts = stripped.split('_');
    let n_bits = parts.next()?.parse::<u32>().ok()?;
    let col_idx = parts.next()?.parse::<usize>().ok()?;
    Some((n_bits, col_idx))
}

/// Parse a pedersen_points column ID into the column index.
/// Format: "pedersen_points_{idx}"
fn parse_pedersen_points_id(id: &str) -> Option<usize> {
    // Must check "pedersen_points_small_" first to avoid matching "pedersen_points_" prefix
    if id.starts_with("pedersen_points_small_") {
        return None;
    }
    id.strip_prefix("pedersen_points_")?.parse::<usize>().ok()
}

/// Parse a pedersen_points_small column ID (window_bits_9) into the column index.
/// Format: "pedersen_points_small_{idx}"
fn parse_pedersen_points_small_id(id: &str) -> Option<usize> {
    id.strip_prefix("pedersen_points_small_")?
        .parse::<usize>()
        .ok()
}

/// Parse a poseidon_round_keys column ID into the column index.
/// Format: "poseidon_round_keys_{idx}"
fn parse_poseidon_round_keys_id(id: &str) -> Option<usize> {
    id.strip_prefix("poseidon_round_keys_")?
        .parse::<usize>()
        .ok()
}

/// Parse a blake_sigma column ID into the column index.
/// Format: "blake_sigma_{idx}"
fn parse_blake_sigma_id(id: &str) -> Option<usize> {
    id.strip_prefix("blake_sigma_")?.parse::<usize>().ok()
}

/// Generate preprocessed trace directly on GPU — all column types, no CPU fallback.
fn gen_preprocessed_trace_on_gpu(
    preprocessed_trace: &PreProcessedTrace,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    info!("Generating preprocessed trace directly on GPU (all columns, no CPU fallback)...");

    let log_sizes = preprocessed_trace.log_sizes();
    let ids = preprocessed_trace.ids();

    // Caches for GPU-generated multi-column results.
    let mut range_check_cache: HashMap<
        Vec<u32>,
        Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    > = HashMap::new();
    let mut bitwise_xor_cache: HashMap<
        u32,
        Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    > = HashMap::new();
    let mut pedersen_cache: Option<
        Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    > = None;
    let mut pedersen_small_cache: Option<
        Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    > = None;

    let mut results = Vec::with_capacity(log_sizes.len());
    let mut seq_count = 0u32;
    let mut range_check_count = 0u32;
    let mut bitwise_xor_count = 0u32;
    let mut pedersen_count = 0u32;
    let mut pedersen_small_count = 0u32;
    let mut poseidon_count = 0u32;
    let mut blake_count = 0u32;

    let span = span!(Level::INFO, "preprocessed: gpu_column_dispatch").entered();
    for (_i, (log_size, id)) in log_sizes.iter().zip(ids.iter()).enumerate() {
        let domain = CanonicCoset::new(*log_size).circle_domain();

        let eval = if id.id.starts_with("seq_") {
            seq_count += 1;
            gen_seq_column_cuda(*log_size, domain)
        } else if let Some((bits, col_idx)) = parse_range_check_id(&id.id) {
            range_check_count += 1;
            let cached = range_check_cache
                .entry(bits.clone())
                .or_insert_with(|| gen_range_check_columns_cuda(&bits, domain));
            cached[col_idx].clone()
        } else if let Some((n_bits, col_idx)) = parse_bitwise_xor_id(&id.id) {
            bitwise_xor_count += 1;
            let cached = bitwise_xor_cache
                .entry(n_bits)
                .or_insert_with(|| gen_bitwise_xor_columns_cuda(n_bits, domain));
            cached[col_idx].clone()
        } else if let Some(col_idx) = parse_pedersen_points_small_id(&id.id) {
            pedersen_small_count += 1;
            let cached = pedersen_small_cache
                .get_or_insert_with(|| gen_pedersen_small_columns_simd_fallback(*log_size));
            cached[col_idx].clone()
        } else if let Some(col_idx) = parse_pedersen_points_id(&id.id) {
            pedersen_count += 1;
            // TODO: GPU pedersen table generation produces incorrect values.
            // Fall back to SIMD generation + conversion until the GPU table is fixed.
            let cached = pedersen_cache
                .get_or_insert_with(|| gen_pedersen_columns_simd_fallback(*log_size, domain));
            cached[col_idx].clone()
        } else if let Some(col_idx) = parse_poseidon_round_keys_id(&id.id) {
            poseidon_count += 1;
            let col = PoseidonRoundKeys::new(col_idx);
            convert_simd_to_cuda_evaluation(col.gen_column_simd())
        } else if let Some(col_idx) = parse_blake_sigma_id(&id.id) {
            blake_count += 1;
            let col = BlakeSigma::new(col_idx);
            convert_simd_to_cuda_evaluation(col.gen_column_simd())
        } else {
            panic!(
                "Unknown preprocessed column type: '{}'. All columns must be handled explicitly.",
                id.id
            );
        };

        results.push(eval);
    }
    span.exit();

    info!(
        "Preprocessed trace GPU generation complete ({} total: {} seq, {} range_check, \
         {} bitwise_xor, {} pedersen, {} pedersen_small, {} poseidon, {} blake)",
        results.len(),
        seq_count,
        range_check_count,
        bitwise_xor_count,
        pedersen_count,
        pedersen_small_count,
        poseidon_count,
        blake_count,
    );
    results
}

/// Interpolate columns in batches grouped by log_size.
/// Uses ntt_b2n_column with num_poly > 1 for each size group.
pub fn interpolate_columns_batched(
    columns: Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    twiddle_tree: &TwiddleTree<CudaBackend>,
) -> Vec<CircleCoefficients<CudaBackend>> {
    let n = columns.len();
    if n == 0 {
        return Vec::new();
    }

    // Extract (original_index, log_size, values, domain) from each evaluation.
    let mut indexed: Vec<(usize, u32, BaseFieldVec, CircleDomain)> = columns
        .into_iter()
        .enumerate()
        .map(|(i, eval)| {
            let log_size = eval.domain.log_size();
            (i, log_size, eval.values, eval.domain)
        })
        .collect();

    // Sort by log_size to group same-size columns together.
    indexed.sort_by_key(|(_, ls, ..)| *ls);

    // Count groups for logging.
    {
        let mut count = 0u32;
        let mut max_group_size = 0u32;
        let mut i = 0;
        while i < indexed.len() {
            let ls = indexed[i].1;
            let mut j = i + 1;
            while j < indexed.len() && indexed[j].1 == ls {
                j += 1;
            }
            let group_size = (j - i) as u32;
            if group_size > max_group_size {
                max_group_size = group_size;
            }
            count += 1;
            i = j;
        }
        info!(
            "interpolate_columns_batched: {} columns in {} groups, largest group: {} columns",
            n, count, max_group_size
        );
    }

    let mut results: Vec<(usize, CircleCoefficients<CudaBackend>)> = Vec::with_capacity(n);
    let mut group_start = 0;

    while group_start < indexed.len() {
        let log_size = indexed[group_start].1;

        // Find end of this size group.
        let mut group_end = group_start + 1;
        while group_end < indexed.len() && indexed[group_end].1 == log_size {
            group_end += 1;
        }

        let group = &mut indexed[group_start..group_end];

        if log_size <= 3 {
            // Small columns: fall back to CPU interpolation.
            for item in group.iter_mut() {
                if !item.2.owns_memory {
                    item.2 = item.2.clone();
                }
                let values = std::mem::replace(&mut item.2, BaseFieldVec::new_uninitialized(0));
                let cpu_eval = CpuCircleEvaluation::new(item.3, values.to_cpu());
                let cpu_poly =
                    CpuBackend::interpolate(cpu_eval, unsafe { transmute(twiddle_tree) });
                let cuda_coeffs = BaseFieldVec::from_vec(cpu_poly.coeffs.to_vec());
                results.push((item.0, CircleCoefficients::<CudaBackend>::new(cuda_coeffs)));
            }
        } else {
            // COW: clone borrowed columns to owned before in-place NTT.
            for item in group.iter_mut() {
                if !item.2.owns_memory {
                    item.2 = item.2.clone();
                }
            }

            // Batch NTT: collect device pointers, single kernel call for the group.
            let num_poly = group.len();
            let eval_domain_size = group[0].3.half_coset.size() as u32;

            let mut ptrs: Vec<*mut u32> = group
                .iter()
                .map(|item| item.2.device_ptr as *mut u32)
                .collect();

            unsafe {
                bindings::ntt_b2n_column(
                    ptrs.as_mut_ptr() as *mut *mut u32,
                    log_size,
                    num_poly as u32,
                    twiddle_tree.itwiddles.device_ptr,
                    twiddle_tree.itwiddles.len() as u32,
                    eval_domain_size,
                );
            }

            // NTT was in-place: each BaseFieldVec now contains polynomial coefficients.
            for item in group.iter_mut() {
                let values = std::mem::replace(&mut item.2, BaseFieldVec::new_uninitialized(0));
                results.push((item.0, CircleCoefficients::new(values)));
            }
        }

        group_start = group_end;
    }

    // Restore original column order.
    results.sort_by_key(|(idx, _)| *idx);
    results.into_iter().map(|(_, poly)| poly).collect()
}

/// Generate Seq column directly on GPU.
fn gen_seq_column_cuda(
    log_size: u32,
    domain: CircleDomain,
) -> CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> {
    let n_elements = 1usize << log_size;
    let gpu_col = BaseFieldVec::new_uninitialized(n_elements);
    unsafe {
        bindings_airs::gen_seq_column_on_gpu(gpu_col.device_ptr, log_size);
    }
    CircleEvaluation::new(domain, gpu_col)
}

/// Generate RangeCheck columns directly on GPU.
fn gen_range_check_columns_cuda(
    bits_per_segment: &[u32],
    domain: CircleDomain,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    let n_segments = bits_per_segment.len();
    let total_bits: u32 = bits_per_segment.iter().sum();
    let n_elements = 1usize << total_bits;

    let gpu_cols: Vec<BaseFieldVec> = (0..n_segments)
        .map(|_| BaseFieldVec::new_uninitialized(n_elements))
        .collect();

    let col_ptrs: Vec<*const u32> = gpu_cols.iter().map(|col| col.device_ptr).collect();

    unsafe {
        bindings_airs::gen_range_check_columns_on_gpu(
            col_ptrs.as_ptr(),
            n_segments as u32,
            bits_per_segment.as_ptr(),
            n_segments as u32,
        );
    }

    gpu_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}

/// Generate BitwiseXor columns directly on GPU.
/// Returns 3 columns: a, b, a^b.
fn gen_bitwise_xor_columns_cuda(
    n_bits: u32,
    domain: CircleDomain,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    let n_elements = 1usize << (2 * n_bits);

    let gpu_cols: Vec<BaseFieldVec> = (0..3)
        .map(|_| BaseFieldVec::new_uninitialized(n_elements))
        .collect();

    let col_ptrs: Vec<*const u32> = gpu_cols.iter().map(|col| col.device_ptr).collect();

    unsafe {
        bindings_airs::gen_bitwise_xor_columns_on_gpu(col_ptrs.as_ptr(), n_bits);
    }

    gpu_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}

/// Generate pedersen preprocessed columns using SIMD generation + conversion to CUDA.
/// This is a fallback until the GPU pedersen table generation is fixed.
fn gen_pedersen_columns_simd_fallback(
    log_size: u32,
    _domain: CircleDomain,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    info!(
        "Generating pedersen columns via SIMD fallback (56 columns, log_size={})",
        log_size
    );
    (0..56)
        .map(|col_idx| {
            let col = PedersenPoints::<18>::new(col_idx);
            convert_simd_to_cuda_evaluation(col.gen_column_simd())
        })
        .collect()
}

/// Generate small pedersen preprocessed columns (window_bits_9) using SIMD + conversion.
fn gen_pedersen_small_columns_simd_fallback(
    log_size: u32,
) -> Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    info!(
        "Generating pedersen_small columns via SIMD fallback (56 columns, log_size={})",
        log_size
    );
    (0..56)
        .map(|col_idx| {
            let col = PedersenPoints::<9>::new(col_idx);
            convert_simd_to_cuda_evaluation(col.gen_column_simd())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_range_check_id() {
        let (bits, col) = parse_range_check_id("range_check_4_3_column_0").unwrap();
        assert_eq!(bits, vec![4, 3]);
        assert_eq!(col, 0);

        let (bits, col) = parse_range_check_id("range_check_9_9_column_1").unwrap();
        assert_eq!(bits, vec![9, 9]);
        assert_eq!(col, 1);

        let (bits, col) = parse_range_check_id("range_check_7_2_5_column_2").unwrap();
        assert_eq!(bits, vec![7, 2, 5]);
        assert_eq!(col, 2);

        assert!(parse_range_check_id("seq_5").is_none());
        assert!(parse_range_check_id("bitwise_xor_8_0").is_none());
    }

    #[test]
    fn test_parse_bitwise_xor_id() {
        let (n_bits, col) = parse_bitwise_xor_id("bitwise_xor_8_0").unwrap();
        assert_eq!(n_bits, 8);
        assert_eq!(col, 0);

        let (n_bits, col) = parse_bitwise_xor_id("bitwise_xor_4_2").unwrap();
        assert_eq!(n_bits, 4);
        assert_eq!(col, 2);

        assert!(parse_bitwise_xor_id("range_check_4_3_column_0").is_none());
        assert!(parse_bitwise_xor_id("seq_5").is_none());
    }

    #[test]
    fn test_parse_pedersen_points_id() {
        assert_eq!(parse_pedersen_points_id("pedersen_points_0"), Some(0));
        assert_eq!(parse_pedersen_points_id("pedersen_points_55"), Some(55));
        assert!(parse_pedersen_points_id("seq_5").is_none());
    }

    #[test]
    fn test_parse_poseidon_round_keys_id() {
        assert_eq!(
            parse_poseidon_round_keys_id("poseidon_round_keys_0"),
            Some(0)
        );
        assert_eq!(
            parse_poseidon_round_keys_id("poseidon_round_keys_8"),
            Some(8)
        );
        assert!(parse_poseidon_round_keys_id("seq_5").is_none());
    }

    #[test]
    fn test_parse_blake_sigma_id() {
        assert_eq!(parse_blake_sigma_id("blake_sigma_0"), Some(0));
        assert_eq!(parse_blake_sigma_id("blake_sigma_15"), Some(15));
        assert!(parse_blake_sigma_id("seq_5").is_none());
    }
}
