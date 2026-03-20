# ⚡ S-two Cairo ⚡

Prove Cairo programs with the blazing-fast [S-Two prover](https://github.com/starkware-libs/stwo), powered by the cryptographic breakthrough of [Circle STARKs](https://eprint.iacr.org/2024/278).

* [Prerequisites](#prerequisites)
* [`scarb-prove`](#scarb-prove)
* [CUDA Accelerated Fork](#cuda-accelerated-fork)

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install/)
- [Scarb](https://docs.swmansion.com/scarb/download.html)
  - The recommended installation method is using [asdf](https://asdf-vm.com/)
  - Make sure to use version 2.10.0 and onwards, and preferably the latest nightly version.

    To use the latest nightly version, run:

    ```
    asdf set -u scarb latest:nightly
    ```

## Installation

This repository now focuses on the prover and verifier crates under `stwo_cairo_prover/` and `stwo_cairo_verifier/`. The former `cairo-prove` CLI has been removed. The equivalent utility is now provided in `proving-utils`: https://github.com/starkware-libs/proving-utils

## `scarb prove`

As of Scarb version 2.10.0, `scarb prove` can be used instead of manually building and running `stwo-cairo`.

However, `scarb prove` is still a work in progress, and using `stwo-cairo` directly is preferable for now.

---

# CUDA Accelerated Fork

This fork adds CUDA GPU acceleration to the stwo-cairo prover.

- [Environment Setup](#environment-setup)
- [Repository Setup](#repository-setup)
- [Build](#build)
- [Test Guide](#test-guide)
- [Performance (RTX 5090)](#performance-rtx-5090)

## Environment Setup

| Dependency | Minimum | Tested |
|------------|---------|--------|
| OS | Ubuntu 22.04 | Ubuntu 22.04.5 LTS |
| CPU | x86\_64 with AVX-512 (for SIMD backend) | AMD EPYC 9355 32-Core (188 GB RAM) |
| GPU | Ada Lovelace (sm\_89+) | RTX 5090 (32 GB) |
| CUDA Toolkit | >= 12.8 | 13.0 (Build cuda\_13.0.r13.0) |
| Rust | nightly | 1.91.0-nightly |
| CMake | >= 3.22 | 3.29.4 |
| Scarb | >= 2.10.0 (optional) | — |

## Repository Setup

1. Clone stwo-cairo.
2. Clone the CUDA-enabled stwo fork and place it at `external/stwo/`:
   ```bash
   mkdir -p external && cd external
   git clone https://github.com/AntChainOpenLabs/NitrooZK-stwo.git
   cd NitrooZK-stwo && git checkout v2.1.1-cuda && cd ..
   mv NitrooZK-stwo stwo
   ```
   The workspace `Cargo.toml` patches stwo crates to `../external/stwo/crates/...`.

## Build

| Step | Command |
|------|---------|
| Prover | `cd stwo_cairo_prover && cargo build --release -p stwo-cairo-prover` |

## Test Guide

> **Important**: All CUDA tests MUST use `--test-threads=1`.
> First run (cold) includes CUDA context init overhead.
> **Warm runs (run 1+) represent true performance** — use `_multi` tests
> with `PROVE_LOOP_COUNT >= 3`, discard run 0.

### Test Matrix

All commands run from `stwo_cairo_prover/`.

| Test | Command | Notes |
|------|---------|-------|
| E2E opcodes (CUDA) | `cargo test --release -p stwo-cairo-prover test_e2e_prove_cuda_all_opcode_components -- --nocapture --test-threads=1` | Smoke test |
| E2E builtins (CUDA) | `cargo test --release -p stwo-cairo-prover test_e2e_prove_cuda_all_builtins -- --nocapture --test-threads=1` | Smoke test |
| Small PIE single | `cargo test --release -p stwo-cairo-prover test_prove_verify_small_pie_cuda_once -- --nocapture --test-threads=1` | Cold run, ~600K steps |
| Small PIE multi | `cargo test --release -p stwo-cairo-prover test_prove_verify_small_pie_cuda_multi -- --nocapture --test-threads=1` | Warm = true perf |
| SIMD baseline single | `cargo test --release -p stwo-cairo-prover test_prove_verify_small_pie_simd_once -- --nocapture --ignored` | CPU comparison |
| SIMD baseline multi | `cargo test --release -p stwo-cairo-prover test_prove_verify_small_pie_simd_multi -- --nocapture --test-threads=1` | CPU comparison |


### Advanced / Manual Tests

| Test | Flag | Notes |
|------|------|-------|
| `test_prove_verify_sn_pie_cuda_multi` | `--ignored` | Large PIE, may OOM |
| `test_gpu_memory_estimator` | `--ignored` | Estimate GPU memory needs |
| `test_prove_verify_sn_pie_simd_mem_profile` | `--ignored` | CPU memory profiling |
| `test_prove_verify_pie10_simd_mem_profile` | `--ignored` | CPU memory profiling (10-transfer PIE) |

### Feature-Gated Tests

| Feature | Gate | What it enables |
|---------|------|-----------------|
| `slow-tests` | `--features slow-tests` | SIMD prove+verify, constraint tests, all builtin tests |
| `nightly` | `--features nightly` | Poseidon e2e with Cairo verifier |

### Test Data

| Directory | Description |
|-----------|-------------|
| `test_prove_verify_all_opcode_components/` | All opcode synthetic input |
| `test_prove_verify_all_builtins/` | All builtin synthetic input |
| `test_prove_verify_{add_mod,bitwise,mul_mod,...}_builtin/` | Per-builtin inputs |
| `test_small_pie/` | Real PIE: 10 transfers + 6 EC ops (1.8 MB zip) |
| `sn_pie/` | Large StarkNet PIE (~130 MB) |
| `test_builtins_segments/` | Builtin segment layout |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVE_LOOP_COUNT` | 20 | Iterations for `_multi` tests |


## Performance (RTX 5090)

### Small PIE (~600K steps) — CUDA vs SIMD

| Metric | SIMD (32-Core 4.4 GHz CPU) | CUDA Cold (run 0) | CUDA Warm (run 1+) | Speedup (warm) |
|--------|-----------|-------------------|---------------------|----------------|
| Proof generation | ~2690 ms | ~883 ms | ~250 ms | **10.8x** |
| Verification | < 10 ms | < 10 ms | < 10 ms | — |
| Peak GPU memory | — | ~6.5 GB | ~6.5 GB | — |

> **Warm runs are the true performance metric.** Cold run includes one-time
> CUDA context initialization and twiddle precomputation.

### Pipeline Breakdown (warm run average, runs 2–4)

| Stage | v1.1.0-cuda | v1.1.1-cuda |
|-------|-------------|-------------|
| Preprocessed trace (gen + interp + commit) | ~8 ms | ~11 ms |
| Base trace (gen + commit) | ~207 ms | ~96 ms |
| Interaction trace (gen + commit) | ~58 ms | ~50 ms |
| prove\_ex (composition + FRI + decommit) | ~106 ms | ~93 ms |
| **Total** | **~430 ms** | **~250 ms** |
