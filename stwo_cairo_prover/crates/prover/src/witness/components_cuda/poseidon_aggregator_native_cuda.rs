//! Native CUDA witness generation for poseidon_aggregator (342 columns).
//!
//! Replaces the SIMD hybrid approach in `poseidon_aggregator_cuda.rs`.
//! The base trace (342 columns) and all sub-component routing happens entirely on GPU.
//! The interaction trace (14 logup columns) also runs on GPU.
//!
//! SIMD reference: components/poseidon_aggregator.rs
//!
//! Sub-component routing:
//!   GPU-direct: poseidon_full_round_chain (8 feeds), poseidon_3_partial_rounds_chain (27 feeds),
//!               cube_252 (2 feeds), range_check_252_width_27 (2 feeds)
//!   GPU-native: memory_id_to_big (6 ID feeds via add_cuda_inputs)
//!   CPU merge:  range_check_3_3_3_3_3, range_check_4_4_4_4, range_check_4_4 (mult merge)

use cairo_air::components::poseidon_aggregator::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

use crate::witness::components_cuda::poseidon_cuda::PoseidonContextCudaClaimGenerator;
use crate::witness::components_cuda::{
    memory_id_to_big_cuda, poseidon_3_partial_rounds_chain_cuda, poseidon_full_round_chain_cuda,
};
use crate::witness::range_checks_cuda::RangeChecksCudaClaimGenerator;
use crate::witness::utils::TreeBuilder;

/// Native CUDA claim generator for poseidon_aggregator.
///
/// Takes 6 ID arrays directly from poseidon_builtin_split (already on GPU),
/// avoiding the GPU→CPU→GPU roundtrip through DashMap.
/// All rows have multiplicity 1 (including builtin-padded duplicates).
pub struct CudaClaimGenerator {
    /// 6 ID arrays already on GPU (from poseidon_builtin_split)
    gpu_ids: [BaseFieldVec; 6],
    /// Log2 of the trace size
    log_size: u32,
}

impl CudaClaimGenerator {
    /// Create from GPU ID arrays produced by poseidon_builtin_split.
    ///
    /// The 6 arrays stay on GPU — no download, sort, or DashMap needed.
    /// Every row gets multiplicity 1 (the builtin already pads to power-of-2).
    pub fn from_gpu_ids(gpu_ids: [BaseFieldVec; 6], log_size: u32) -> Self {
        Self { gpu_ids, log_size }
    }

    /// Write the base trace (342 columns) using native CUDA kernel.
    ///
    /// Routes sub-component inputs:
    /// - memory_id_to_big: 6 ID feeds (GPU-native)
    /// - poseidon_full_round_chain: 8 feeds (GPU-direct via lookup data)
    /// - poseidon_3_partial_rounds_chain: 27 feeds (GPU-direct via lookup data)
    /// - cube_252, rc_252w27: via poseidon_context_cuda
    /// - range_check_3_3_3_3_3, rc_4_4_4_4, rc_4_4: CPU merge multiplicities
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        poseidon_context_cuda: &mut Option<PoseidonContextCudaClaimGenerator>,
        memory_id_to_big_cuda: &mut memory_id_to_big_cuda::CudaClaimGenerator,
        range_checks_trace_generator: &mut RangeChecksCudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let log_size = self.log_size;
        let trace_size = 1usize << log_size;

        // 1. Input data is already on GPU — no upload needed
        let gpu_ids = self.gpu_ids;
        // All rows have multiplicity 1 (builtin pads by repeating instances)
        let gpu_mults = BaseFieldVec::from_vec(vec![M31(1); trace_size]);

        // 2. Allocate 342 trace columns
        let trace_cols: Vec<BaseFieldVec> = (0..342)
            .map(|_| BaseFieldVec::new_uninitialized(trace_size))
            .collect();

        // 3. Allocate lookup data arrays
        // 6 memory_id_to_big (29 elements each)
        let lk_mem: [[BaseFieldVec; 29]; 6] =
            std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size)));
        // 2 range_check_3_3_3_3_3 (5 elements each)
        let lk_rc_3_3_3_3_3: [[BaseFieldVec; 5]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size)));
        // 6 range_check_4_4_4_4 (4 elements each)
        let lk_rc_4_4_4_4: [[BaseFieldVec; 4]; 6] =
            std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size)));
        // 3 range_check_4_4 (2 elements each)
        let lk_rc_4_4: [[BaseFieldVec; 2]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size)));
        // 10 poseidon_full_round_chain (32 elements each)
        // pfrc 0-7: sub-component feeds (rounds 0,1,2,3,31,32,33,34)
        // pfrc 8-9: IT boundary entries (round 4, round 35)
        let lk_pfrc: [[BaseFieldVec; 32]; 10] =
            std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size)));
        // 28 poseidon_3_partial_rounds_chain (42 elements each)
        // p3prc 0-26: sub-component feeds (input to each of 27 groups)
        // p3prc 27: IT boundary (round=31, chain exit)
        let lk_p3prc: [[BaseFieldVec; 42]; 28] =
            std::array::from_fn(|_| std::array::from_fn(|_| BaseFieldVec::new_uninitialized(trace_size)));

        // 4. Collect device pointers
        let traces_ptrs: Vec<*const u32> = trace_cols.iter().map(|c| c.device_ptr).collect();
        let lk_mem_ptrs: [Vec<*const u32>; 6] =
            std::array::from_fn(|i| lk_mem[i].iter().map(|c| c.device_ptr).collect());
        let lk_rc_3_3_3_3_3_ptrs: [Vec<*const u32>; 2] =
            std::array::from_fn(|i| lk_rc_3_3_3_3_3[i].iter().map(|c| c.device_ptr).collect());
        let lk_rc_4_4_4_4_ptrs: [Vec<*const u32>; 6] =
            std::array::from_fn(|i| lk_rc_4_4_4_4[i].iter().map(|c| c.device_ptr).collect());
        let lk_rc_4_4_ptrs: [Vec<*const u32>; 3] =
            std::array::from_fn(|i| lk_rc_4_4[i].iter().map(|c| c.device_ptr).collect());
        let lk_pfrc_ptrs: [Vec<*const u32>; 10] =
            std::array::from_fn(|i| lk_pfrc[i].iter().map(|c| c.device_ptr).collect());
        let lk_p3prc_ptrs: [Vec<*const u32>; 28] =
            std::array::from_fn(|i| lk_p3prc[i].iter().map(|c| c.device_ptr).collect());

        // Memory table pointers
        let big_values_ptrs: Vec<*const u32> = memory_id_to_big_cuda
            .transposed_big_values
            .iter()
            .map(|c| c.device_ptr)
            .collect();

        // 5. Call CUDA base trace kernel
        unsafe {
            bindings_airs::gen_poseidon_aggregator_trace(
                traces_ptrs.as_ptr() as *const u32,
                log_size,
                // 6 input ID arrays
                gpu_ids[0].device_ptr,
                gpu_ids[1].device_ptr,
                gpu_ids[2].device_ptr,
                gpu_ids[3].device_ptr,
                gpu_ids[4].device_ptr,
                gpu_ids[5].device_ptr,
                gpu_mults.device_ptr,
                // Memory tables
                big_values_ptrs.as_ptr() as *const *const u32,
                memory_id_to_big_cuda.small_values.device_ptr,
                // 6 memory_id_to_big lookup data
                lk_mem_ptrs[0].as_ptr() as *const u32,
                lk_mem_ptrs[1].as_ptr() as *const u32,
                lk_mem_ptrs[2].as_ptr() as *const u32,
                lk_mem_ptrs[3].as_ptr() as *const u32,
                lk_mem_ptrs[4].as_ptr() as *const u32,
                lk_mem_ptrs[5].as_ptr() as *const u32,
                // 2 range_check_3_3_3_3_3
                lk_rc_3_3_3_3_3_ptrs[0].as_ptr() as *const u32,
                lk_rc_3_3_3_3_3_ptrs[1].as_ptr() as *const u32,
                // 6 range_check_4_4_4_4
                lk_rc_4_4_4_4_ptrs[0].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[1].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[2].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[3].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[4].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[5].as_ptr() as *const u32,
                // 3 range_check_4_4
                lk_rc_4_4_ptrs[0].as_ptr() as *const u32,
                lk_rc_4_4_ptrs[1].as_ptr() as *const u32,
                lk_rc_4_4_ptrs[2].as_ptr() as *const u32,
                // 10 poseidon_full_round_chain (0-7 sub-component, 8-9 IT boundary)
                lk_pfrc_ptrs[0].as_ptr() as *const u32,
                lk_pfrc_ptrs[1].as_ptr() as *const u32,
                lk_pfrc_ptrs[2].as_ptr() as *const u32,
                lk_pfrc_ptrs[3].as_ptr() as *const u32,
                lk_pfrc_ptrs[4].as_ptr() as *const u32,
                lk_pfrc_ptrs[5].as_ptr() as *const u32,
                lk_pfrc_ptrs[6].as_ptr() as *const u32,
                lk_pfrc_ptrs[7].as_ptr() as *const u32,
                lk_pfrc_ptrs[8].as_ptr() as *const u32,
                lk_pfrc_ptrs[9].as_ptr() as *const u32,
                // 27 poseidon_3_partial_rounds_chain
                lk_p3prc_ptrs[0].as_ptr() as *const u32,
                lk_p3prc_ptrs[1].as_ptr() as *const u32,
                lk_p3prc_ptrs[2].as_ptr() as *const u32,
                lk_p3prc_ptrs[3].as_ptr() as *const u32,
                lk_p3prc_ptrs[4].as_ptr() as *const u32,
                lk_p3prc_ptrs[5].as_ptr() as *const u32,
                lk_p3prc_ptrs[6].as_ptr() as *const u32,
                lk_p3prc_ptrs[7].as_ptr() as *const u32,
                lk_p3prc_ptrs[8].as_ptr() as *const u32,
                lk_p3prc_ptrs[9].as_ptr() as *const u32,
                lk_p3prc_ptrs[10].as_ptr() as *const u32,
                lk_p3prc_ptrs[11].as_ptr() as *const u32,
                lk_p3prc_ptrs[12].as_ptr() as *const u32,
                lk_p3prc_ptrs[13].as_ptr() as *const u32,
                lk_p3prc_ptrs[14].as_ptr() as *const u32,
                lk_p3prc_ptrs[15].as_ptr() as *const u32,
                lk_p3prc_ptrs[16].as_ptr() as *const u32,
                lk_p3prc_ptrs[17].as_ptr() as *const u32,
                lk_p3prc_ptrs[18].as_ptr() as *const u32,
                lk_p3prc_ptrs[19].as_ptr() as *const u32,
                lk_p3prc_ptrs[20].as_ptr() as *const u32,
                lk_p3prc_ptrs[21].as_ptr() as *const u32,
                lk_p3prc_ptrs[22].as_ptr() as *const u32,
                lk_p3prc_ptrs[23].as_ptr() as *const u32,
                lk_p3prc_ptrs[24].as_ptr() as *const u32,
                lk_p3prc_ptrs[25].as_ptr() as *const u32,
                lk_p3prc_ptrs[26].as_ptr() as *const u32,
                lk_p3prc_ptrs[27].as_ptr() as *const u32,
            );
        }

        // Save base trace columns for interaction trace.
        // IMPORTANT: Must clone AFTER the CUDA kernel has populated the columns.
        // Cloning before the kernel would copy uninitialized GPU data.
        let base_trace_keep_alive: Vec<BaseFieldVec> =
            trace_cols.iter().map(|c| c.clone()).collect();
        let base_trace_ptrs: Vec<*const u32> =
            base_trace_keep_alive.iter().map(|c| c.device_ptr).collect();

        // 6. Route sub-components

        // 6a. memory_id_to_big: feed 6 IDs (trace cols 0-5)
        // The IDs are in the trace columns, which are already on GPU.
        // Clone them since add_cuda_inputs takes references.
        let mem_inputs: Vec<[BaseFieldVec; 1]> = (0..6)
            .map(|i| [trace_cols[i].clone()])
            .collect();
        memory_id_to_big_cuda.add_cuda_inputs(&mem_inputs);

        // 6b. Poseidon context: full_round_chain, 3_partial_rounds_chain, cube_252, rc_252w27
        if let Some(ref mut ctx) = poseidon_context_cuda {
            // poseidon_full_round_chain: 8 sub-component feeds from pfrc[0..7]
            // (32 elements = chain_idx + round + 3×10 state Width27 limbs)
            let pfrc_inputs = build_pfrc_input(&lk_pfrc[..8]);
            ctx.poseidon_full_round_chain_cuda
                .add_cuda_inputs(&pfrc_inputs);

            // poseidon_3_partial_rounds_chain: 27 sub-component feeds from p3prc[0..26]
            let p3prc_inputs = build_p3prc_input(&lk_p3prc[..27]);
            ctx.poseidon_3_partial_rounds_chain_cuda
                .add_cuda_inputs(&p3prc_inputs);

            // cube_252: 2 feeds — input Width27 limbs from trace columns
            // First cube: input = cols 143-152 (state[2] Width27), output = cols 153-162
            let cube_input_0: [BaseFieldVec; 10] =
                std::array::from_fn(|i| trace_cols[143 + i].clone());
            ctx.cube_252_cuda.add_cuda_inputs(&cube_input_0);
            // Second cube: input = cols 163-172 (LC_partial result), output = cols 184-193
            let cube_input_1: [BaseFieldVec; 10] =
                std::array::from_fn(|i| trace_cols[163 + i].clone());
            ctx.cube_252_cuda.add_cuda_inputs(&cube_input_1);

            // range_check_252_width_27: 2 feeds — Width27 limbs from trace columns
            // First: LC0 output = cols 123-132
            let rc252_input_0: [BaseFieldVec; 10] =
                std::array::from_fn(|i| trace_cols[123 + i].clone());
            ctx.range_check_252_width_27_cuda
                .add_cuda_inputs(&rc252_input_0);
            // Second: LC1 output = cols 133-142
            let rc252_input_1: [BaseFieldVec; 10] =
                std::array::from_fn(|i| trace_cols[133 + i].clone());
            ctx.range_check_252_width_27_cuda
                .add_cuda_inputs(&rc252_input_1);
        }

        // 6c. Range check multiplicities: download + merge on CPU
        // rc_3_3_3_3_3: 2 lookups × 5 carry values → use the lookup data carries
        for lk_data in &lk_rc_3_3_3_3_3 {
            let carry_arrays: Vec<BaseFieldVec> = lk_data.iter().map(|c| c.clone()).collect();
            let input: [BaseFieldVec; 5] = carry_arrays.try_into().unwrap();
            range_checks_trace_generator
                .rc_3_3_3_3_3_trace_generator
                .add_cuda_inputs(&[input]);
        }
        for lk_data in &lk_rc_4_4_4_4 {
            let carry_arrays: Vec<BaseFieldVec> = lk_data.iter().map(|c| c.clone()).collect();
            let input: [BaseFieldVec; 4] = carry_arrays.try_into().unwrap();
            range_checks_trace_generator
                .rc_4_4_4_4_trace_generator
                .add_cuda_inputs(&[input]);
        }
        for lk_data in &lk_rc_4_4 {
            let carry_arrays: Vec<BaseFieldVec> = lk_data.iter().map(|c| c.clone()).collect();
            let input: [BaseFieldVec; 2] = carry_arrays.try_into().unwrap();
            range_checks_trace_generator
                .rc_4_4_trace_generator
                .add_cuda_inputs(&[input]);
        }

        // 7. Wrap trace columns as CircleEvaluations and extend tree builder
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: Vec<CircleEvaluation<CudaBackend, M31, BitReversedOrder>> = trace_cols
            .into_iter()
            .map(|col| CircleEvaluation::new(domain, col))
            .collect();
        tree_builder.extend_evals(trace_evals);

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                lookup_data: CudaLookupData {
                    lk_mem,
                    lk_rc_3_3_3_3_3,
                    lk_rc_4_4_4_4,
                    lk_rc_4_4,
                    lk_pfrc,
                    lk_p3prc,
                },
                base_trace_ptrs,
                _keep_alive: base_trace_keep_alive,
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Sub-component input builders
// ---------------------------------------------------------------------------

/// Build CudaPackedInputType for poseidon_full_round_chain from lookup data.
/// Each PFRC lookup has 32 elements: [chain_idx, round, state0[0..9], state1[0..9], state2[0..9]]
/// Takes a slice of 8 sub-component feed arrays (pfrc[0..7]).
fn build_pfrc_input(
    lk_pfrc: &[[BaseFieldVec; 32]],
) -> poseidon_full_round_chain_cuda::CudaPackedInputType {
    assert_eq!(lk_pfrc.len(), 8, "Expected 8 sub-component PFRC feeds");
    // Concatenate all 8 feeds into a single CudaPackedInputType
    let total_size: usize = lk_pfrc[0][0].size;
    let n = total_size * 8;

    let mut input_limb_0 = BaseFieldVec::new_uninitialized(n);
    let mut input_limb_1 = BaseFieldVec::new_uninitialized(n);
    let mut state_0: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));
    let mut state_1: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));
    let mut state_2: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));

    // For each of the 8 feeds, copy the lookup data to the concatenated arrays
    for (feed_idx, feed) in lk_pfrc.iter().enumerate() {
        let offset = feed_idx * total_size;
        // Copy chain_idx (element 0)
        copy_gpu_region(&feed[0], &mut input_limb_0, offset, total_size);
        // Copy round (element 1)
        copy_gpu_region(&feed[1], &mut input_limb_1, offset, total_size);
        // Copy state0 (elements 2-11)
        for i in 0..10 {
            copy_gpu_region(&feed[2 + i], &mut state_0[i], offset, total_size);
        }
        // Copy state1 (elements 12-21)
        for i in 0..10 {
            copy_gpu_region(&feed[12 + i], &mut state_1[i], offset, total_size);
        }
        // Copy state2 (elements 22-31)
        for i in 0..10 {
            copy_gpu_region(&feed[22 + i], &mut state_2[i], offset, total_size);
        }
    }

    poseidon_full_round_chain_cuda::CudaPackedInputType {
        input_limb_0,
        input_limb_1,
        state_0,
        state_1,
        state_2,
    }
}

/// Build CudaPackedInputType for poseidon_3_partial_rounds_chain from lookup data.
/// Each P3PRC lookup has 42 elements: [chain_idx, round, state0[0..9], state1[0..9],
///                                      state2[0..9], state3[0..9]]
/// Takes a slice of 27 sub-component feed arrays (p3prc[0..26]).
fn build_p3prc_input(
    lk_p3prc: &[[BaseFieldVec; 42]],
) -> poseidon_3_partial_rounds_chain_cuda::CudaPackedInputType {
    assert_eq!(lk_p3prc.len(), 27, "Expected 27 sub-component P3PRC feeds");
    let total_size: usize = lk_p3prc[0][0].size;
    let n = total_size * 27;

    let mut input_limb_0 = BaseFieldVec::new_uninitialized(n);
    let mut input_limb_1 = BaseFieldVec::new_uninitialized(n);
    let mut state_0: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));
    let mut state_1: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));
    let mut state_2: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));
    let mut state_3: [BaseFieldVec; 10] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n));

    for (feed_idx, feed) in lk_p3prc.iter().enumerate() {
        let offset = feed_idx * total_size;
        copy_gpu_region(&feed[0], &mut input_limb_0, offset, total_size);
        copy_gpu_region(&feed[1], &mut input_limb_1, offset, total_size);
        for i in 0..10 {
            copy_gpu_region(&feed[2 + i], &mut state_0[i], offset, total_size);
        }
        for i in 0..10 {
            copy_gpu_region(&feed[12 + i], &mut state_1[i], offset, total_size);
        }
        for i in 0..10 {
            copy_gpu_region(&feed[22 + i], &mut state_2[i], offset, total_size);
        }
        for i in 0..10 {
            copy_gpu_region(&feed[32 + i], &mut state_3[i], offset, total_size);
        }
    }

    poseidon_3_partial_rounds_chain_cuda::CudaPackedInputType {
        input_limb_0,
        input_limb_1,
        state_0,
        state_1,
        state_2,
        state_3,
    }
}

/// Copy `count` elements from `src` GPU array to `dst` at `offset`.
/// Uses device-to-device memcpy — no CPU roundtrip.
fn copy_gpu_region(src: &BaseFieldVec, dst: &mut BaseFieldVec, offset: usize, count: usize) {
    dst.copy_region_from(src, offset, count);
}

// ---------------------------------------------------------------------------
// Interaction trace
// ---------------------------------------------------------------------------

struct CudaLookupData {
    lk_mem: [[BaseFieldVec; 29]; 6],
    lk_rc_3_3_3_3_3: [[BaseFieldVec; 5]; 2],
    lk_rc_4_4_4_4: [[BaseFieldVec; 4]; 6],
    lk_rc_4_4: [[BaseFieldVec; 2]; 3],
    lk_pfrc: [[BaseFieldVec; 32]; 10],  // 0-7: sub-component, 8-9: IT boundary
    lk_p3prc: [[BaseFieldVec; 42]; 28],  // 0-26: sub-component, 27: IT boundary
}

pub struct CudaInteractionClaimGenerator {
    log_size: u32,
    lookup_data: CudaLookupData,
    base_trace_ptrs: Vec<*const u32>,
    /// Keep GPU memory alive for base_trace_ptrs (the IT kernel reads certain columns
    /// directly from the base trace, so we must prevent them from being freed).
    _keep_alive: Vec<BaseFieldVec>,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        common_lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        let log_size = self.log_size;
        let trace_size = 1usize << log_size;
        let n_logup_cols = 14;
        let n_interaction_columns = n_logup_cols * 4; // 14 logup × 4 M31 each = 56

        // Allocate GPU columns for interaction trace (zero-initialized)
        let interaction_trace: Vec<BaseFieldVec> = (0..n_interaction_columns)
            .map(|_| BaseFieldVec::new_zeroes(trace_size))
            .collect();

        // Allocate GPU buffer for claimed_sum (4 M31s = 1 QM31)
        let cuda_claimed_sum = BaseFieldVec::new_zeroes(4);

        // Collect interaction trace pointers
        let interaction_trace_ptrs: Vec<*const u32> =
            interaction_trace.iter().map(|c| c.device_ptr).collect();

        // Collect lookup data pointers for all 14 columns
        let lk_mem_ptrs: [Vec<*const u32>; 6] = std::array::from_fn(|i| {
            self.lookup_data.lk_mem[i]
                .iter()
                .map(|c| c.device_ptr)
                .collect()
        });
        let lk_rc_3_3_3_3_3_ptrs: [Vec<*const u32>; 2] = std::array::from_fn(|i| {
            self.lookup_data.lk_rc_3_3_3_3_3[i]
                .iter()
                .map(|c| c.device_ptr)
                .collect()
        });
        let lk_rc_4_4_4_4_ptrs: [Vec<*const u32>; 6] = std::array::from_fn(|i| {
            self.lookup_data.lk_rc_4_4_4_4[i]
                .iter()
                .map(|c| c.device_ptr)
                .collect()
        });
        let lk_rc_4_4_ptrs: [Vec<*const u32>; 3] = std::array::from_fn(|i| {
            self.lookup_data.lk_rc_4_4[i]
                .iter()
                .map(|c| c.device_ptr)
                .collect()
        });
        // For interaction trace, 4 boundary PFRCs from the 10 base pfrc arrays.
        // Mapping: IT pfrc 0-3 → base pfrc [0, 8, 4, 9]
        //   IT pfrc_0 = base pfrc[0]: chain 0 entry (round=0)
        //   IT pfrc_1 = base pfrc[8]: chain 0 exit (round=4)
        //   IT pfrc_2 = base pfrc[4]: chain 1 entry (round=31)
        //   IT pfrc_3 = base pfrc[9]: chain 1 exit (round=35)
        const PFRC_IT_INDICES: [usize; 4] = [0, 8, 4, 9];
        let lk_pfrc_it_ptrs: [Vec<*const u32>; 4] = std::array::from_fn(|i| {
            self.lookup_data.lk_pfrc[PFRC_IT_INDICES[i]]
                .iter()
                .map(|c| c.device_ptr)
                .collect()
        });
        // For interaction trace, 2 boundary p3prc entries:
        //   IT p3prc_0 = base p3prc[0]: chain entry (round=4)
        //   IT p3prc_1 = base p3prc[27]: chain exit (round=31)
        const P3PRC_IT_INDICES: [usize; 2] = [0, 27];
        let lk_p3prc_it_ptrs: [Vec<*const u32>; 2] = std::array::from_fn(|i| {
            self.lookup_data.lk_p3prc[P3PRC_IT_INDICES[i]]
                .iter()
                .map(|c| c.device_ptr)
                .collect()
        });

        unsafe {
            bindings_airs::gen_poseidon_aggregator_interaction_trace(
                common_lookup_elements as *const CommonLookupElements
                    as *mut std::os::raw::c_void,
                // 6 memory_id_to_big lookup data
                lk_mem_ptrs[0].as_ptr() as *const u32,
                lk_mem_ptrs[1].as_ptr() as *const u32,
                lk_mem_ptrs[2].as_ptr() as *const u32,
                lk_mem_ptrs[3].as_ptr() as *const u32,
                lk_mem_ptrs[4].as_ptr() as *const u32,
                lk_mem_ptrs[5].as_ptr() as *const u32,
                // 2 rc_3_3_3_3_3
                lk_rc_3_3_3_3_3_ptrs[0].as_ptr() as *const u32,
                lk_rc_3_3_3_3_3_ptrs[1].as_ptr() as *const u32,
                // 6 rc_4_4_4_4
                lk_rc_4_4_4_4_ptrs[0].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[1].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[2].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[3].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[4].as_ptr() as *const u32,
                lk_rc_4_4_4_4_ptrs[5].as_ptr() as *const u32,
                // 3 rc_4_4
                lk_rc_4_4_ptrs[0].as_ptr() as *const u32,
                lk_rc_4_4_ptrs[1].as_ptr() as *const u32,
                lk_rc_4_4_ptrs[2].as_ptr() as *const u32,
                // 4 pfrc (for interaction trace)
                lk_pfrc_it_ptrs[0].as_ptr() as *const u32,
                lk_pfrc_it_ptrs[1].as_ptr() as *const u32,
                lk_pfrc_it_ptrs[2].as_ptr() as *const u32,
                lk_pfrc_it_ptrs[3].as_ptr() as *const u32,
                // 2 p3prc (for interaction trace)
                lk_p3prc_it_ptrs[0].as_ptr() as *const u32,
                lk_p3prc_it_ptrs[1].as_ptr() as *const u32,
                // Base trace column pointers
                self.base_trace_ptrs.as_ptr() as *const u32,
                log_size,
                // Output
                interaction_trace_ptrs.as_ptr() as *const u32,
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        // Read claimed_sum from GPU
        let cs = cuda_claimed_sum.to_vec();
        let claimed_sum = SecureField::from_m31_array([cs[0], cs[1], cs[2], cs[3]]);

        // Wrap interaction trace columns as CircleEvaluations and extend tree builder
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: Vec<CircleEvaluation<CudaBackend, M31, BitReversedOrder>> =
            interaction_trace
                .into_iter()
                .map(|col| CircleEvaluation::new(domain, col))
                .collect();
        tree_builder.extend_evals(trace_evals);

        InteractionClaim { claimed_sum }
    }
}
