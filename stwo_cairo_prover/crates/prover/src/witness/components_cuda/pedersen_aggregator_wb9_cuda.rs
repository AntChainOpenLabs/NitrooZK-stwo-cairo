//! Native CUDA trace generation for pedersen_aggregator_window_bits_9 (234 columns).
//!
//! Adapted from pedersen_aggregator_cuda.rs (window_bits_18 variant).
//! Uses g_pedersen_table_small_columns on GPU (56 cols × 32K rows, ~7MB).
//!
//! Architecture:
//! - CudaClaimGenerator collects inputs via the inner SIMD DashMap (from pedersen_builtin)
//! - write_trace() sorts + packs inputs on CPU, uploads to GPU, calls CUDA kernel
//! - CUDA kernel generates 234 trace cols + lookup data + sub-component inputs
//! - Sub-component feeds: 3× memory_id_to_big, 4× range_check_8, 56× partial_ec_mul_wb9
//! - CudaInteractionClaimGenerator: computes 6 logup columns entirely on GPU

#![allow(unused_parens)]

use cairo_air::components::pedersen_aggregator_window_bits_9::{
    Claim, InteractionClaim, N_TRACE_COLUMNS,
};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use crate::witness::components::pedersen_aggregator_window_bits_9;
use crate::witness::components_cuda::range_check::rc_8;
use crate::witness::components_cuda::{
    memory_id_to_big_cuda, partial_ec_mul_wb9_cuda, pedersen_points_table_wb9_cuda,
};
use crate::witness::prelude::*;

/// Number of logup columns for the interaction trace.
pub const N_LOGUP_COLUMNS: usize = 6;

// ---------------------------------------------------------------------------
// Macros (matching partial_ec_mul_cuda pattern)
// ---------------------------------------------------------------------------

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}

macro_rules! collect_ptrs {
    ($data:expr) => {
        $data.iter().map(|x| x.device_ptr).collect_vec()
    };
}

fn init_lookup_vec(count: usize, log_size: u32) -> Vec<BaseFieldVec> {
    let size = 1usize << log_size;
    unsafe {
        (0..count)
            .map(|_| BaseFieldVec::uninitialized(size))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CudaClaimGenerator
// ---------------------------------------------------------------------------

/// Native CUDA claim generator for pedersen_aggregator_window_bits_9.
pub struct CudaClaimGenerator {
    inner: pedersen_aggregator_window_bits_9::ClaimGenerator,
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            inner: pedersen_aggregator_window_bits_9::ClaimGenerator::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Delegate input tracking to the inner SIMD generator.
    pub fn inner(&self) -> &pedersen_aggregator_window_bits_9::ClaimGenerator {
        &self.inner
    }

    /// Write the aggregator trace (234 columns) to a CUDA tree builder.
    pub fn write_trace(
        self,
        cuda_tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        rc_8_state: &rc_8::CudaClaimGenerator,
        pem_wb9_cuda: &mut partial_ec_mul_wb9_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        // 1. Sort + pack inputs on CPU (same as SIMD path).
        let mut inputs_mults = self
            .inner
            .mults
            .iter()
            .map(|entry| {
                (
                    *entry.key(),
                    M31(entry.value().load(std::sync::atomic::Ordering::Relaxed)),
                )
            })
            .collect::<Vec<_>>();
        inputs_mults.sort_by_key(|(input, _)| input.0);
        let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

        let n_rows = inputs.len();
        assert_ne!(
            n_rows, 0,
            "pedersen_aggregator_wb9_cuda: write_trace called with 0 inputs"
        );
        let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
        let log_size = size.ilog2();
        inputs.resize(size, *inputs.first().unwrap());
        mults.resize(size, M31::zero());

        // 2. Upload inputs to GPU as 3 BaseFieldVec columns.
        let mut input_col_0 = Vec::with_capacity(size);
        let mut input_col_1 = Vec::with_capacity(size);
        let mut input_col_2 = Vec::with_capacity(size);
        let mut mults_vec = Vec::with_capacity(size);
        for (input, mult) in inputs.iter().zip(mults.iter()) {
            let ([limb0, limb1], limb2) = input;
            input_col_0.push(*limb0);
            input_col_1.push(*limb1);
            input_col_2.push(*limb2);
            mults_vec.push(*mult);
        }
        let input_cols = [
            BaseFieldVec::from_vec(input_col_0),
            BaseFieldVec::from_vec(input_col_1),
            BaseFieldVec::from_vec(input_col_2),
        ];
        let mults_gpu = BaseFieldVec::from_vec(mults_vec);

        // 3. Call the CUDA kernel.
        let (trace, lookup_data, sub_component_inputs) = write_trace_cuda(
            &input_cols,
            mults_gpu,
            memory_id_to_big_state,
            n_rows,
            log_size,
        );

        // 4. Feed sub-component inputs to CUDA generators.
        // memory_id_to_big: 3 feeds × 1 column each
        let mem_inputs: Vec<memory_id_to_big_cuda::CudaPackedInputType> =
            sub_component_inputs.mem.into_iter().map(|v| [v]).collect();
        memory_id_to_big_state.add_cuda_inputs(&mem_inputs);

        // range_check_8: 4 feeds × 1 column each
        let rc8_inputs: Vec<rc_8::CudaPackedInputType> =
            sub_component_inputs.rc8.into_iter().map(|v| [v]).collect();
        rc_8_state.add_cuda_inputs(&rc8_inputs);

        // partial_ec_mul_wb9: 86 columns × (56 * size) rows, feed via set_cuda_inputs
        let pem_total_rows = 56 * size;
        let pem_inputs: [BaseFieldVec; 86] = sub_component_inputs
            .pem
            .try_into()
            .expect("sub_pem should have 86 columns");
        pem_wb9_cuda.set_cuda_inputs(pem_inputs, pem_total_rows);

        // 5. Extend tree builder with trace columns.
        tree_builder_extend_cuda(cuda_tree_builder, trace);

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                log_size,
                lookup_data,
            },
        )
    }
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

fn tree_builder_extend_cuda(
    tree_builder: &mut impl TreeBuilder<CudaBackend>,
    trace: CudaComponentTrace<N_TRACE_COLUMNS>,
) {
    tree_builder.extend_evals(trace.to_evals().to_vec());
}

// ---------------------------------------------------------------------------
// CudaLookupData
// ---------------------------------------------------------------------------

struct CudaLookupData {
    mem_0: [BaseFieldVec; 30],
    mem_1: [BaseFieldVec; 30],
    mem_2: [BaseFieldVec; 30],
    rc8_0: [BaseFieldVec; 2],
    rc8_1: [BaseFieldVec; 2],
    rc8_2: [BaseFieldVec; 2],
    rc8_3: [BaseFieldVec; 2],
    pem_0: [BaseFieldVec; 87],
    pem_1: [BaseFieldVec; 87],
    pem_2: [BaseFieldVec; 87],
    pem_3: [BaseFieldVec; 87],
    agg_0: [BaseFieldVec; 4],
    mults: BaseFieldVec,
}

// ---------------------------------------------------------------------------
// CudaSubComponentInputs
// ---------------------------------------------------------------------------

struct CudaSubComponentInputs {
    mem: Vec<BaseFieldVec>, // 3 arrays
    rc8: Vec<BaseFieldVec>, // 4 arrays
    pem: Vec<BaseFieldVec>, // 86 arrays (each 56*trace_size)
}

// ---------------------------------------------------------------------------
// write_trace_cuda
// ---------------------------------------------------------------------------

#[allow(clippy::useless_conversion)]
fn write_trace_cuda(
    inputs: &[BaseFieldVec; 3],
    mults: BaseFieldVec,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    n_rows: usize,
    log_size: u32,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    // Ensure the small pedersen table is initialized on GPU.
    pedersen_points_table_wb9_cuda::ensure_pedersen_table_small();

    let trace_size = 1usize << log_size;

    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                mem_0: init_lookup_array!(log_size),
                mem_1: init_lookup_array!(log_size),
                mem_2: init_lookup_array!(log_size),
                rc8_0: init_lookup_array!(log_size),
                rc8_1: init_lookup_array!(log_size),
                rc8_2: init_lookup_array!(log_size),
                rc8_3: init_lookup_array!(log_size),
                pem_0: init_lookup_array!(log_size),
                pem_1: init_lookup_array!(log_size),
                pem_2: init_lookup_array!(log_size),
                pem_3: init_lookup_array!(log_size),
                agg_0: init_lookup_array!(log_size),
                mults,
            },
            CudaSubComponentInputs {
                mem: init_lookup_vec(3, log_size),
                rc8: init_lookup_vec(4, log_size),
                // PEM wb9: 86 columns, each with 56 * trace_size elements
                pem: (0..86)
                    .map(|_| BaseFieldVec::uninitialized(56 * trace_size))
                    .collect(),
            },
        )
    };

    // Collect device pointers for FFI call
    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();
    let input_vec: Vec<*const u32> = inputs.iter().map(|c| c.device_ptr).collect_vec();

    // Lookup data pointers
    let lk_mem_0_vec = collect_ptrs!(lookup_data.mem_0);
    let lk_mem_1_vec = collect_ptrs!(lookup_data.mem_1);
    let lk_mem_2_vec = collect_ptrs!(lookup_data.mem_2);
    let lk_rc8_0_vec = collect_ptrs!(lookup_data.rc8_0);
    let lk_rc8_1_vec = collect_ptrs!(lookup_data.rc8_1);
    let lk_rc8_2_vec = collect_ptrs!(lookup_data.rc8_2);
    let lk_rc8_3_vec = collect_ptrs!(lookup_data.rc8_3);
    let lk_pem_0_vec = collect_ptrs!(lookup_data.pem_0);
    let lk_pem_1_vec = collect_ptrs!(lookup_data.pem_1);
    let lk_pem_2_vec = collect_ptrs!(lookup_data.pem_2);
    let lk_pem_3_vec = collect_ptrs!(lookup_data.pem_3);
    let lk_agg_0_vec = collect_ptrs!(lookup_data.agg_0);

    // Sub-component input pointers
    let sub_mem_vec = collect_ptrs!(sub_component_inputs.mem);
    let sub_rc8_vec = collect_ptrs!(sub_component_inputs.rc8);
    let sub_pem_vec = collect_ptrs!(sub_component_inputs.pem);

    // Memory state pointers for the kernel
    let memory_transposed_big_values_vec: Vec<*const u32> = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::gen_pedersen_aggregator_wb9_trace(
            traces_vec.as_ptr(),
            lk_mem_0_vec.as_ptr(),
            lk_mem_1_vec.as_ptr(),
            lk_mem_2_vec.as_ptr(),
            lk_rc8_0_vec.as_ptr(),
            lk_rc8_1_vec.as_ptr(),
            lk_rc8_2_vec.as_ptr(),
            lk_rc8_3_vec.as_ptr(),
            lk_pem_0_vec.as_ptr(),
            lk_pem_1_vec.as_ptr(),
            lk_pem_2_vec.as_ptr(),
            lk_pem_3_vec.as_ptr(),
            lk_agg_0_vec.as_ptr(),
            lookup_data.mults.device_ptr,
            sub_mem_vec.as_ptr(),
            sub_rc8_vec.as_ptr(),
            sub_pem_vec.as_ptr(),
            input_vec.as_ptr(),
            memory_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr as *const u32,
            n_rows as u32,
            log_size,
        );
    }

    (trace, lookup_data, sub_component_inputs)
}

// ---------------------------------------------------------------------------
// CudaInteractionClaimGenerator
// ---------------------------------------------------------------------------

pub struct CudaInteractionClaimGenerator {
    log_size: u32,
    lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        use stwo::core::fields::m31::M31;
        use stwo::core::fields::qm31::SecureField;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::backend::Column;
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;

        let log_size = self.log_size;
        let trace_size = 1usize << log_size;

        // Allocate interaction trace columns on GPU (4 × 6 = 24 columns)
        let interaction_trace: Vec<BaseFieldVec> = (0..4 * N_LOGUP_COLUMNS)
            .map(|_| <BaseFieldVec as Column<BaseField>>::zeros(trace_size))
            .collect();
        let interaction_trace_ptrs: Vec<*const u32> = interaction_trace
            .iter()
            .map(|col| col.device_ptr)
            .collect_vec();

        // Allocate claimed_sum on GPU (4 M31s for QM31)
        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Collect lookup data pointers
        let lk_mem_0_vec = collect_ptrs!(self.lookup_data.mem_0);
        let lk_mem_1_vec = collect_ptrs!(self.lookup_data.mem_1);
        let lk_mem_2_vec = collect_ptrs!(self.lookup_data.mem_2);
        let lk_rc8_0_vec = collect_ptrs!(self.lookup_data.rc8_0);
        let lk_rc8_1_vec = collect_ptrs!(self.lookup_data.rc8_1);
        let lk_rc8_2_vec = collect_ptrs!(self.lookup_data.rc8_2);
        let lk_rc8_3_vec = collect_ptrs!(self.lookup_data.rc8_3);
        let lk_pem_0_vec = collect_ptrs!(self.lookup_data.pem_0);
        let lk_pem_1_vec = collect_ptrs!(self.lookup_data.pem_1);
        let lk_pem_2_vec = collect_ptrs!(self.lookup_data.pem_2);
        let lk_pem_3_vec = collect_ptrs!(self.lookup_data.pem_3);
        let lk_agg_0_vec = collect_ptrs!(self.lookup_data.agg_0);

        unsafe {
            bindings_airs::gen_pedersen_aggregator_wb9_interaction_trace(
                lookup_elements as *const CommonLookupElements as *mut std::os::raw::c_void,
                lk_mem_0_vec.as_ptr(),
                lk_mem_1_vec.as_ptr(),
                lk_mem_2_vec.as_ptr(),
                lk_rc8_0_vec.as_ptr(),
                lk_rc8_1_vec.as_ptr(),
                lk_rc8_2_vec.as_ptr(),
                lk_rc8_3_vec.as_ptr(),
                lk_pem_0_vec.as_ptr(),
                lk_pem_1_vec.as_ptr(),
                lk_pem_2_vec.as_ptr(),
                lk_pem_3_vec.as_ptr(),
                lk_agg_0_vec.as_ptr(),
                self.lookup_data.mults.device_ptr,
                log_size,
                interaction_trace_ptrs.as_ptr(),
                cuda_claimed_sum.device_ptr as *mut u32,
            );
        }

        // Read claimed_sum from GPU
        let cs: Vec<M31> = cuda_claimed_sum.to_vec();
        let claimed_sum = SecureField::from_m31_array([cs[0], cs[1], cs[2], cs[3]]);

        // Wrap as CircleEvaluations and extend tree builder
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace: Vec<CircleEvaluation<CudaBackend, _, BitReversedOrder>> = interaction_trace
            .into_iter()
            .map(|col| CircleEvaluation::new(domain, col))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
