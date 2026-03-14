//! CUDA version of partial_ec_mul_window_bits_18 component.
//!
//! This module provides GPU-accelerated trace generation following the blake_g_cuda
//! merged trace generation pattern:
//! - CudaClaimGenerator collects packed inputs (from pedersen_aggregator)
//! - write_trace() generates 297-col base trace + lookup data + sub-component inputs via CUDA
//! - CudaInteractionClaimGenerator generates 65 logup interaction columns via CUDA
//!
//! Sub-component feeds:
//! - pedersen_points_table: 1 feed (table index)
//! - range_check_9_9: 8 relation variants (6,6,6,6,6,6,3,3 feeds)
//! - range_check_20: 8 relation variants (12,12,12,12,9,9,9,9 feeds)

#![allow(unused_parens)]

use cairo_air::components::partial_ec_mul_window_bits_18::{
    Claim, InteractionClaim, N_TRACE_COLUMNS,
};
use cairo_air::relations::CommonLookupElements;
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::Column;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;

use crate::witness::components::partial_ec_mul_window_bits_18;
use crate::witness::components_cuda::pedersen_points_table_cuda;
use crate::witness::components_cuda::range_check::{rc_20, rc_9_9};
use crate::witness::prelude::*;

/// Number of logup columns for the interaction trace.
pub const N_LOGUP_COLUMNS: usize = 65;

/// Number of input columns (from PackedInputType: 1+1+14+28+28 = 72).
pub const N_INPUT_COLUMNS: usize = 72;

/// Re-export the packed input type from the SIMD component.
pub type PackedInputType = partial_ec_mul_window_bits_18::PackedInputType;

/// Instance counts per rc_20 variant [a,b,c,d,e,f,g,h] (from SIMD LookupData).
const RC_20_COUNTS: [usize; 8] = [12, 12, 12, 12, 9, 9, 9, 9];

/// Instance counts per rc_9_9 variant [a,b,c,d,e,f,g,h] (from SIMD LookupData).
const RC_9_9_COUNTS: [usize; 8] = [6, 6, 6, 6, 6, 6, 3, 3];

// ---------------------------------------------------------------------------
// Macros (matching blake_g_cuda pattern)
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

/// Initialize a Vec of BaseFieldVec with `count * elems_per_entry` elements.
fn init_lookup_vec(count: usize, elems_per_entry: usize, log_size: u32) -> Vec<BaseFieldVec> {
    let size = 1usize << log_size;
    (0..count * elems_per_entry)
        .map(|_| unsafe { BaseFieldVec::uninitialized(size) })
        .collect()
}

// ---------------------------------------------------------------------------
// CudaClaimGenerator
// ---------------------------------------------------------------------------

pub struct CudaClaimGenerator {
    /// Column-major GPU input columns (72 BaseFieldVec columns).
    /// Populated by add_packed_inputs() from SIMD packed inputs or set_cuda_inputs() from GPU
    /// data.
    pub input_columns: Vec<BaseFieldVec>,
    /// Number of valid (non-padding) rows.
    pub size: u32,
}

impl Default for CudaClaimGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaClaimGenerator {
    pub fn new() -> Self {
        Self {
            input_columns: Vec::new(),
            size: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Add packed inputs from the SIMD side. Converts to column-major GPU format.
    pub fn add_packed_inputs(&mut self, inputs: &[PackedInputType]) {
        // Each PackedInputType = (PackedM31, PackedM31, ([PackedM31; 14], [PackedFelt252; 2]))
        // Unpacks to 72 columns × 16 lanes per packed input.
        let new_rows = inputs.len() * 16;

        // Initialize column buffers if needed
        if self.input_columns.is_empty() {
            self.input_columns = (0..N_INPUT_COLUMNS)
                .map(|_| BaseFieldVec::new_zeroes(0))
                .collect();
        }

        let mut col_data: Vec<Vec<u32>> = (0..N_INPUT_COLUMNS)
            .map(|i| {
                let cpu = self.input_columns[i].to_vec();
                let mut v: Vec<u32> = cpu.into_iter().map(|b| b.0).collect();
                v.reserve(new_rows);
                v
            })
            .collect();

        for input in inputs {
            let (limb0, limb1, (limbs_14, felts_2)) = input;
            // Extract 16 SIMD lanes
            let limb0_arr: [u32; 16] = unsafe { std::mem::transmute(*limb0) };
            let limb1_arr: [u32; 16] = unsafe { std::mem::transmute(*limb1) };

            for lane in 0..16 {
                // Col 0: limb_0
                col_data[0].push(limb0_arr[lane]);
                // Col 1: limb_1
                col_data[1].push(limb1_arr[lane]);
            }

            // Cols 2-15: 14 limbs from [PackedM31; 14]
            for j in 0..14 {
                let arr: [u32; 16] = unsafe { std::mem::transmute(limbs_14[j]) };
                col_data[2 + j].extend_from_slice(&arr);
            }

            // Cols 16-43: Felt252[0] (28 limbs)
            for k in 0..28 {
                let packed_limb = felts_2[0].get_m31(k);
                let arr: [u32; 16] = unsafe { std::mem::transmute(packed_limb) };
                col_data[16 + k].extend_from_slice(&arr);
            }

            // Cols 44-71: Felt252[1] (28 limbs)
            for k in 0..28 {
                let packed_limb = felts_2[1].get_m31(k);
                let arr: [u32; 16] = unsafe { std::mem::transmute(packed_limb) };
                col_data[44 + k].extend_from_slice(&arr);
            }
        }

        self.input_columns = col_data
            .into_iter()
            .map(|v| {
                let bf_vec: Vec<BaseField> =
                    v.into_iter().map(BaseField::from_u32_unchecked).collect();
                BaseFieldVec::from_vec(bf_vec)
            })
            .collect();
        self.size = self.input_columns[0].size as u32;
    }

    /// Set CUDA inputs directly from GPU-resident columns (used when aggregator runs on CUDA).
    pub fn set_cuda_inputs(
        &mut self,
        cuda_inputs: [BaseFieldVec; N_INPUT_COLUMNS],
        total_rows: usize,
    ) {
        self.input_columns = cuda_inputs.into_iter().collect();
        self.size = total_rows as u32;
    }

    pub fn write_trace(
        mut self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        pedersen_points_table_state: &pedersen_points_table_cuda::CudaClaimGenerator,
        range_check_9_9_state: &rc_9_9::CudaClaimGenerator,
        range_check_20_state: &rc_20::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        assert!(
            self.size > 0,
            "partial_ec_mul_cuda: write_trace called with 0 inputs"
        );

        let n_rows = self.size as usize;
        let padded_size = n_rows.next_power_of_two();
        let log_size = padded_size.ilog2();

        // Pad input columns to next power of 2 by repeating the first packed row
        // (first 16 elements), matching the SIMD path.
        // GPU in-place padding — no download/upload roundtrip.
        if padded_size > n_rows {
            const N_LANES: usize = 16;
            let cycle_len = N_LANES.min(n_rows);
            for col in self.input_columns.iter_mut() {
                col.pad_with_cycle(n_rows, padded_size, cycle_len);
            }
        }

        let (trace, lookup_data, sub_component_inputs) =
            write_trace_cuda(&self.input_columns, n_rows, log_size);

        // Feed sub-components with GPU-resident data
        // PPT: 1 feed of [BaseFieldVec; 1]
        let ppt_input: &[pedersen_points_table_cuda::CudaPackedInputType] = unsafe {
            std::slice::from_raw_parts(
                sub_component_inputs.ppt.as_ptr()
                    as *const pedersen_points_table_cuda::CudaPackedInputType,
                1,
            )
        };
        pedersen_points_table_state.add_cuda_inputs(ppt_input);

        // rc_9_9: 8 relation variants, each with varying feeds of [BaseFieldVec; 2]
        for rel in 0..8 {
            let count = RC_9_9_COUNTS[rel];
            let feeds = &sub_component_inputs.rc_9_9[rel];
            debug_assert_eq!(feeds.len(), count * 2);
            let typed: &[rc_9_9::CudaPackedInputType] = unsafe {
                std::slice::from_raw_parts(
                    feeds.as_ptr() as *const rc_9_9::CudaPackedInputType,
                    count,
                )
            };
            range_check_9_9_state.add_cuda_inputs_for_relation(typed, rel);
        }

        // rc_20: 8 relation variants, each with varying feeds of [BaseFieldVec; 1]
        for rel in 0..8 {
            let count = RC_20_COUNTS[rel];
            let feeds = &sub_component_inputs.rc_20[rel];
            debug_assert_eq!(feeds.len(), count);
            let typed: &[rc_20::CudaPackedInputType] = unsafe {
                std::slice::from_raw_parts(
                    feeds.as_ptr() as *const rc_20::CudaPackedInputType,
                    count,
                )
            };
            range_check_20_state.add_cuda_inputs_for_relation(typed, rel);
        }

        tree_builder.extend_evals(trace.to_evals().to_vec());

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

// ---------------------------------------------------------------------------
// CudaLookupData — stores lookup arrays for interaction trace generation
// ---------------------------------------------------------------------------

struct CudaLookupData {
    // Self-interaction lookups (provide/consume)
    partial_ec_mul_0: [BaseFieldVec; 73],
    partial_ec_mul_1: [BaseFieldVec; 73],
    // Pedersen points table lookup
    ppt_0: [BaseFieldVec; 58],
    // rc_20 lookups: 8 variants, each a flat Vec of BaseFieldVec
    // Group i has RC_20_COUNTS[i] entries × 2 elements per entry
    rc_20: [Vec<BaseFieldVec>; 8],
    // rc_9_9 lookups: 8 variants, each a flat Vec of BaseFieldVec
    // Group i has RC_9_9_COUNTS[i] entries × 3 elements per entry
    rc_9_9: [Vec<BaseFieldVec>; 8],
}

// ---------------------------------------------------------------------------
// CudaSubComponentInputs — stores sub-component feed arrays
// ---------------------------------------------------------------------------

struct CudaSubComponentInputs {
    // PPT: 1 feed of 1 column (table index)
    ppt: [BaseFieldVec; 1],
    // rc_9_9: 8 variant groups, flat Vec of BaseFieldVec
    // Group i has RC_9_9_COUNTS[i] feeds × 2 columns per feed
    rc_9_9: [Vec<BaseFieldVec>; 8],
    // rc_20: 8 variant groups, flat Vec of BaseFieldVec
    // Group i has RC_20_COUNTS[i] feeds × 1 column per feed
    rc_20: [Vec<BaseFieldVec>; 8],
}

// ---------------------------------------------------------------------------
// write_trace_cuda — merged trace + lookup + sub-inputs generation
// ---------------------------------------------------------------------------

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_cuda(
    inputs: &[BaseFieldVec],
    n_rows: usize,
    log_size: u32,
) -> (
    CudaComponentTrace<N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    // Ensure the pedersen table is initialized on GPU.
    // The kernel reads g_pedersen_table_columns which must be initialized.
    pedersen_points_table_cuda::ensure_pedersen_table();

    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                partial_ec_mul_0: init_lookup_array!(log_size),
                partial_ec_mul_1: init_lookup_array!(log_size),
                ppt_0: init_lookup_array!(log_size),
                rc_20: std::array::from_fn(|i| init_lookup_vec(RC_20_COUNTS[i], 2, log_size)),
                rc_9_9: std::array::from_fn(|i| init_lookup_vec(RC_9_9_COUNTS[i], 3, log_size)),
            },
            CudaSubComponentInputs {
                ppt: init_lookup_array!(log_size),
                rc_9_9: std::array::from_fn(|i| init_lookup_vec(RC_9_9_COUNTS[i], 2, log_size)),
                rc_20: std::array::from_fn(|i| init_lookup_vec(RC_20_COUNTS[i], 1, log_size)),
            },
        )
    };

    // Collect device pointers for FFI call
    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Lookup data pointers
    let lookup_pem_0_vec = collect_ptrs!(lookup_data.partial_ec_mul_0);
    let lookup_pem_1_vec = collect_ptrs!(lookup_data.partial_ec_mul_1);
    let lookup_ppt_0_vec = collect_ptrs!(lookup_data.ppt_0);

    let lookup_rc_20_vecs: Vec<Vec<*const u32>> = lookup_data
        .rc_20
        .iter()
        .map(|group| group.iter().map(|b| b.device_ptr).collect_vec())
        .collect();

    let lookup_rc_9_9_vecs: Vec<Vec<*const u32>> = lookup_data
        .rc_9_9
        .iter()
        .map(|group| group.iter().map(|b| b.device_ptr).collect_vec())
        .collect();

    // Sub-component input pointers
    let sub_ppt_vec = collect_ptrs!(sub_component_inputs.ppt);

    let sub_rc_9_9_vecs: Vec<Vec<*const u32>> = sub_component_inputs
        .rc_9_9
        .iter()
        .map(|group| group.iter().map(|b| b.device_ptr).collect_vec())
        .collect();

    let sub_rc_20_vecs: Vec<Vec<*const u32>> = sub_component_inputs
        .rc_20
        .iter()
        .map(|group| group.iter().map(|b| b.device_ptr).collect_vec())
        .collect();

    // Input pointers
    let input_vec: Vec<*const u32> = inputs.iter().map(|c| c.device_ptr).collect_vec();

    unsafe {
        bindings_airs::gen_partial_ec_mul_wb18_trace(
            traces_vec.as_ptr(),
            // Lookup data
            lookup_pem_0_vec.as_ptr(),
            lookup_pem_1_vec.as_ptr(),
            lookup_ppt_0_vec.as_ptr(),
            lookup_rc_20_vecs[0].as_ptr(),
            lookup_rc_20_vecs[1].as_ptr(),
            lookup_rc_20_vecs[2].as_ptr(),
            lookup_rc_20_vecs[3].as_ptr(),
            lookup_rc_20_vecs[4].as_ptr(),
            lookup_rc_20_vecs[5].as_ptr(),
            lookup_rc_20_vecs[6].as_ptr(),
            lookup_rc_20_vecs[7].as_ptr(),
            lookup_rc_9_9_vecs[0].as_ptr(),
            lookup_rc_9_9_vecs[1].as_ptr(),
            lookup_rc_9_9_vecs[2].as_ptr(),
            lookup_rc_9_9_vecs[3].as_ptr(),
            lookup_rc_9_9_vecs[4].as_ptr(),
            lookup_rc_9_9_vecs[5].as_ptr(),
            lookup_rc_9_9_vecs[6].as_ptr(),
            lookup_rc_9_9_vecs[7].as_ptr(),
            // Sub-component inputs
            sub_ppt_vec.as_ptr(),
            sub_rc_9_9_vecs[0].as_ptr(),
            sub_rc_9_9_vecs[1].as_ptr(),
            sub_rc_9_9_vecs[2].as_ptr(),
            sub_rc_9_9_vecs[3].as_ptr(),
            sub_rc_9_9_vecs[4].as_ptr(),
            sub_rc_9_9_vecs[5].as_ptr(),
            sub_rc_9_9_vecs[6].as_ptr(),
            sub_rc_9_9_vecs[7].as_ptr(),
            sub_rc_20_vecs[0].as_ptr(),
            sub_rc_20_vecs[1].as_ptr(),
            sub_rc_20_vecs[2].as_ptr(),
            sub_rc_20_vecs[3].as_ptr(),
            sub_rc_20_vecs[4].as_ptr(),
            sub_rc_20_vecs[5].as_ptr(),
            sub_rc_20_vecs[6].as_ptr(),
            sub_rc_20_vecs[7].as_ptr(),
            // Inputs
            input_vec.as_ptr(),
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
    n_rows: usize,
    log_size: u32,
    lookup_data: CudaLookupData,
}

impl CudaInteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        lookup_elements: &CommonLookupElements,
    ) -> InteractionClaim {
        use crate::witness::components_cuda::cuda_lookup_helper::*;

        let log_size = self.log_size;
        let trace_size = 1usize << log_size;

        // Allocate claimed_sum on GPU (4 u32s for QM31)
        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        // Allocate interaction trace columns on GPU (4 × 65 = 260 columns)
        let interaction_trace: Vec<BaseFieldVec> = (0..4 * N_LOGUP_COLUMNS)
            .map(|_| <BaseFieldVec as Column<BaseField>>::zeros(trace_size))
            .collect();

        // Collect lookup data pointers (already on GPU — no download needed)
        let lookup_pem_0_vec = collect_ptrs!(self.lookup_data.partial_ec_mul_0);
        let lookup_pem_1_vec = collect_ptrs!(self.lookup_data.partial_ec_mul_1);
        let lookup_ppt_0_vec = collect_ptrs!(self.lookup_data.ppt_0);

        let lookup_rc_20_vecs: Vec<Vec<*const u32>> = self
            .lookup_data
            .rc_20
            .iter()
            .map(|group| group.iter().map(|b| b.device_ptr).collect_vec())
            .collect();

        let lookup_rc_9_9_vecs: Vec<Vec<*const u32>> = self
            .lookup_data
            .rc_9_9
            .iter()
            .map(|group| group.iter().map(|b| b.device_ptr).collect_vec())
            .collect();

        let interaction_trace_vec: Vec<*const u32> = interaction_trace
            .iter()
            .map(|col| col.device_ptr)
            .collect_vec();

        // Create modified lookup elements for each relation (18 total)
        let mod_pem = create_modified_lookup_for_cuda(lookup_elements, PARTIAL_EC_MUL_RELATION_ID);
        let mod_ppt =
            create_modified_lookup_for_cuda(lookup_elements, PEDERSEN_POINTS_TABLE_RELATION_ID);

        let rc20_ids = [
            RC_19_RELATION_ID,
            RC_19_B_RELATION_ID,
            RC_19_C_RELATION_ID,
            RC_19_D_RELATION_ID,
            RC_19_E_RELATION_ID,
            RC_19_F_RELATION_ID,
            RC_19_G_RELATION_ID,
            RC_19_H_RELATION_ID,
        ];
        let mod_rc20: Vec<_> = rc20_ids
            .iter()
            .map(|&id| create_modified_lookup_for_cuda(lookup_elements, id))
            .collect();

        let mod_rc99: Vec<_> = RC_9_9_RELATION_IDS
            .iter()
            .map(|&id| create_modified_lookup_for_cuda(lookup_elements, id))
            .collect();

        unsafe {
            bindings_airs::gen_partial_ec_mul_wb18_interaction_trace(
                // Lookup elements (18 total)
                &mod_pem as *const _ as *mut std::os::raw::c_void,
                &mod_ppt as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[0] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[1] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[2] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[3] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[4] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[5] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[6] as *const _ as *mut std::os::raw::c_void,
                &mod_rc20[7] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[0] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[1] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[2] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[3] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[4] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[5] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[6] as *const _ as *mut std::os::raw::c_void,
                &mod_rc99[7] as *const _ as *mut std::os::raw::c_void,
                // Lookup data pointers
                lookup_pem_0_vec.as_ptr(),
                lookup_pem_1_vec.as_ptr(),
                lookup_ppt_0_vec.as_ptr(),
                lookup_rc_20_vecs[0].as_ptr(),
                lookup_rc_20_vecs[1].as_ptr(),
                lookup_rc_20_vecs[2].as_ptr(),
                lookup_rc_20_vecs[3].as_ptr(),
                lookup_rc_20_vecs[4].as_ptr(),
                lookup_rc_20_vecs[5].as_ptr(),
                lookup_rc_20_vecs[6].as_ptr(),
                lookup_rc_20_vecs[7].as_ptr(),
                lookup_rc_9_9_vecs[0].as_ptr(),
                lookup_rc_9_9_vecs[1].as_ptr(),
                lookup_rc_9_9_vecs[2].as_ptr(),
                lookup_rc_9_9_vecs[3].as_ptr(),
                lookup_rc_9_9_vecs[4].as_ptr(),
                lookup_rc_9_9_vecs[5].as_ptr(),
                lookup_rc_9_9_vecs[6].as_ptr(),
                lookup_rc_9_9_vecs[7].as_ptr(),
                // Sizes
                self.n_rows as u32,
                log_size,
                // Output
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        // Read claimed_sum from GPU
        let cs = cuda_claimed_sum.to_cpu();
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
