#![allow(unused_parens)]
use cairo_air::components::generic_opcode::{Claim, InteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::prover::backend::cuda::CudaBackend;

use super::super::{
    memory_address_to_id_cuda, memory_id_to_big_cuda, range_check_11_cuda, range_check_18_cuda,
    range_check_20_cuda, range_check_9_9_cuda, verify_instruction_cuda,
};
use crate::witness::prelude::*;

pub type InputType = CasmState;
pub type PackedInputType = PackedCasmState;
use stwo::core::fields::m31::BaseField;
use stwo_air_utils::trace::component_trace::CudaComponentTrace;
/// Number of trace columns the CUDA kernel actually generates (includes enabler column).
const CUDA_N_TRACE_COLUMNS: usize = 244;
/// Number of trace columns the AIR expects (no enabler column).
pub const N_TRACE_COLUMNS: usize = 243;
pub const N_INTERACTION_TRACE_COLUMNS: usize = 34;

pub type CudaPackedInputs = [BaseFieldVec; 3];
use itertools::Itertools;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::Col;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings_airs;

macro_rules! init_lookup_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
    };
}
macro_rules! init_subcomponent_basefield_array {
    ($log_size:expr) => {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseFieldVec::uninitialized(1 << $log_size))
        })
    };
}

macro_rules! collect_lookup_ptrs {
    ($lookup_data:expr, $field:ident) => {
        $lookup_data
            .$field
            .iter()
            .map(|x| x.device_ptr)
            .collect_vec()
    };
}

macro_rules! collect_sub_input_ptrs {
    ($sub_inputs:expr, $field:ident) => {
        $sub_inputs
            .$field
            .iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec()
    };
}

macro_rules! collect_input_ptrs {
    ($input:expr) => {
        $input.iter().map(|x| x.device_ptr).collect_vec()
    };
}

pub struct CudaClaimGenerator {
    pub n_rows: usize,
    pub inputs: CudaPackedInputs,
}

impl CudaClaimGenerator {
    pub fn new(inputs: Vec<InputType>) -> Self {
        let n_rows = inputs.len();

        let mut pcs = Vec::with_capacity(n_rows);
        let mut aps = Vec::with_capacity(n_rows);
        let mut fps = Vec::with_capacity(n_rows);

        for input in inputs.clone() {
            pcs.push(BaseField::from(input.pc));
            aps.push(BaseField::from(input.ap));
            fps.push(BaseField::from(input.fp));
        }
        let size = std::cmp::max(inputs.len().next_power_of_two(), N_LANES);
        pcs.resize(size, pcs[0]);
        aps.resize(size, aps[0]);
        fps.resize(size, fps[0]);

        let pc_vec = BaseFieldVec::from_vec(pcs);
        let ap_vec = BaseFieldVec::from_vec(aps);
        let fp_vec = BaseFieldVec::from_vec(fps);

        Self {
            n_rows,
            inputs: [pc_vec, ap_vec, fp_vec],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_trace(
        self,
        tree_builder: &mut impl TreeBuilder<CudaBackend>,
        memory_address_to_id_cuda_state: &mut memory_address_to_id_cuda::CudaClaimGenerator,
        memory_id_to_big_cuda_state: &memory_id_to_big_cuda::CudaClaimGenerator,
        verify_instruction_cuda_state: &verify_instruction_cuda::CudaClaimGenerator,
        // Range check CUDA generators (consolidated - use relation_index to distinguish instances)
        range_check_9_9_cuda_state: &range_check_9_9_cuda::CudaClaimGenerator,
        range_check_20_cuda_state: &range_check_20_cuda::CudaClaimGenerator,
        range_check_18_cuda_state: &range_check_18_cuda::CudaClaimGenerator,
        range_check_11_cuda_state: &range_check_11_cuda::CudaClaimGenerator,
    ) -> (Claim, CudaInteractionClaimGenerator) {
        let size = self.inputs[0].size;
        let log_size = size.ilog2();
        let packed_inputs = self.inputs;

        let (trace, mut lookup_data, sub_component_inputs) = write_trace_cuda(
            self.n_rows,
            packed_inputs,
            memory_address_to_id_cuda_state,
            memory_id_to_big_cuda_state,
            verify_instruction_cuda_state,
        );

        // Normalize range_check_19_h_0 offset: CUDA stores k_col + 262144 but all other
        // rc_19 carry lookups use offset 131072. Subtract 131072 so all rc_19 lookups
        // have uniform base offset (131072), enabling a single offset correction (393216)
        // to reach the correct rc_20 offset (524288).
        // In M31: -131072 ≡ P - 131072 = 2147352575.
        lookup_data.range_check_19_h_0[0].add_offset_in_place(M31(2147483647 - 131072));

        // Add to CUDA generators for multiplicity accumulation
        verify_instruction_cuda_state.add_cuda_inputs(&sub_component_inputs.verify_instruction);
        memory_address_to_id_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_address_to_id);
        memory_id_to_big_cuda_state.add_cuda_inputs(&sub_component_inputs.memory_id_to_big);

        // NOTE: Do NOT add to memory SIMD generators here - that would cause double-counting
        // when merge_simd_multiplicities() is called later.

        // Add range check inputs via CUDA path.
        // range_check_9_9: relation 0, 4 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_0.clone(),
                lookup_data.range_check_9_9_1.clone(),
                lookup_data.range_check_9_9_2.clone(),
                lookup_data.range_check_9_9_3.clone(),
            ],
            0,
        );
        // range_check_9_9_b: relation 1, 4 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_b_0.clone(),
                lookup_data.range_check_9_9_b_1.clone(),
                lookup_data.range_check_9_9_b_2.clone(),
                lookup_data.range_check_9_9_b_3.clone(),
            ],
            1,
        );
        // range_check_9_9_c: relation 2, 4 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_c_0.clone(),
                lookup_data.range_check_9_9_c_1.clone(),
                lookup_data.range_check_9_9_c_2.clone(),
                lookup_data.range_check_9_9_c_3.clone(),
            ],
            2,
        );
        // range_check_9_9_d: relation 3, 4 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_d_0.clone(),
                lookup_data.range_check_9_9_d_1.clone(),
                lookup_data.range_check_9_9_d_2.clone(),
                lookup_data.range_check_9_9_d_3.clone(),
            ],
            3,
        );
        // range_check_9_9_e: relation 4, 4 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_e_0.clone(),
                lookup_data.range_check_9_9_e_1.clone(),
                lookup_data.range_check_9_9_e_2.clone(),
                lookup_data.range_check_9_9_e_3.clone(),
            ],
            4,
        );
        // range_check_9_9_f: relation 5, 4 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_f_0.clone(),
                lookup_data.range_check_9_9_f_1.clone(),
                lookup_data.range_check_9_9_f_2.clone(),
                lookup_data.range_check_9_9_f_3.clone(),
            ],
            5,
        );
        // range_check_9_9_g: relation 6, 2 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_g_0.clone(),
                lookup_data.range_check_9_9_g_1.clone(),
            ],
            6,
        );
        // range_check_9_9_h: relation 7, 2 lookups
        range_check_9_9_cuda_state.add_cuda_inputs_for_relation(
            &[
                lookup_data.range_check_9_9_h_0.clone(),
                lookup_data.range_check_9_9_h_1.clone(),
            ],
            7,
        );

        // Add range check 20 inputs via CUDA path with offset correction.
        let rc20_offset = M31(393216);

        // CUDA _h → relation 0
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_h_0.clone(),
                lookup_data.range_check_19_h_1.clone(),
                lookup_data.range_check_19_h_2.clone(),
                lookup_data.range_check_19_h_3.clone(),
            ],
            0,
            rc20_offset,
        );
        // CUDA _0 → relation 1
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_0.clone(),
                lookup_data.range_check_19_1.clone(),
                lookup_data.range_check_19_2.clone(),
                lookup_data.range_check_19_3.clone(),
            ],
            1,
            rc20_offset,
        );
        // CUDA _b → relation 2
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_b_0.clone(),
                lookup_data.range_check_19_b_1.clone(),
                lookup_data.range_check_19_b_2.clone(),
                lookup_data.range_check_19_b_3.clone(),
            ],
            2,
            rc20_offset,
        );
        // CUDA _c → relation 3
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_c_0.clone(),
                lookup_data.range_check_19_c_1.clone(),
                lookup_data.range_check_19_c_2.clone(),
                lookup_data.range_check_19_c_3.clone(),
            ],
            3,
            rc20_offset,
        );
        // CUDA _d → relation 4
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_d_0.clone(),
                lookup_data.range_check_19_d_1.clone(),
                lookup_data.range_check_19_d_2.clone(),
            ],
            4,
            rc20_offset,
        );
        // CUDA _e → relation 5
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_e_0.clone(),
                lookup_data.range_check_19_e_1.clone(),
                lookup_data.range_check_19_e_2.clone(),
            ],
            5,
            rc20_offset,
        );
        // CUDA _f → relation 6
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_f_0.clone(),
                lookup_data.range_check_19_f_1.clone(),
                lookup_data.range_check_19_f_2.clone(),
            ],
            6,
            rc20_offset,
        );
        // CUDA _g → relation 7
        range_check_20_cuda_state.add_cuda_inputs_for_relation_with_offset(
            &[
                lookup_data.range_check_19_g_0.clone(),
                lookup_data.range_check_19_g_1.clone(),
                lookup_data.range_check_19_g_2.clone(),
            ],
            7,
            rc20_offset,
        );

        range_check_18_cuda_state
            .add_cuda_inputs_for_relation(&[lookup_data.range_check_18_0.clone()], 0);

        range_check_11_cuda_state.add_cuda_inputs(&[lookup_data.range_check_11_0.clone()]);

        // CUDA kernel generates 244 columns but AIR expects 243. The extra column is at
        // CUDA index 227 (an intermediate partial_limb_msb from res_limbs[3] that the
        // current AIR doesn't use). Extract all columns except 227.
        const EXTRA_COL_IDX: usize = 227;
        let mut all_evals: Vec<Option<_>> =
            trace.to_evals().to_vec().into_iter().map(Some).collect();
        let selected_evals: Vec<_> = (0..CUDA_N_TRACE_COLUMNS)
            .filter(|&i| i != EXTRA_COL_IDX)
            .map(|i| all_evals[i].take().unwrap())
            .collect();
        assert_eq!(selected_evals.len(), N_TRACE_COLUMNS);
        tree_builder.extend_evals(selected_evals);

        (
            Claim { log_size },
            CudaInteractionClaimGenerator {
                n_rows: self.n_rows,
                log_size,
                lookup_data,
            },
        )
    }
}

struct CudaSubComponentInputs {
    verify_instruction: [verify_instruction_cuda::CudaPackedInputType; 1],
    memory_address_to_id: [memory_address_to_id_cuda::CudaPackedInputType; 3],
    memory_id_to_big: [memory_id_to_big_cuda::CudaPackedInputType; 3],
}

#[allow(unused_variables)]
fn write_trace_cuda(
    n_rows: usize,
    inputs: CudaPackedInputs,
    memory_address_to_id_state: &memory_address_to_id_cuda::CudaClaimGenerator,
    memory_id_to_big_state: &memory_id_to_big_cuda::CudaClaimGenerator,
    verify_instruction_state: &verify_instruction_cuda::CudaClaimGenerator,
) -> (
    CudaComponentTrace<CUDA_N_TRACE_COLUMNS>,
    CudaLookupData,
    CudaSubComponentInputs,
) {
    let log_size = inputs[0].size.ilog2();
    let (trace, lookup_data, sub_component_inputs) = unsafe {
        (
            CudaComponentTrace::<CUDA_N_TRACE_COLUMNS>::uninitialized(log_size),
            CudaLookupData {
                // memory_address_to_id: 3 lookups × 2 fields
                memory_address_to_id_0: init_lookup_array!(log_size),
                memory_address_to_id_1: init_lookup_array!(log_size),
                memory_address_to_id_2: init_lookup_array!(log_size),
                // memory_id_to_big: 3 lookups × 29 fields
                memory_id_to_big_0: init_lookup_array!(log_size),
                memory_id_to_big_1: init_lookup_array!(log_size),
                memory_id_to_big_2: init_lookup_array!(log_size),
                // opcodes: 2 lookups × 3 fields
                opcodes_0: init_lookup_array!(log_size),
                opcodes_1: init_lookup_array!(log_size),
                // range_check_9_9: 4 lookups × 2 fields
                range_check_9_9_0: init_lookup_array!(log_size),
                range_check_9_9_1: init_lookup_array!(log_size),
                range_check_9_9_2: init_lookup_array!(log_size),
                range_check_9_9_3: init_lookup_array!(log_size),
                // range_check_9_9_b: 4 lookups × 2 fields
                range_check_9_9_b_0: init_lookup_array!(log_size),
                range_check_9_9_b_1: init_lookup_array!(log_size),
                range_check_9_9_b_2: init_lookup_array!(log_size),
                range_check_9_9_b_3: init_lookup_array!(log_size),
                // range_check_9_9_c: 4 lookups × 2 fields
                range_check_9_9_c_0: init_lookup_array!(log_size),
                range_check_9_9_c_1: init_lookup_array!(log_size),
                range_check_9_9_c_2: init_lookup_array!(log_size),
                range_check_9_9_c_3: init_lookup_array!(log_size),
                // range_check_9_9_d: 4 lookups × 2 fields
                range_check_9_9_d_0: init_lookup_array!(log_size),
                range_check_9_9_d_1: init_lookup_array!(log_size),
                range_check_9_9_d_2: init_lookup_array!(log_size),
                range_check_9_9_d_3: init_lookup_array!(log_size),
                // range_check_9_9_e: 4 lookups × 2 fields
                range_check_9_9_e_0: init_lookup_array!(log_size),
                range_check_9_9_e_1: init_lookup_array!(log_size),
                range_check_9_9_e_2: init_lookup_array!(log_size),
                range_check_9_9_e_3: init_lookup_array!(log_size),
                // range_check_9_9_f: 4 lookups × 2 fields
                range_check_9_9_f_0: init_lookup_array!(log_size),
                range_check_9_9_f_1: init_lookup_array!(log_size),
                range_check_9_9_f_2: init_lookup_array!(log_size),
                range_check_9_9_f_3: init_lookup_array!(log_size),
                // range_check_9_9_g: 2 lookups × 2 fields
                range_check_9_9_g_0: init_lookup_array!(log_size),
                range_check_9_9_g_1: init_lookup_array!(log_size),
                // range_check_9_9_h: 2 lookups × 2 fields
                range_check_9_9_h_0: init_lookup_array!(log_size),
                range_check_9_9_h_1: init_lookup_array!(log_size),
                // range_check_19: 4 lookups × 1 field
                range_check_19_0: init_lookup_array!(log_size),
                range_check_19_1: init_lookup_array!(log_size),
                range_check_19_2: init_lookup_array!(log_size),
                range_check_19_3: init_lookup_array!(log_size),
                // range_check_19_b: 4 lookups × 1 field
                range_check_19_b_0: init_lookup_array!(log_size),
                range_check_19_b_1: init_lookup_array!(log_size),
                range_check_19_b_2: init_lookup_array!(log_size),
                range_check_19_b_3: init_lookup_array!(log_size),
                // range_check_19_c: 4 lookups × 1 field
                range_check_19_c_0: init_lookup_array!(log_size),
                range_check_19_c_1: init_lookup_array!(log_size),
                range_check_19_c_2: init_lookup_array!(log_size),
                range_check_19_c_3: init_lookup_array!(log_size),
                // range_check_19_d: 3 lookups × 1 field
                range_check_19_d_0: init_lookup_array!(log_size),
                range_check_19_d_1: init_lookup_array!(log_size),
                range_check_19_d_2: init_lookup_array!(log_size),
                // range_check_19_e: 3 lookups × 1 field
                range_check_19_e_0: init_lookup_array!(log_size),
                range_check_19_e_1: init_lookup_array!(log_size),
                range_check_19_e_2: init_lookup_array!(log_size),
                // range_check_19_f: 3 lookups × 1 field
                range_check_19_f_0: init_lookup_array!(log_size),
                range_check_19_f_1: init_lookup_array!(log_size),
                range_check_19_f_2: init_lookup_array!(log_size),
                // range_check_19_g: 3 lookups × 1 field
                range_check_19_g_0: init_lookup_array!(log_size),
                range_check_19_g_1: init_lookup_array!(log_size),
                range_check_19_g_2: init_lookup_array!(log_size),
                // range_check_19_h: 4 lookups × 1 field
                range_check_19_h_0: init_lookup_array!(log_size),
                range_check_19_h_1: init_lookup_array!(log_size),
                range_check_19_h_2: init_lookup_array!(log_size),
                range_check_19_h_3: init_lookup_array!(log_size),
                // range_check_18: 1 lookup × 1 field
                range_check_18_0: init_lookup_array!(log_size),
                // range_check_11: 1 lookup × 1 field
                range_check_11_0: init_lookup_array!(log_size),
                // verify_instruction: 1 lookup × 7 fields
                verify_instruction_0: init_lookup_array!(log_size),
            },
            CudaSubComponentInputs {
                verify_instruction: init_subcomponent_basefield_array!(log_size),
                memory_address_to_id: init_subcomponent_basefield_array!(log_size),
                memory_id_to_big: init_subcomponent_basefield_array!(log_size),
            },
        )
    };

    let traces_vec = trace.data.iter().map(|c| c.device_ptr).collect_vec();

    // Collect all lookup pointers
    let lookup_memory_address_to_id_0 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_0);
    let lookup_memory_address_to_id_1 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_1);
    let lookup_memory_address_to_id_2 = collect_lookup_ptrs!(lookup_data, memory_address_to_id_2);
    let lookup_memory_id_to_big_0 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_0);
    let lookup_memory_id_to_big_1 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_1);
    let lookup_memory_id_to_big_2 = collect_lookup_ptrs!(lookup_data, memory_id_to_big_2);
    let lookup_opcodes_0 = collect_lookup_ptrs!(lookup_data, opcodes_0);
    let lookup_opcodes_1 = collect_lookup_ptrs!(lookup_data, opcodes_1);

    let lookup_range_check_9_9_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_0);
    let lookup_range_check_9_9_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_1);
    let lookup_range_check_9_9_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_2);
    let lookup_range_check_9_9_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_3);

    let lookup_range_check_9_9_b_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_0);
    let lookup_range_check_9_9_b_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_1);
    let lookup_range_check_9_9_b_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_2);
    let lookup_range_check_9_9_b_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_b_3);

    let lookup_range_check_9_9_c_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_0);
    let lookup_range_check_9_9_c_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_1);
    let lookup_range_check_9_9_c_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_2);
    let lookup_range_check_9_9_c_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_c_3);

    let lookup_range_check_9_9_d_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_0);
    let lookup_range_check_9_9_d_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_1);
    let lookup_range_check_9_9_d_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_2);
    let lookup_range_check_9_9_d_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_d_3);

    let lookup_range_check_9_9_e_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_0);
    let lookup_range_check_9_9_e_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_1);
    let lookup_range_check_9_9_e_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_2);
    let lookup_range_check_9_9_e_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_e_3);

    let lookup_range_check_9_9_f_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_0);
    let lookup_range_check_9_9_f_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_1);
    let lookup_range_check_9_9_f_2 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_2);
    let lookup_range_check_9_9_f_3 = collect_lookup_ptrs!(lookup_data, range_check_9_9_f_3);

    let lookup_range_check_9_9_g_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_g_0);
    let lookup_range_check_9_9_g_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_g_1);

    let lookup_range_check_9_9_h_0 = collect_lookup_ptrs!(lookup_data, range_check_9_9_h_0);
    let lookup_range_check_9_9_h_1 = collect_lookup_ptrs!(lookup_data, range_check_9_9_h_1);

    let lookup_range_check_19_0 = collect_lookup_ptrs!(lookup_data, range_check_19_0);
    let lookup_range_check_19_1 = collect_lookup_ptrs!(lookup_data, range_check_19_1);
    let lookup_range_check_19_2 = collect_lookup_ptrs!(lookup_data, range_check_19_2);
    let lookup_range_check_19_3 = collect_lookup_ptrs!(lookup_data, range_check_19_3);

    let lookup_range_check_19_b_0 = collect_lookup_ptrs!(lookup_data, range_check_19_b_0);
    let lookup_range_check_19_b_1 = collect_lookup_ptrs!(lookup_data, range_check_19_b_1);
    let lookup_range_check_19_b_2 = collect_lookup_ptrs!(lookup_data, range_check_19_b_2);
    let lookup_range_check_19_b_3 = collect_lookup_ptrs!(lookup_data, range_check_19_b_3);

    let lookup_range_check_19_c_0 = collect_lookup_ptrs!(lookup_data, range_check_19_c_0);
    let lookup_range_check_19_c_1 = collect_lookup_ptrs!(lookup_data, range_check_19_c_1);
    let lookup_range_check_19_c_2 = collect_lookup_ptrs!(lookup_data, range_check_19_c_2);
    let lookup_range_check_19_c_3 = collect_lookup_ptrs!(lookup_data, range_check_19_c_3);

    let lookup_range_check_19_d_0 = collect_lookup_ptrs!(lookup_data, range_check_19_d_0);
    let lookup_range_check_19_d_1 = collect_lookup_ptrs!(lookup_data, range_check_19_d_1);
    let lookup_range_check_19_d_2 = collect_lookup_ptrs!(lookup_data, range_check_19_d_2);

    let lookup_range_check_19_e_0 = collect_lookup_ptrs!(lookup_data, range_check_19_e_0);
    let lookup_range_check_19_e_1 = collect_lookup_ptrs!(lookup_data, range_check_19_e_1);
    let lookup_range_check_19_e_2 = collect_lookup_ptrs!(lookup_data, range_check_19_e_2);

    let lookup_range_check_19_f_0 = collect_lookup_ptrs!(lookup_data, range_check_19_f_0);
    let lookup_range_check_19_f_1 = collect_lookup_ptrs!(lookup_data, range_check_19_f_1);
    let lookup_range_check_19_f_2 = collect_lookup_ptrs!(lookup_data, range_check_19_f_2);

    let lookup_range_check_19_g_0 = collect_lookup_ptrs!(lookup_data, range_check_19_g_0);
    let lookup_range_check_19_g_1 = collect_lookup_ptrs!(lookup_data, range_check_19_g_1);
    let lookup_range_check_19_g_2 = collect_lookup_ptrs!(lookup_data, range_check_19_g_2);

    let lookup_range_check_19_h_0 = collect_lookup_ptrs!(lookup_data, range_check_19_h_0);
    let lookup_range_check_19_h_1 = collect_lookup_ptrs!(lookup_data, range_check_19_h_1);
    let lookup_range_check_19_h_2 = collect_lookup_ptrs!(lookup_data, range_check_19_h_2);
    let lookup_range_check_19_h_3 = collect_lookup_ptrs!(lookup_data, range_check_19_h_3);

    let lookup_range_check_18_0 = collect_lookup_ptrs!(lookup_data, range_check_18_0);
    let lookup_range_check_11_0 = collect_lookup_ptrs!(lookup_data, range_check_11_0);
    let lookup_verify_instruction_0 = collect_lookup_ptrs!(lookup_data, verify_instruction_0);

    let sub_component_inputs_verify_instruction_vec =
        collect_sub_input_ptrs!(sub_component_inputs, verify_instruction);
    let sub_component_inputs_memory_address_to_id_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_address_to_id);
    let sub_component_inputs_memory_id_to_big_vec =
        collect_sub_input_ptrs!(sub_component_inputs, memory_id_to_big);

    let opcodes_input_vec = collect_input_ptrs!(inputs);
    let memory_id_to_big_transposed_big_values_vec = memory_id_to_big_state
        .transposed_big_values
        .iter()
        .map(|x| x.device_ptr)
        .collect_vec();

    unsafe {
        bindings_airs::generate_generic_opcode_traces(
            traces_vec.as_ptr(),
            // memory_address_to_id (3 lookups)
            lookup_memory_address_to_id_0.as_ptr(),
            lookup_memory_address_to_id_1.as_ptr(),
            lookup_memory_address_to_id_2.as_ptr(),
            // memory_id_to_big (3 lookups)
            lookup_memory_id_to_big_0.as_ptr(),
            lookup_memory_id_to_big_1.as_ptr(),
            lookup_memory_id_to_big_2.as_ptr(),
            // opcodes (2 lookups)
            lookup_opcodes_0.as_ptr(),
            lookup_opcodes_1.as_ptr(),
            // range_check_9_9 (4 lookups)
            lookup_range_check_9_9_0.as_ptr(),
            lookup_range_check_9_9_1.as_ptr(),
            lookup_range_check_9_9_2.as_ptr(),
            lookup_range_check_9_9_3.as_ptr(),
            // range_check_9_9_b (4 lookups)
            lookup_range_check_9_9_b_0.as_ptr(),
            lookup_range_check_9_9_b_1.as_ptr(),
            lookup_range_check_9_9_b_2.as_ptr(),
            lookup_range_check_9_9_b_3.as_ptr(),
            // range_check_9_9_c (4 lookups)
            lookup_range_check_9_9_c_0.as_ptr(),
            lookup_range_check_9_9_c_1.as_ptr(),
            lookup_range_check_9_9_c_2.as_ptr(),
            lookup_range_check_9_9_c_3.as_ptr(),
            // range_check_9_9_d (4 lookups)
            lookup_range_check_9_9_d_0.as_ptr(),
            lookup_range_check_9_9_d_1.as_ptr(),
            lookup_range_check_9_9_d_2.as_ptr(),
            lookup_range_check_9_9_d_3.as_ptr(),
            // range_check_9_9_e (4 lookups)
            lookup_range_check_9_9_e_0.as_ptr(),
            lookup_range_check_9_9_e_1.as_ptr(),
            lookup_range_check_9_9_e_2.as_ptr(),
            lookup_range_check_9_9_e_3.as_ptr(),
            // range_check_9_9_f (4 lookups)
            lookup_range_check_9_9_f_0.as_ptr(),
            lookup_range_check_9_9_f_1.as_ptr(),
            lookup_range_check_9_9_f_2.as_ptr(),
            lookup_range_check_9_9_f_3.as_ptr(),
            // range_check_9_9_g (2 lookups)
            lookup_range_check_9_9_g_0.as_ptr(),
            lookup_range_check_9_9_g_1.as_ptr(),
            // range_check_9_9_h (2 lookups)
            lookup_range_check_9_9_h_0.as_ptr(),
            lookup_range_check_9_9_h_1.as_ptr(),
            // range_check_19 (4 lookups)
            lookup_range_check_19_0.as_ptr(),
            lookup_range_check_19_1.as_ptr(),
            lookup_range_check_19_2.as_ptr(),
            lookup_range_check_19_3.as_ptr(),
            // range_check_19_b (4 lookups)
            lookup_range_check_19_b_0.as_ptr(),
            lookup_range_check_19_b_1.as_ptr(),
            lookup_range_check_19_b_2.as_ptr(),
            lookup_range_check_19_b_3.as_ptr(),
            // range_check_19_c (4 lookups)
            lookup_range_check_19_c_0.as_ptr(),
            lookup_range_check_19_c_1.as_ptr(),
            lookup_range_check_19_c_2.as_ptr(),
            lookup_range_check_19_c_3.as_ptr(),
            // range_check_19_d (3 lookups)
            lookup_range_check_19_d_0.as_ptr(),
            lookup_range_check_19_d_1.as_ptr(),
            lookup_range_check_19_d_2.as_ptr(),
            // range_check_19_e (3 lookups)
            lookup_range_check_19_e_0.as_ptr(),
            lookup_range_check_19_e_1.as_ptr(),
            lookup_range_check_19_e_2.as_ptr(),
            // range_check_19_f (3 lookups)
            lookup_range_check_19_f_0.as_ptr(),
            lookup_range_check_19_f_1.as_ptr(),
            lookup_range_check_19_f_2.as_ptr(),
            // range_check_19_g (3 lookups)
            lookup_range_check_19_g_0.as_ptr(),
            lookup_range_check_19_g_1.as_ptr(),
            lookup_range_check_19_g_2.as_ptr(),
            // range_check_19_h (4 lookups)
            lookup_range_check_19_h_0.as_ptr(),
            lookup_range_check_19_h_1.as_ptr(),
            lookup_range_check_19_h_2.as_ptr(),
            lookup_range_check_19_h_3.as_ptr(),
            // range_check_18 (1 lookup)
            lookup_range_check_18_0.as_ptr(),
            // range_check_11 (1 lookup)
            lookup_range_check_11_0.as_ptr(),
            // verify_instruction (1 lookup)
            lookup_verify_instruction_0.as_ptr(),
            // Sub-component inputs
            sub_component_inputs_verify_instruction_vec.as_ptr(),
            sub_component_inputs_memory_address_to_id_vec.as_ptr(),
            sub_component_inputs_memory_id_to_big_vec.as_ptr(),
            // Opcode inputs
            opcodes_input_vec.as_ptr(),
            // Memory lookup tables
            memory_address_to_id_state.address_to_raw_id.device_ptr,
            memory_id_to_big_transposed_big_values_vec.as_ptr(),
            memory_id_to_big_state.small_values.device_ptr,
            n_rows as u32,
            log_size,
        );
    }
    (trace, lookup_data, sub_component_inputs)
}

struct CudaLookupData {
    // memory_address_to_id: 3 lookups × 2 fields
    memory_address_to_id_0: [BaseFieldVec; 2],
    memory_address_to_id_1: [BaseFieldVec; 2],
    memory_address_to_id_2: [BaseFieldVec; 2],
    // memory_id_to_big: 3 lookups × 29 fields
    memory_id_to_big_0: [BaseFieldVec; 29],
    memory_id_to_big_1: [BaseFieldVec; 29],
    memory_id_to_big_2: [BaseFieldVec; 29],
    // opcodes: 2 lookups × 3 fields
    opcodes_0: [BaseFieldVec; 3],
    opcodes_1: [BaseFieldVec; 3],
    // range_check_9_9: 4 lookups × 2 fields
    range_check_9_9_0: [BaseFieldVec; 2],
    range_check_9_9_1: [BaseFieldVec; 2],
    range_check_9_9_2: [BaseFieldVec; 2],
    range_check_9_9_3: [BaseFieldVec; 2],
    // range_check_9_9_b: 4 lookups × 2 fields
    range_check_9_9_b_0: [BaseFieldVec; 2],
    range_check_9_9_b_1: [BaseFieldVec; 2],
    range_check_9_9_b_2: [BaseFieldVec; 2],
    range_check_9_9_b_3: [BaseFieldVec; 2],
    // range_check_9_9_c: 4 lookups × 2 fields
    range_check_9_9_c_0: [BaseFieldVec; 2],
    range_check_9_9_c_1: [BaseFieldVec; 2],
    range_check_9_9_c_2: [BaseFieldVec; 2],
    range_check_9_9_c_3: [BaseFieldVec; 2],
    // range_check_9_9_d: 4 lookups × 2 fields
    range_check_9_9_d_0: [BaseFieldVec; 2],
    range_check_9_9_d_1: [BaseFieldVec; 2],
    range_check_9_9_d_2: [BaseFieldVec; 2],
    range_check_9_9_d_3: [BaseFieldVec; 2],
    // range_check_9_9_e: 4 lookups × 2 fields
    range_check_9_9_e_0: [BaseFieldVec; 2],
    range_check_9_9_e_1: [BaseFieldVec; 2],
    range_check_9_9_e_2: [BaseFieldVec; 2],
    range_check_9_9_e_3: [BaseFieldVec; 2],
    // range_check_9_9_f: 4 lookups × 2 fields
    range_check_9_9_f_0: [BaseFieldVec; 2],
    range_check_9_9_f_1: [BaseFieldVec; 2],
    range_check_9_9_f_2: [BaseFieldVec; 2],
    range_check_9_9_f_3: [BaseFieldVec; 2],
    // range_check_9_9_g: 2 lookups × 2 fields
    range_check_9_9_g_0: [BaseFieldVec; 2],
    range_check_9_9_g_1: [BaseFieldVec; 2],
    // range_check_9_9_h: 2 lookups × 2 fields
    range_check_9_9_h_0: [BaseFieldVec; 2],
    range_check_9_9_h_1: [BaseFieldVec; 2],
    // range_check_19: 4 lookups × 1 field
    range_check_19_0: [BaseFieldVec; 1],
    range_check_19_1: [BaseFieldVec; 1],
    range_check_19_2: [BaseFieldVec; 1],
    range_check_19_3: [BaseFieldVec; 1],
    // range_check_19_b: 4 lookups × 1 field
    range_check_19_b_0: [BaseFieldVec; 1],
    range_check_19_b_1: [BaseFieldVec; 1],
    range_check_19_b_2: [BaseFieldVec; 1],
    range_check_19_b_3: [BaseFieldVec; 1],
    // range_check_19_c: 4 lookups × 1 field
    range_check_19_c_0: [BaseFieldVec; 1],
    range_check_19_c_1: [BaseFieldVec; 1],
    range_check_19_c_2: [BaseFieldVec; 1],
    range_check_19_c_3: [BaseFieldVec; 1],
    // range_check_19_d: 3 lookups × 1 field
    range_check_19_d_0: [BaseFieldVec; 1],
    range_check_19_d_1: [BaseFieldVec; 1],
    range_check_19_d_2: [BaseFieldVec; 1],
    // range_check_19_e: 3 lookups × 1 field
    range_check_19_e_0: [BaseFieldVec; 1],
    range_check_19_e_1: [BaseFieldVec; 1],
    range_check_19_e_2: [BaseFieldVec; 1],
    // range_check_19_f: 3 lookups × 1 field
    range_check_19_f_0: [BaseFieldVec; 1],
    range_check_19_f_1: [BaseFieldVec; 1],
    range_check_19_f_2: [BaseFieldVec; 1],
    // range_check_19_g: 3 lookups × 1 field
    range_check_19_g_0: [BaseFieldVec; 1],
    range_check_19_g_1: [BaseFieldVec; 1],
    range_check_19_g_2: [BaseFieldVec; 1],
    // range_check_19_h: 4 lookups × 1 field
    range_check_19_h_0: [BaseFieldVec; 1],
    range_check_19_h_1: [BaseFieldVec; 1],
    range_check_19_h_2: [BaseFieldVec; 1],
    range_check_19_h_3: [BaseFieldVec; 1],
    // range_check_18: 1 lookup × 1 field
    range_check_18_0: [BaseFieldVec; 1],
    // range_check_11: 1 lookup × 1 field
    range_check_11_0: [BaseFieldVec; 1],
    // verify_instruction: 1 lookup × 7 fields
    verify_instruction_0: [BaseFieldVec; 7],
}

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
        let trace_log_size = self.log_size;

        let cuda_claimed_sum = BaseFieldVec::new_uninitialized(4);

        let interaction_trace = (0..4 * N_INTERACTION_TRACE_COLUMNS)
            .map(|_| Col::<CudaBackend, BaseField>::zeros(1 << trace_log_size))
            .collect_vec();

        // Collect all lookup pointers for interaction trace generation
        let lookup_memory_address_to_id_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_0);
        let lookup_memory_address_to_id_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_1);
        let lookup_memory_address_to_id_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_address_to_id_2);
        let lookup_memory_id_to_big_0_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_0);
        let lookup_memory_id_to_big_1_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_1);
        let lookup_memory_id_to_big_2_vec =
            collect_lookup_ptrs!(self.lookup_data, memory_id_to_big_2);
        let lookup_opcodes_0_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_0);
        let lookup_opcodes_1_vec = collect_lookup_ptrs!(self.lookup_data, opcodes_1);

        let lookup_range_check_9_9_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_0);
        let lookup_range_check_9_9_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_1);
        let lookup_range_check_9_9_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_2);
        let lookup_range_check_9_9_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_3);

        let lookup_range_check_9_9_b_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_0);
        let lookup_range_check_9_9_b_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_1);
        let lookup_range_check_9_9_b_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_2);
        let lookup_range_check_9_9_b_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_b_3);

        let lookup_range_check_9_9_c_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_0);
        let lookup_range_check_9_9_c_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_1);
        let lookup_range_check_9_9_c_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_2);
        let lookup_range_check_9_9_c_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_c_3);

        let lookup_range_check_9_9_d_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_0);
        let lookup_range_check_9_9_d_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_1);
        let lookup_range_check_9_9_d_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_2);
        let lookup_range_check_9_9_d_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_d_3);

        let lookup_range_check_9_9_e_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_0);
        let lookup_range_check_9_9_e_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_1);
        let lookup_range_check_9_9_e_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_2);
        let lookup_range_check_9_9_e_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_e_3);

        let lookup_range_check_9_9_f_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_0);
        let lookup_range_check_9_9_f_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_1);
        let lookup_range_check_9_9_f_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_2);
        let lookup_range_check_9_9_f_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_f_3);

        let lookup_range_check_9_9_g_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_g_0);
        let lookup_range_check_9_9_g_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_g_1);

        let lookup_range_check_9_9_h_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_h_0);
        let lookup_range_check_9_9_h_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_9_9_h_1);

        let lookup_range_check_19_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_0);
        let lookup_range_check_19_1_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_1);
        let lookup_range_check_19_2_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_2);
        let lookup_range_check_19_3_vec = collect_lookup_ptrs!(self.lookup_data, range_check_19_3);

        let lookup_range_check_19_b_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_0);
        let lookup_range_check_19_b_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_1);
        let lookup_range_check_19_b_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_2);
        let lookup_range_check_19_b_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_b_3);

        let lookup_range_check_19_c_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_0);
        let lookup_range_check_19_c_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_1);
        let lookup_range_check_19_c_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_2);
        let lookup_range_check_19_c_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_c_3);

        let lookup_range_check_19_d_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_d_0);
        let lookup_range_check_19_d_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_d_1);
        let lookup_range_check_19_d_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_d_2);

        let lookup_range_check_19_e_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_e_0);
        let lookup_range_check_19_e_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_e_1);
        let lookup_range_check_19_e_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_e_2);

        let lookup_range_check_19_f_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_f_0);
        let lookup_range_check_19_f_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_f_1);
        let lookup_range_check_19_f_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_f_2);

        let lookup_range_check_19_g_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_g_0);
        let lookup_range_check_19_g_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_g_1);
        let lookup_range_check_19_g_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_g_2);

        let lookup_range_check_19_h_0_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_0);
        let lookup_range_check_19_h_1_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_1);
        let lookup_range_check_19_h_2_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_2);
        let lookup_range_check_19_h_3_vec =
            collect_lookup_ptrs!(self.lookup_data, range_check_19_h_3);

        let lookup_range_check_18_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_18_0);
        let lookup_range_check_11_0_vec = collect_lookup_ptrs!(self.lookup_data, range_check_11_0);
        let lookup_verify_instruction_0_vec =
            collect_lookup_ptrs!(self.lookup_data, verify_instruction_0);

        let interaction_trace_vec = interaction_trace
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();

        unsafe {
            use crate::witness::components_cuda::cuda_lookup_helper::*;
            let mod_addr =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ADDRESS_TO_ID_RELATION_ID);
            let mod_mem =
                create_modified_lookup_for_cuda(lookup_elements, MEMORY_ID_TO_BIG_RELATION_ID);
            let mod_ops = create_modified_lookup_for_cuda(lookup_elements, OPCODES_RELATION_ID);
            let mod_rc99 = create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[0]);
            let mod_rc99b =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[1]);
            let mod_rc99c =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[2]);
            let mod_rc99d =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[3]);
            let mod_rc99e =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[4]);
            let mod_rc99f =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[5]);
            let mod_rc99g =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[6]);
            let mod_rc99h =
                create_modified_lookup_for_cuda(lookup_elements, RC_9_9_RELATION_IDS[7]);
            // CUDA rc_19 arrays contain values with offset 131072 (after _h_0 normalization).
            // SIMD rc_20 expects values with offset 524288. Correction: 393216.
            // Also, CUDA naming is shifted from SIMD: CUDA _h = SIMD relation 0, etc.
            // Each modified lookup element must use the CORRECT relation constant for the
            // variables actually stored in that CUDA array.
            let rc20_offset_corr = M31(393216); // 524288 - 131072
                                                // CUDA _h stores k_col,carry7,carry15,carry23 → SIMD relation 0 constant
            let mod_rc19h = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _0 stores carry0,carry8,carry16,carry24 → SIMD relation 1 constant
            let mod_rc19 = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_B_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _b stores carry1,carry9,carry17,carry25 → SIMD relation 2 constant
            let mod_rc19b = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_C_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _c → SIMD relation 3
            let mod_rc19c = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_D_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _d → SIMD relation 4
            let mod_rc19d = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_E_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _e → SIMD relation 5
            let mod_rc19e = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_F_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _f → SIMD relation 6
            let mod_rc19f = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_G_RELATION_ID,
                rc20_offset_corr,
            );
            // CUDA _g → SIMD relation 7
            let mod_rc19g = create_modified_lookup_for_cuda_with_offset(
                lookup_elements,
                RC_19_H_RELATION_ID,
                rc20_offset_corr,
            );
            let mod_rc18 = create_modified_lookup_for_cuda(lookup_elements, RC_18_RELATION_ID);
            let mod_rc11 = create_modified_lookup_for_cuda(lookup_elements, RC_11_RELATION_ID);
            let mod_vi =
                create_modified_lookup_for_cuda(lookup_elements, VERIFY_INSTRUCTION_RELATION_ID);

            bindings_airs::generate_generic_opcode_interaction_traces(
                // Relation pointers (22 relations)
                &mod_addr as *const _ as *mut std::os::raw::c_void,
                &mod_mem as *const _ as *mut std::os::raw::c_void,
                &mod_ops as *const _ as *mut std::os::raw::c_void,
                &mod_rc99 as *const _ as *mut std::os::raw::c_void,
                &mod_rc99b as *const _ as *mut std::os::raw::c_void,
                &mod_rc99c as *const _ as *mut std::os::raw::c_void,
                &mod_rc99d as *const _ as *mut std::os::raw::c_void,
                &mod_rc99e as *const _ as *mut std::os::raw::c_void,
                &mod_rc99f as *const _ as *mut std::os::raw::c_void,
                &mod_rc99g as *const _ as *mut std::os::raw::c_void,
                &mod_rc99h as *const _ as *mut std::os::raw::c_void,
                &mod_rc19 as *const _ as *mut std::os::raw::c_void,
                &mod_rc19b as *const _ as *mut std::os::raw::c_void,
                &mod_rc19c as *const _ as *mut std::os::raw::c_void,
                &mod_rc19d as *const _ as *mut std::os::raw::c_void,
                &mod_rc19e as *const _ as *mut std::os::raw::c_void,
                &mod_rc19f as *const _ as *mut std::os::raw::c_void,
                &mod_rc19g as *const _ as *mut std::os::raw::c_void,
                &mod_rc19h as *const _ as *mut std::os::raw::c_void,
                &mod_rc18 as *const _ as *mut std::os::raw::c_void,
                &mod_rc11 as *const _ as *mut std::os::raw::c_void,
                &mod_vi as *const _ as *mut std::os::raw::c_void,
                // All lookup data (67 lookups)
                lookup_memory_address_to_id_0_vec.as_ptr(),
                lookup_memory_address_to_id_1_vec.as_ptr(),
                lookup_memory_address_to_id_2_vec.as_ptr(),
                lookup_memory_id_to_big_0_vec.as_ptr(),
                lookup_memory_id_to_big_1_vec.as_ptr(),
                lookup_memory_id_to_big_2_vec.as_ptr(),
                lookup_opcodes_0_vec.as_ptr(),
                lookup_opcodes_1_vec.as_ptr(),
                lookup_range_check_9_9_0_vec.as_ptr(),
                lookup_range_check_9_9_1_vec.as_ptr(),
                lookup_range_check_9_9_2_vec.as_ptr(),
                lookup_range_check_9_9_3_vec.as_ptr(),
                lookup_range_check_9_9_b_0_vec.as_ptr(),
                lookup_range_check_9_9_b_1_vec.as_ptr(),
                lookup_range_check_9_9_b_2_vec.as_ptr(),
                lookup_range_check_9_9_b_3_vec.as_ptr(),
                lookup_range_check_9_9_c_0_vec.as_ptr(),
                lookup_range_check_9_9_c_1_vec.as_ptr(),
                lookup_range_check_9_9_c_2_vec.as_ptr(),
                lookup_range_check_9_9_c_3_vec.as_ptr(),
                lookup_range_check_9_9_d_0_vec.as_ptr(),
                lookup_range_check_9_9_d_1_vec.as_ptr(),
                lookup_range_check_9_9_d_2_vec.as_ptr(),
                lookup_range_check_9_9_d_3_vec.as_ptr(),
                lookup_range_check_9_9_e_0_vec.as_ptr(),
                lookup_range_check_9_9_e_1_vec.as_ptr(),
                lookup_range_check_9_9_e_2_vec.as_ptr(),
                lookup_range_check_9_9_e_3_vec.as_ptr(),
                lookup_range_check_9_9_f_0_vec.as_ptr(),
                lookup_range_check_9_9_f_1_vec.as_ptr(),
                lookup_range_check_9_9_f_2_vec.as_ptr(),
                lookup_range_check_9_9_f_3_vec.as_ptr(),
                lookup_range_check_9_9_g_0_vec.as_ptr(),
                lookup_range_check_9_9_g_1_vec.as_ptr(),
                lookup_range_check_9_9_h_0_vec.as_ptr(),
                lookup_range_check_9_9_h_1_vec.as_ptr(),
                lookup_range_check_19_0_vec.as_ptr(),
                lookup_range_check_19_1_vec.as_ptr(),
                lookup_range_check_19_2_vec.as_ptr(),
                lookup_range_check_19_3_vec.as_ptr(),
                lookup_range_check_19_b_0_vec.as_ptr(),
                lookup_range_check_19_b_1_vec.as_ptr(),
                lookup_range_check_19_b_2_vec.as_ptr(),
                lookup_range_check_19_b_3_vec.as_ptr(),
                lookup_range_check_19_c_0_vec.as_ptr(),
                lookup_range_check_19_c_1_vec.as_ptr(),
                lookup_range_check_19_c_2_vec.as_ptr(),
                lookup_range_check_19_c_3_vec.as_ptr(),
                lookup_range_check_19_d_0_vec.as_ptr(),
                lookup_range_check_19_d_1_vec.as_ptr(),
                lookup_range_check_19_d_2_vec.as_ptr(),
                lookup_range_check_19_e_0_vec.as_ptr(),
                lookup_range_check_19_e_1_vec.as_ptr(),
                lookup_range_check_19_e_2_vec.as_ptr(),
                lookup_range_check_19_f_0_vec.as_ptr(),
                lookup_range_check_19_f_1_vec.as_ptr(),
                lookup_range_check_19_f_2_vec.as_ptr(),
                lookup_range_check_19_g_0_vec.as_ptr(),
                lookup_range_check_19_g_1_vec.as_ptr(),
                lookup_range_check_19_g_2_vec.as_ptr(),
                lookup_range_check_19_h_0_vec.as_ptr(),
                lookup_range_check_19_h_1_vec.as_ptr(),
                lookup_range_check_19_h_2_vec.as_ptr(),
                lookup_range_check_19_h_3_vec.as_ptr(),
                lookup_range_check_18_0_vec.as_ptr(),
                lookup_range_check_11_0_vec.as_ptr(),
                lookup_verify_instruction_0_vec.as_ptr(),
                self.n_rows as u32,
                trace_log_size as u32,
                interaction_trace_vec.as_ptr(),
                cuda_claimed_sum.device_ptr,
            );
        }

        let claimed_sum_vec = cuda_claimed_sum.to_cpu();
        let claimed_sum = SecureField::from_m31_array([
            claimed_sum_vec[0],
            claimed_sum_vec[1],
            claimed_sum_vec[2],
            claimed_sum_vec[3],
        ]);

        let domain = CanonicCoset::new(trace_log_size).circle_domain();
        let trace: Vec<_> = interaction_trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CudaBackend, M31, BitReversedOrder>::new(domain, eval))
            .collect();

        tree_builder.extend_evals(trace);

        InteractionClaim { claimed_sum }
    }
}
