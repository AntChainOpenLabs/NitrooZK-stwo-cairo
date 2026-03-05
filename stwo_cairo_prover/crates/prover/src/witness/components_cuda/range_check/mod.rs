mod base;
pub use base::{
    CudaInteractionClaimGeneratorCuda, CudaMultiRelationInteractionGen,
    CudaMultiRelationRangeCheckGenerator, CudaRangeCheckGenerator,
};

pub mod rc_11;
pub mod rc_12;
pub mod rc_18;
pub mod rc_20;
pub mod rc_6;
pub mod rc_8;

pub mod rc_4_3;
pub mod rc_4_4;
// rc_5_4 removed in v1.1.0 — no range_check_5_4 component exists
pub mod rc_7_2_5;
pub mod rc_9_9;

pub mod rc_3_3_3_3_3;
pub mod rc_3_6_6_3;
pub mod rc_4_4_4_4;
