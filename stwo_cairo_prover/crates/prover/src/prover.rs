use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use cairo_air::air::CairoComponents;
use cairo_air::claims::lookup_sum;
use cairo_air::relations::CommonLookupElements;
use cairo_air::utils::{serialize_proof_to_file, ProofFormat};
use cairo_air::verifier::{verify_cairo_ex, INTERACTION_POW_BITS};
use cairo_air::{CairoProof, CairoProofCuda, PreProcessedTraceVariant};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleChannel};
use stwo::core::vcs::MerkleHasher;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::BackendForChannel;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{prove_ex, CommitmentSchemeProver, ProvingError};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_serialize::CairoSerialize;
use tracing::{event, span, Level};

use crate::utils::cairo_provers;
use crate::witness::cairo::create_cairo_claim_generator;
use crate::witness::preprocessed_trace::gen_trace;
use crate::witness::utils::witness_trace_cells;

mod json {
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    pub use serde_json::from_str;
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    pub use sonic_rs::from_str;
}

pub(crate) const LOG_MAX_ROWS: u32 = 27;

fn prove_verify_serialize<MC: MerkleChannel>(
    input: ProverInput,
    verify: bool,
    proof_path: &Path,
    proof_format: ProofFormat,
    proof_params: ProverParameters,
) -> Result<()>
where
    SimdBackend: BackendForChannel<MC>,
    MC::H: MerkleHasher + Serialize,
    <MC::H as MerkleHasher>::Hash: CairoSerialize,
{
    let cairo_proof = prove_cairo::<MC>(input, proof_params)?;
    if verify {
        verify_cairo_ex::<MC>(
            cairo_proof.clone().into(),
            proof_params.include_all_preprocessed_columns,
        )?;
    }
    serialize_proof_to_file(&cairo_proof, proof_path, proof_format)?;
    Ok(())
}

pub fn prove_cairo<MC: MerkleChannel>(
    input: ProverInput,
    prover_params: ProverParameters,
) -> Result<CairoProof<MC::H>, ProvingError>
where
    SimdBackend: BackendForChannel<MC>,
{
    let _span = span!(Level::INFO, "prove_cairo").entered();
    let ProverParameters {
        channel_hash: _,
        channel_salt,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients,
        include_all_preprocessed_columns,
    } = prover_params;

    let max_domain_size = if let Some(lifting_log_size) = prover_params.pcs_config.lifting_log_size
    {
        lifting_log_size
    } else {
        let cairo_air_log_degree_bound = 1;
        LOG_MAX_ROWS
            + std::cmp::max(
                cairo_air_log_degree_bound,
                pcs_config.fri_config.log_blowup_factor,
            )
    };
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_domain_size)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let channel = &mut MC::C::default();

    // Mix channel salt. Note that we first reduce it modulo `M31::P`, then cast it as QM31.
    channel.mix_felts(&[channel_salt.into()]);
    pcs_config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, MC>::new(pcs_config, &twiddles);
    if store_polynomials_coefficients {
        commitment_scheme.set_store_polynomials_coefficients();
    }
    // Preprocessed trace.
    let span = span!(Level::INFO, "Preprocessed trace (SIMD)").entered();
    let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(gen_trace(preprocessed_trace.clone()));
    tree_builder.commit(channel);
    span.exit();

    // Run Cairo.
    let cairo_claim_generator = create_cairo_claim_generator(input, preprocessed_trace.clone());
    // Base trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    let span = span!(Level::INFO, "Base trace (SIMD)").entered();
    let (claim, interaction_generator) = cairo_claim_generator.write_trace(&mut tree_builder);
    span.exit();

    claim.mix_into(channel);
    tree_builder.commit(channel);

    // Draw interaction elements.
    let interaction_pow = SimdBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow);
    let interaction_elements = CommonLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction trace (SIMD)").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let interaction_claim =
        interaction_generator.write_interaction_trace(&mut tree_builder, &interaction_elements);
    span.exit();

    tracing::info!(
        "Witness trace cells: {:?}",
        witness_trace_cells(&claim, &preprocessed_trace)
    );
    // Validate lookup argument.
    debug_assert_eq!(
        lookup_sum(&claim, &interaction_elements, &interaction_claim),
        SecureField::zero()
    );

    interaction_claim.mix_into(channel);
    tree_builder.commit(channel);

    // Component provers.
    let component_builder = CairoComponents::new(
        &claim,
        &interaction_elements,
        &interaction_claim,
        &preprocessed_trace.ids(),
    );

    // TODO(Ohad): move to a testing routine.
    #[cfg(feature = "relation-tracker")]
    {
        use crate::debug_tools::relation_tracker::track_and_summarize_cairo_relations;
        let summary = track_and_summarize_cairo_relations(
            &commitment_scheme,
            &component_builder,
            &claim.public_data,
        );
        tracing::info!("Relations summary: {:?}", summary);
    }

    let components = cairo_provers(&component_builder);

    // Prove stark.
    let span = span!(Level::INFO, "Prove STARKs (SIMD)").entered();
    let proof = prove_ex::<SimdBackend, _>(
        &components,
        channel,
        commitment_scheme,
        include_all_preprocessed_columns,
    )?;
    span.exit();

    event!(name: "component_info", Level::DEBUG, "Components: {}", component_builder);

    Ok(CairoProof {
        claim,
        interaction_pow,
        interaction_claim,
        extended_stark_proof: proof,
        channel_salt,
        preprocessed_trace_variant: prover_params.preprocessed_trace,
    })
}

/// CUDA V0 proving path: SIMD trace generation, GPU constraint evaluation.
///
/// This function generates the witness traces using SIMD (same as `prove_cairo`),
/// converts them to CUDA format, then uses CudaBackend for the STARK proving step
/// (commitment scheme, FRI, and constraint evaluation on GPU).
///
/// Uses non-lifted MerkleHasher for CUDA compatibility.
pub fn prove_cairo_cuda_v0<MC: MerkleChannel>(
    input: ProverInput,
    prover_params: ProverParameters,
) -> Result<CairoProofCuda<MC::H>, ProvingError>
where
    stwo::prover::backend::cuda::CudaBackend: BackendForChannel<MC>,
    SimdBackend: BackendForChannel<MC>,
{
    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::prove;

    let _span = span!(Level::INFO, "prove_cairo_cuda_v0").entered();
    let ProverParameters {
        channel_hash: _,
        channel_salt,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients,
        include_all_preprocessed_columns: _,
    } = prover_params;

    stwo::stwo_cuda::print_cuda_memory("[V0] START");

    let cairo_air_log_degree_bound = 1;
    let max_domain_size = LOG_MAX_ROWS
        + std::cmp::max(
            cairo_air_log_degree_bound,
            pcs_config.fri_config.log_blowup_factor,
        );

    // Use CudaBackend for twiddles and commitment scheme.
    tracing::info!("[V0] Computing twiddles for CUDA backend");
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(max_domain_size)
            .circle_domain()
            .half_coset,
    );
    stwo::stwo_cuda::print_cuda_memory("[V0] After twiddles");

    // Setup protocol with CudaBackend.
    let channel = &mut MC::C::default();
    channel.mix_felts(&[channel_salt.into()]);
    pcs_config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, MC>::new(pcs_config, &twiddles);
    if store_polynomials_coefficients {
        commitment_scheme.set_store_polynomials_coefficients();
    }

    // Preprocessed trace — generate on GPU.
    tracing::info!("[V0] Generating preprocessed trace");
    let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
    let evals =
        crate::witness::preprocessed_trace_cuda::gen_preprocessed_trace_cuda(&preprocessed_trace);
    let polys =
        crate::witness::preprocessed_trace_cuda::interpolate_columns_batched(evals, &twiddles);
    stwo::stwo_cuda::print_cuda_memory("[V0] After preprocessed interpolation");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(polys);
    tree_builder.commit(channel);
    stwo::stwo_cuda::print_cuda_memory("[V0] After preprocessed commit");

    // Base trace — pure SIMD using SimdTraceCollector, then convert to CUDA.
    tracing::info!("[V0] Generating base trace (SIMD -> CUDA)");
    let cairo_claim_generator = create_cairo_claim_generator(input, preprocessed_trace.clone());
    let mut simd_tree_builder = crate::witness::cairo_cuda::SimdTraceCollector::new(1);
    let span = span!(Level::INFO, "Base trace (SIMD)").entered();
    let (claim, interaction_generator) = cairo_claim_generator.write_trace(&mut simd_tree_builder);
    span.exit();

    // Convert SIMD traces to CUDA and commit.
    let mut tree_builder = commitment_scheme.tree_builder();
    simd_tree_builder.extend_cuda_tree_builder(&mut tree_builder);
    stwo::stwo_cuda::print_cuda_memory("[V0] After base trace extend");
    claim.mix_into(channel);
    tree_builder.commit(channel);
    stwo::stwo_cuda::print_cuda_memory("[V0] After base trace commit");

    // Draw interaction elements.
    let interaction_pow = CudaBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow);
    let interaction_elements = CommonLookupElements::draw(channel);

    // Interaction trace — pure SIMD, convert to CUDA.
    tracing::info!("[V0] Generating interaction trace (SIMD -> CUDA)");
    let mut simd_tree_builder = crate::witness::cairo_cuda::SimdTraceCollector::new(2);
    let span = span!(Level::INFO, "Interaction trace (SIMD)").entered();
    let interaction_claim = interaction_generator
        .write_interaction_trace(&mut simd_tree_builder, &interaction_elements);
    span.exit();

    let mut tree_builder = commitment_scheme.tree_builder();
    simd_tree_builder.extend_cuda_tree_builder(&mut tree_builder);
    stwo::stwo_cuda::print_cuda_memory("[V0] After interaction trace extend");

    tracing::info!(
        "[V0] Witness trace cells: {:?}",
        witness_trace_cells(&claim, &preprocessed_trace)
    );

    // Validate lookup argument.
    debug_assert_eq!(
        lookup_sum(&claim, &interaction_elements, &interaction_claim),
        SecureField::zero()
    );

    interaction_claim.mix_into(channel);
    tree_builder.commit(channel);
    stwo::stwo_cuda::print_cuda_memory("[V0] After interaction trace commit");

    // Component provers with CUDA backend.
    let component_builder = CairoComponents::new(
        &claim,
        &interaction_elements,
        &interaction_claim,
        &preprocessed_trace.ids(),
    );
    let components = component_builder.provers_cuda();
    stwo::stwo_cuda::print_cuda_memory("[V0] After provers_cuda");

    // Prove STARK with CudaBackend.
    let span = span!(Level::INFO, "Prove STARKs (CUDA V0)").entered();
    stwo::stwo_cuda::print_cuda_memory("[V0] Before prove()");
    let proof = prove::<CudaBackend, _>(&components, channel, commitment_scheme)?;
    stwo::stwo_cuda::print_cuda_memory("[V0] After prove()");
    span.exit();

    event!(
        name: "component_info",
        Level::DEBUG,
        "Components: {}",
        component_builder
    );

    Ok(CairoProofCuda {
        claim,
        interaction_pow,
        interaction_claim,
        stark_proof: proof,
        channel_salt: Some(channel_salt as u64),
        preprocessed_trace_variant: prover_params.preprocessed_trace,
    })
}

/// Native CUDA proving path with per-component GPU witness generation.
///
/// Unlike `prove_cairo_cuda_v0` (which batch-collects all SIMD traces then converts),
/// this path generates trace columns directly on GPU for the native CUDA components and
/// keeps the proving pipeline on `CudaBackend` end-to-end.
pub fn prove_cairo_cuda<MC: MerkleChannel>(
    input: ProverInput,
    prover_params: ProverParameters,
) -> Result<CairoProofCuda<MC::H>, ProvingError>
where
    stwo::prover::backend::cuda::CudaBackend: BackendForChannel<MC>,
    SimdBackend: BackendForChannel<MC>,
{
    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::prove;

    let _span = span!(Level::INFO, "prove_cairo_cuda").entered();
    let ProverParameters {
        channel_hash: _,
        channel_salt,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients,
        include_all_preprocessed_columns: _,
    } = prover_params;

    stwo::stwo_cuda::print_cuda_memory("[CUDA] START");

    let cairo_air_log_degree_bound = 1;
    let max_domain_size = LOG_MAX_ROWS
        + std::cmp::max(
            cairo_air_log_degree_bound,
            pcs_config.fri_config.log_blowup_factor,
        );

    // Twiddles for CUDA backend.
    tracing::info!("[CUDA] Computing twiddles for CUDA backend");
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(max_domain_size)
            .circle_domain()
            .half_coset,
    );
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After twiddles");

    // Protocol setup.
    let channel = &mut MC::C::default();
    channel.mix_felts(&[channel_salt.into()]);
    pcs_config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, MC>::new(pcs_config, &twiddles);

    if store_polynomials_coefficients {
        commitment_scheme.set_store_polynomials_coefficients();
    }

    // Preprocessed trace — generate on GPU.
    tracing::info!("[CUDA] Generating preprocessed trace");
    let span = span!(Level::INFO, "Preprocessed trace (CUDA native)").entered();
    let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
    let evals =
        crate::witness::preprocessed_trace_cuda::gen_preprocessed_trace_cuda(&preprocessed_trace);
    let polys =
        crate::witness::preprocessed_trace_cuda::interpolate_columns_batched(evals, &twiddles);
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After preprocessed interpolation");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(polys);
    tree_builder.commit(channel);
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After preprocessed commit");
    span.exit();

    // Base trace — native CUDA generation (per-component GPU kernels, no SIMD bridge).
    tracing::info!("[CUDA] Generating base trace (native CUDA)");
    let native_cuda_gen = crate::witness::cairo_cuda::create_native_cairo_cuda_claim_generator(
        input,
        preprocessed_trace.clone(),
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    let span = span!(Level::INFO, "Base trace (CUDA native)").entered();
    let (claim, interaction_generator) = native_cuda_gen.write_trace(&mut tree_builder);
    span.exit();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After base trace extend");

    claim.mix_into(channel);
    tree_builder.commit(channel);
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After base trace commit");

    // Draw interaction elements.
    let interaction_pow = CudaBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow);
    let interaction_elements = CommonLookupElements::draw(channel);

    // Interaction trace — native CUDA generation (per-component GPU kernels).
    tracing::info!("[CUDA] Generating interaction trace (native CUDA)");
    let mut tree_builder = commitment_scheme.tree_builder();
    let span = span!(Level::INFO, "Interaction trace (CUDA native)").entered();
    let interaction_claim =
        interaction_generator.write_interaction_trace(&mut tree_builder, &interaction_elements);
    span.exit();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After interaction trace extend");

    tracing::info!(
        "[CUDA] Witness trace cells: {:?}",
        witness_trace_cells(&claim, &preprocessed_trace)
    );

    // Validate lookup argument.
    debug_assert_eq!(
        lookup_sum(&claim, &interaction_elements, &interaction_claim),
        SecureField::zero()
    );

    interaction_claim.mix_into(channel);
    tree_builder.commit(channel);
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After interaction trace commit");

    // Component provers with CUDA backend.
    let component_builder = CairoComponents::new(
        &claim,
        &interaction_elements,
        &interaction_claim,
        &preprocessed_trace.ids(),
    );
    let components = component_builder.provers_cuda();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After provers_cuda");

    // Prove STARK with CudaBackend.
    let span = span!(Level::INFO, "Prove STARKs (CUDA native)").entered();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] Before prove()");
    let proof = prove::<CudaBackend, _>(&components, channel, commitment_scheme)?;
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After prove()");
    span.exit();

    event!(
        name: "component_info",
        Level::DEBUG,
        "Components: {}",
        component_builder
    );

    Ok(CairoProofCuda {
        claim,
        interaction_pow,
        interaction_claim,
        stark_proof: proof,
        channel_salt: Some(channel_salt as u64),
        preprocessed_trace_variant: prover_params.preprocessed_trace,
    })
}

/// Concrete parameters of the proving system.
/// Used both for producing and verifying proofs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProverParameters {
    /// Channel hash function.
    pub channel_hash: ChannelHash,
    /// Salt for the channel initialization.
    /// Note that the salt is only used to allow recomputation of the proof with other draws
    /// of the randomness, in case of failure due to unprovable draws (e.g. a zero in the
    /// denominator).
    pub channel_salt: u32,
    /// Parameters of the commitment scheme.
    pub pcs_config: PcsConfig,
    /// Preprocessed trace.
    pub preprocessed_trace: PreProcessedTraceVariant,
    /// Whether or not to store the polynomials coefficients. Affects runtime-memory usage
    /// trade-off. Default is `false`.
    pub store_polynomials_coefficients: bool,
    /// Whether to include samples for every preprocessed column in the proof. Default is `false`.
    /// If `false`, the proof only includes samples for columns used by at least one component.
    pub include_all_preprocessed_columns: bool,
}

/// The hash function used for commitments, for the prover-verifier channel,
/// and for PoW grinding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelHash {
    /// Default variant, the fastest option.
    Blake2s,
    /// A variant for Blake2s where modulo M31 is applied to every 32bits in the output.
    Blake2sM31,
    /// A variant for recursive proof verification.
    /// Note that using `Poseidon252` results in a significant decrease in proving speed compared
    /// to `Blake2s` (because of the large field emulation)
    Poseidon252,
}

/// Generates proof given the Cairo VM output and prover config/parameters.
/// Serializes the proof as JSON and write to the output path.
/// Verifies the proof in case the respective flag is set.
pub fn create_and_serialize_proof(
    input: ProverInput,
    verify: bool,
    proof_path: PathBuf,
    proof_format: ProofFormat,
    proof_params_json: Option<PathBuf>,
) -> Result<()> {
    let proof_params = if let Some(proof_params_json) = proof_params_json {
        json::from_str(&read_to_string(&proof_params_json)?)?
    } else {
        // The default prover parameters for prod use (96 bits of security).
        // The formula is `security_bits = pow_bits + log_blowup_factor * n_queries`.
        ProverParameters {
            channel_hash: ChannelHash::Blake2s,
            channel_salt: 0,
            pcs_config: PcsConfig {
                // Stay within 500ms on M3.
                pow_bits: 26,
                fri_config: FriConfig {
                    log_last_layer_degree_bound: 0,
                    // Blowup factor > 1 significantly degrades proving speed.
                    // Can be in range [1, 16].
                    log_blowup_factor: 1,
                    // The more FRI queries, the larger the proof.
                    // Proving time is not affected much by increasing this value.
                    n_queries: 70,
                    line_fold_step: 1,
                },
                lifting_log_size: None,
            },
            preprocessed_trace: PreProcessedTraceVariant::Canonical,
            store_polynomials_coefficients: false,
            include_all_preprocessed_columns: false,
        }
    };

    match proof_params.channel_hash {
        ChannelHash::Blake2s => {
            prove_verify_serialize::<Blake2sMerkleChannel>(
                input,
                verify,
                &proof_path,
                proof_format,
                proof_params,
            )?;
        }
        ChannelHash::Blake2sM31 => {
            prove_verify_serialize::<Blake2sM31MerkleChannel>(
                input,
                verify,
                &proof_path,
                proof_format,
                proof_params,
            )?;
        }
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        ChannelHash::Poseidon252 => {
            unimplemented!("Poseidon252 is not supported for wasm targets");
        }
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        ChannelHash::Poseidon252 => {
            use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
            prove_verify_serialize::<Poseidon252MerkleChannel>(
                input,
                verify,
                &proof_path,
                proof_format,
                proof_params,
            )?;
        }
    };

    Ok(())
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
        testing_preprocessed_tree, PreProcessedTrace,
    };
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    use crate::debug_tools::assert_constraints::assert_cairo_constraints;

    #[test]
    fn test_all_cairo_constraints() {
        let compiled_program =
            get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
        let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
        let pp_tree = Arc::new(testing_preprocessed_tree(24));
        assert_cairo_constraints(input, pp_tree);
    }

    #[test]
    fn test_all_cairo_constraints_small_ppt() {
        let compiled_program =
            get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
        let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
        let pp_tree = Arc::new(PreProcessedTrace::canonical_small());
        assert_cairo_constraints(input, pp_tree);
    }

    #[cfg(test)]
    #[cfg(feature = "nightly")]
    mod nightly_tests {
        use std::io::Write;
        use std::process::Command;

        use cairo_air::PreProcessedTraceVariant;
        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
        use stwo_cairo_dev_utils::utils::get_proof_file_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
        use stwo_cairo_serialize::CairoSerialize;
        use tempfile::NamedTempFile;
        use test_log::test;

        use super::*;
        use crate::prover::{prove_cairo, ChannelHash, ProverParameters};

        #[test]
        fn test_poseidon_e2e_prove_cairo_verify_ret_opcode_components() {
            let compiled_program = get_compiled_cairo_program_path("test_prove_verify_ret_opcode");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Poseidon252,
                pcs_config: PcsConfig {
                    pow_bits: 20,
                    fri_config: FriConfig::new(0, 1, 90, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
                channel_salt: 42,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo::<Poseidon252MerkleChannel>(input, prover_params).unwrap();
            let mut proof_file = NamedTempFile::new().unwrap();
            let mut serialized: Vec<starknet_ff::FieldElement> = Vec::new();
            CairoSerialize::serialize(&cairo_proof, &mut serialized);
            let proof_hex: Vec<String> = serialized
                .into_iter()
                .map(|felt| format!("0x{felt:x}"))
                .collect();
            proof_file
                .write_all(sonic_rs::to_string_pretty(&proof_hex).unwrap().as_bytes())
                .unwrap();
            let expected_proof_file = get_proof_file_path("test_prove_verify_ret_opcode");

            if std::env::var("FIX_PROOF").is_ok() {
                std::fs::copy(proof_file.path(), &expected_proof_file)
                    .expect("Failed to overwrite expected proof file");
            }

            // Compare the contents of proof_file and expected_proof_file
            let proof_file_contents = std::fs::read_to_string(proof_file.path())
                .expect("Failed to read generated proof file");
            let expected_proof_contents = std::fs::read_to_string(&expected_proof_file)
                .expect("Failed to read expected proof file");
            assert!(
                proof_file_contents == expected_proof_contents,
                "Generated proof file does not match the expected proof file"
            );

            let status = Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "(cd ../../../stwo_cairo_verifier; \
                    scarb execute --package stwo_cairo_verifier \
                    --arguments-file {} --output standard --target standalone \
                    --features poseidon252_verifier
                    )",
                    proof_file.path().to_str().unwrap()
                ))
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .status()
                .unwrap();

            assert!(status.success());
        }
    }

    #[cfg(test)]
    #[cfg(feature = "slow-tests")]
    pub mod slow_tests {

        use std::io::Write;
        use std::process::Command;

        use cairo_air::verifier::verify_cairo;
        use cairo_air::CairoProofForRustVerifier;
        use itertools::Itertools;
        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
        use stwo_cairo_dev_utils::utils::{get_compiled_cairo_program_path, get_proof_file_path};
        use stwo_cairo_serialize::CairoSerialize;
        use tempfile::NamedTempFile;
        use test_log::test;

        use super::*;
        use crate::debug_tools::assert_constraints::assert_cairo_constraints;
        use crate::prover::{
            prove_cairo, ChannelHash, PreProcessedTraceVariant, ProverInput, ProverParameters,
        };

        // TODO(Ohad): fine-grained constraints tests.
        #[test]
        fn test_cairo_constraints() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            assert_cairo_constraints(
                input,
                Arc::new(PreProcessedTrace::canonical_without_pedersen()),
            );
        }

        #[test_log::test]
        fn test_prove_verify_all_opcode_components() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            for (opcode, n_instances) in &input.state_transitions.casm_states_by_opcode.counts() {
                assert!(
                    *n_instances > 0,
                    "{opcode} isn't used in E2E full-Cairo opcode test"
                );
            }
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig::default(),
                preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
                channel_salt: 0,
                store_polynomials_coefficients: true,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof = prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo::<Blake2sMerkleChannel>(cairo_proof.into()).unwrap();
        }

        #[test]
        fn test_e2e_prove_cairo_verify_all_opcode_components() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::Canonical,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof = prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            let mut proof_file = NamedTempFile::new().unwrap();
            let mut serialized: Vec<starknet_ff::FieldElement> = Vec::new();
            CairoSerialize::serialize(&cairo_proof, &mut serialized);
            let proof_hex: Vec<String> = serialized
                .into_iter()
                .map(|felt| format!("0x{felt:x}"))
                .collect();
            proof_file
                .write_all(sonic_rs::to_string_pretty(&proof_hex).unwrap().as_bytes())
                .unwrap();

            let expected_proof_file =
                get_proof_file_path("test_prove_verify_all_opcode_components");
            if std::env::var("FIX_PROOF").is_ok() {
                std::fs::copy(proof_file.path(), &expected_proof_file)
                    .expect("Failed to overwrite expected proof file");
            }

            // Compare the contents of proof_file and expected_proof_file
            let proof_file_contents = std::fs::read_to_string(proof_file.path())
                .expect("Failed to read generated proof file");
            let expected_proof_contents = std::fs::read_to_string(&expected_proof_file)
                .expect("Failed to read expected proof file");
            assert!(
                proof_file_contents == expected_proof_contents,
                "Generated proof file does not match the expected proof file"
            );

            let status = Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "(cd ../../../stwo_cairo_verifier; \
                    scarb execute --package stwo_cairo_verifier \
                    --arguments-file {} --output standard --target standalone \
                    --features qm31_opcode
                    )",
                    proof_file.path().to_str().unwrap()
                ))
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .status()
                .unwrap();

            assert!(status.success());
        }

        #[test]
        fn test_e2e_prove_cairo_verify_all_builtins() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_builtins");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::Canonical,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof = prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            let mut proof_file = NamedTempFile::new().unwrap();
            let mut serialized: Vec<starknet_ff::FieldElement> = Vec::new();
            CairoSerialize::serialize(&cairo_proof, &mut serialized);
            let proof_hex: Vec<String> = serialized
                .into_iter()
                .map(|felt| format!("0x{felt:x}"))
                .collect();
            proof_file
                .write_all(sonic_rs::to_string_pretty(&proof_hex).unwrap().as_bytes())
                .unwrap();

            let status = Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "(cd ../../../stwo_cairo_verifier; \
                    scarb execute --package stwo_cairo_verifier \
                    --arguments-file {} --output standard --target standalone \
                    --features qm31_opcode
                    )",
                    proof_file.path().to_str().unwrap()
                ))
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .status()
                .unwrap();

            assert!(status.success());
        }

        fn test_proof_stability(path: &str, n_proofs_to_compare: usize) {
            let compiled_program = get_compiled_cairo_program_path(path);
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig::default(),
                preprocessed_trace: PreProcessedTraceVariant::Canonical,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let proofs = (0..n_proofs_to_compare)
                .map(|_| {
                    let proof: CairoProofForRustVerifier<_> =
                        prove_cairo::<Blake2sMerkleChannel>(input.clone(), prover_params)
                            .unwrap()
                            .into();
                    sonic_rs::to_string(&proof).unwrap()
                })
                .collect_vec();

            assert!(proofs.iter().all_equal());
        }

        #[test]
        fn test_opcodes_proof_stability() {
            test_proof_stability("test_prove_verify_all_opcode_components", 2);
        }

        #[test]
        fn test_builtins_proof_stability() {
            test_proof_stability("test_prove_verify_all_builtins", 2);
        }

        /// These tests' inputs were generated using cairo-vm with 50 instances of each builtin.
        pub mod builtin_tests {
            use stwo::core::pcs::PcsConfig;
            use stwo_cairo_common::preprocessed_columns::preprocessed_trace::testing_preprocessed_tree;
            use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
            use test_log::test;

            use super::*;

            /// Asserts that all supported builtins are present in the input.
            /// Panics if any of the builtins is missing.
            fn assert_all_builtins_in_input(input: &ProverInput) {
                let empty_builtins: Vec<_> = input
                    .builtin_segments
                    .get_counts()
                    .into_iter()
                    .filter(|(_, count)| *count == 0)
                    .map(|(name, _)| name)
                    .collect();

                if !empty_builtins.is_empty() {
                    panic!("Builtins missing in the input: {empty_builtins:?}");
                }
            }

            #[test]
            fn test_prove_verify_all_builtins() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_all_builtins");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_all_builtins_in_input(&input);
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::Canonical,
                    channel_salt: 0,
                    store_polynomials_coefficients: false,
                    include_all_preprocessed_columns: false,
                };
                let cairo_proof =
                    prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
                verify_cairo::<Blake2sMerkleChannel>(cairo_proof.into()).unwrap();
            }

            #[test]
            fn test_prove_verify_all_builtins_canonical_small() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_all_builtins");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_all_builtins_in_input(&input);
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                    channel_salt: 0,
                    store_polynomials_coefficients: false,
                    include_all_preprocessed_columns: false,
                };
                let cairo_proof =
                    prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
                verify_cairo::<Blake2sMerkleChannel>(cairo_proof.into()).unwrap();
            }

            #[test]
            fn test_add_mod_builtin_constraints() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_add_mod_builtin");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(
                    input,
                    Arc::new(PreProcessedTrace::canonical_without_pedersen()),
                );
            }

            #[test]
            fn test_bitwise_builtin_constraints() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_bitwise_builtin");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(testing_preprocessed_tree(20)));
            }

            #[test]
            fn test_mul_mod_builtin_constraints() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_mul_mod_builtin");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(testing_preprocessed_tree(20)));
            }

            #[test]
            fn test_pedersen_builtin_constraints() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(PreProcessedTrace::canonical()));
            }

            #[test]
            fn test_pedersen_narrow_windows_builtin_constraints() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(PreProcessedTrace::canonical_small()));
            }

            #[test]
            fn test_poseidon_builtin_constraints() {
                let compiled_program =
                    get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin");
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(testing_preprocessed_tree(20)));
            }

            #[test]
            fn test_range_check_bits_96_builtin_constraints() {
                let compiled_program = get_compiled_cairo_program_path(
                    "test_prove_verify_range_check_bits_96_builtin",
                );
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(testing_preprocessed_tree(20)));
            }

            #[test]
            fn test_range_check_bits_128_builtin_constraints() {
                let compiled_program = get_compiled_cairo_program_path(
                    "test_prove_verify_range_check_bits_128_builtin",
                );
                let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
                assert_cairo_constraints(input, Arc::new(testing_preprocessed_tree(20)));
            }

            #[test]
            fn test_poseidon_aggregator() {
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::Canonical,
                    channel_salt: 0,
                    store_polynomials_coefficients: false,
                    include_all_preprocessed_columns: false,
                };

                // Run poseidon builtin with 15 different instances.
                let compiled_program_a =
                    get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin");
                let input_a = run_and_adapt(&compiled_program_a, ProgramType::Json, None).unwrap();
                let proof_a = prove_cairo::<Blake2sMerkleChannel>(input_a, prover_params).unwrap();
                let poseidon_builtin_size_a = 2u32.pow(
                    proof_a
                        .claim
                        .poseidon_builtin
                        .expect("Poseidon builtin is not present in the claim")
                        .log_size,
                );
                assert!(poseidon_builtin_size_a == 16, "Expected program to contain 15 poseidon instances, which then padded to the next power of two");

                let poseidon_aggregator_log_size_a = proof_a
                    .claim
                    .poseidon_aggregator
                    .expect("Poseidon context is not present in the claim")
                    .log_size;

                // Run poseidon builtin with 15 different instances, each one 30 times.
                let compiled_program_b =
                    get_compiled_cairo_program_path("test_poseidon_aggregator");
                let input_b = run_and_adapt(&compiled_program_b, ProgramType::Json, None).unwrap();
                let proof_b = prove_cairo::<Blake2sMerkleChannel>(input_b, prover_params).unwrap();
                let poseidon_builtin_size_b = 2u32.pow(
                    proof_b
                        .claim
                        .poseidon_builtin
                        .expect("Poseidon builtin is not present in the claim")
                        .log_size,
                );
                assert!(poseidon_builtin_size_b == 512, "Expected program to contain 15*30 poseidon instances, which then padded to the next power of two");

                let poseidon_aggregator_log_size_b = proof_b
                    .claim
                    .poseidon_aggregator
                    .expect("Poseidon context is not present in the claim")
                    .log_size;

                assert_eq!(
                    poseidon_aggregator_log_size_a,
                    poseidon_aggregator_log_size_b,
                    "Poseidon aggregator log size should be the same for both proof because it uses multiplicity"
                );
            }

            #[test]
            fn test_pedersen_aggregator() {
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::Canonical,
                    channel_salt: 0,
                    store_polynomials_coefficients: false,
                    include_all_preprocessed_columns: false,
                };

                // Run pedersen builtin with 15 different instances.
                let compiled_program_a =
                    get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
                let input_a = run_and_adapt(&compiled_program_a, ProgramType::Json, None).unwrap();
                let proof_a = prove_cairo::<Blake2sMerkleChannel>(input_a, prover_params).unwrap();
                let pedersen_builtin_size_a = 2u32.pow(
                    proof_a
                        .claim
                        .pedersen_builtin
                        .expect("Pedersen builtin is not present in the claim")
                        .log_size,
                );
                assert!(pedersen_builtin_size_a == 16, "Expected program to contain 15 pedersen instances, which then padded to the next power of two");

                let pedersen_aggregator_log_size_a = proof_a
                    .claim
                    .pedersen_aggregator_window_bits_18
                    .expect("Pedersen context is not present in the claim")
                    .log_size;

                // Run pedersen builtin with 15 different instances, each one 30 times.
                let compiled_program_b =
                    get_compiled_cairo_program_path("test_pedersen_aggregator");
                let input_b = run_and_adapt(&compiled_program_b, ProgramType::Json, None).unwrap();
                let proof_b = prove_cairo::<Blake2sMerkleChannel>(input_b, prover_params).unwrap();
                let pedersen_builtin_size_b = 2u32.pow(
                    proof_b
                        .claim
                        .pedersen_builtin
                        .expect("Pedersen builtin is not present in the claim")
                        .log_size,
                );
                assert!(pedersen_builtin_size_b == 512, "Expected program to contain 15*30 pedersen instances, which then padded to the next power of two");

                let pedersen_aggregator_log_size_b = proof_b
                    .claim
                    .pedersen_aggregator_window_bits_18
                    .expect("Pedersen context is not present in the claim")
                    .log_size;

                assert_eq!(
                    pedersen_aggregator_log_size_a,
                    pedersen_aggregator_log_size_b,
                    "Pedersen aggregator log size should be the same for both proof because it uses multiplicity"
                );
            }
        }
    }

    pub mod cuda_tests {
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        use cairo_air::air::CairoComponents;
        use cairo_air::claims::lookup_sum;
        use cairo_air::relations::CommonLookupElements;
        use cairo_air::verifier::{verify_cairo_ex, INTERACTION_POW_BITS};
        use cairo_air::{CairoProof, CairoProofCuda, PreProcessedTraceVariant};
        use num_traits::Zero;
        use stwo::core::channel::{Channel, MerkleChannel};
        use stwo::core::fields::qm31::SecureField;
        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::core::proof_of_work::GrindOps;
        use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
        use stwo::prover::backend::cuda::CudaBackend;
        use stwo::prover::backend::simd::SimdBackend;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::{prove, prove_ex, CommitmentSchemeProver};
        use stwo_cairo_adapter::ProverInput;
        use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
        use test_log::test;

        use crate::prover::{
            prove_cairo, prove_cairo_cuda, prove_cairo_cuda_v0, ChannelHash, ProverParameters,
            LOG_MAX_ROWS,
        };
        use crate::utils::cairo_provers;
        use crate::witness::cairo::create_cairo_claim_generator;
        use crate::witness::preprocessed_trace::gen_trace;
        use crate::witness::preprocessed_trace_cuda::{
            gen_preprocessed_trace_cuda, interpolate_columns_batched,
        };

        /// Load a CairoPie zip file and produce ProverInput via bootloader execution.
        fn load_pie_input(pie_path: &std::path::Path) -> ProverInput {
            use std::rc::Rc;

            use cairo_program_runner_lib::tasks::create_pie_task;
            use cairo_program_runner_lib::types::{HashFunc, RunMode};
            use cairo_program_runner_lib::{
                cairo_run_program, ProgramInput, SimpleBootloaderInput, TaskSpec,
            };
            use cairo_vm::types::layout_name::LayoutName;
            use cairo_vm::types::program::Program;
            use stwo_cairo_adapter::adapter::adapt;

            let task = create_pie_task(pie_path).expect("Failed to load CairoPie zip");
            let task_spec = TaskSpec {
                task: Rc::new(task),
                program_hash_function: HashFunc::Blake,
            };
            let bootloader_input = SimpleBootloaderInput {
                fact_topologies_path: None,
                single_page: true,
                tasks: vec![task_spec],
            };

            let bootloader_path = std::path::PathBuf::from(env!("BOOTLOADER_JSON_PATH"));
            let bootloader_program = Program::from_file(bootloader_path.as_path(), Some("main"))
                .expect("Failed to load simple_bootloader_compiled.json");

            let cairo_run_config = RunMode::Proof {
                layout: LayoutName::all_cairo_stwo,
                dynamic_layout_params: None,
                disable_trace_padding: true,
                relocate_mem: false,
            }
            .create_config();

            let runner = cairo_run_program(
                &bootloader_program,
                Some(ProgramInput::Value(Box::new(bootloader_input))),
                cairo_run_config,
                None,
            )
            .expect("Failed to run PIE through bootloader");

            adapt(&runner).expect("Failed to adapt runner to ProverInput")
        }

        #[test]
        fn test_e2e_prove_cuda_v0_all_opcode_components() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo_cuda_v0::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo_ex::<Blake2sMerkleChannel>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();
        }

        #[test]
        fn test_e2e_prove_cuda_v0_all_builtins() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_builtins");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo_cuda_v0::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo_ex::<Blake2sMerkleChannel>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();
        }

        // --- Native CUDA path tests (per-component inline conversion) ---

        #[test]
        fn test_e2e_prove_cuda_all_opcode_components() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
                channel_salt: 0,
                store_polynomials_coefficients: true,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo_ex::<Blake2sMerkleChannel>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();
        }

        #[test]
        fn test_e2e_prove_cuda_all_builtins() {
            let compiled_program =
                get_compiled_cairo_program_path("test_prove_verify_all_builtins");
            let input = run_and_adapt(&compiled_program, ProgramType::Json, None).unwrap();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                channel_salt: 0,
                store_polynomials_coefficients: true,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo_ex::<Blake2sMerkleChannel>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();
        }

        // ==================== PIE File Tests ====================

        fn load_pie_10_transfers_input() -> ProverInput {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/test_small_pie/cairo_pie_10_transfers_with_6_ecop.zip");
            load_pie_input(&path)
        }

        fn load_sn_pie_input() -> ProverInput {
            let path =
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/sn_pie");
            load_pie_input(&path)
        }

        #[derive(Clone, Copy, Debug)]
        struct BenchmarkTimings {
            setup: Duration,
            preprocessed: Duration,
            base: Duration,
            interaction: Duration,
            prove: Duration,
            total: Duration,
        }

        impl BenchmarkTimings {
            fn print_report(&self, label: &str) {
                println!("\n{label} timings:");
                println!("  setup:        {:8.3} ms", millis(self.setup));
                println!("  preprocessed: {:8.3} ms", millis(self.preprocessed));
                println!("  base trace:   {:8.3} ms", millis(self.base));
                println!("  interaction:  {:8.3} ms", millis(self.interaction));
                println!("  stark prove:  {:8.3} ms", millis(self.prove));
                println!("  total:        {:8.3} ms", millis(self.total));
            }
        }

        fn millis(duration: Duration) -> f64 {
            duration.as_secs_f64() * 1000.0
        }

        fn small_pie_benchmark_params() -> ProverParameters {
            ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                channel_salt: 0,
                store_polynomials_coefficients: true,
                include_all_preprocessed_columns: false,
            }
        }

        fn print_speedups(simd: &BenchmarkTimings, cuda: &BenchmarkTimings) {
            let rows = [
                ("setup", simd.setup, cuda.setup),
                ("preprocessed", simd.preprocessed, cuda.preprocessed),
                ("base trace", simd.base, cuda.base),
                ("interaction", simd.interaction, cuda.interaction),
                ("stark prove", simd.prove, cuda.prove),
                ("total", simd.total, cuda.total),
            ];

            println!("\nStage speedups (SIMD / CUDA warm):");
            println!(
                "  {:<14} {:>12} {:>12} {:>10}",
                "stage", "SIMD ms", "CUDA ms", "speedup"
            );
            for (label, simd_dur, cuda_dur) in rows {
                println!(
                    "  {:<14} {:>12.3} {:>12.3} {:>9.2}x",
                    label,
                    millis(simd_dur),
                    millis(cuda_dur),
                    simd_dur.as_secs_f64() / cuda_dur.as_secs_f64()
                );
            }
        }

        fn simd_max_domain_size(prover_params: ProverParameters) -> u32 {
            if let Some(lifting_log_size) = prover_params.pcs_config.lifting_log_size {
                lifting_log_size
            } else {
                let cairo_air_log_degree_bound = 1;
                LOG_MAX_ROWS
                    + std::cmp::max(
                        cairo_air_log_degree_bound,
                        prover_params.pcs_config.fri_config.log_blowup_factor,
                    )
            }
        }

        fn run_small_pie_simd_benchmark() -> BenchmarkTimings {
            type MC = Blake2sMerkleChannel;

            let input = load_pie_10_transfers_input();
            let prover_params = small_pie_benchmark_params();
            let ProverParameters {
                channel_hash: _,
                channel_salt,
                pcs_config,
                preprocessed_trace,
                store_polynomials_coefficients,
                include_all_preprocessed_columns,
            } = prover_params;

            let total_timer = Instant::now();

            let setup_timer = Instant::now();
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(simd_max_domain_size(prover_params))
                    .circle_domain()
                    .half_coset,
            );
            let channel = &mut <MC as MerkleChannel>::C::default();
            channel.mix_felts(&[channel_salt.into()]);
            pcs_config.mix_into(channel);
            let mut commitment_scheme =
                CommitmentSchemeProver::<SimdBackend, MC>::new(pcs_config, &twiddles);
            if store_polynomials_coefficients {
                commitment_scheme.set_store_polynomials_coefficients();
            }
            let setup = setup_timer.elapsed();

            let preprocessed_timer = Instant::now();
            let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(gen_trace(preprocessed_trace.clone()));
            tree_builder.commit(channel);
            let preprocessed = preprocessed_timer.elapsed();

            let base_timer = Instant::now();
            let cairo_claim_generator =
                create_cairo_claim_generator(input, preprocessed_trace.clone());
            let mut tree_builder = commitment_scheme.tree_builder();
            let (claim, interaction_generator) =
                cairo_claim_generator.write_trace(&mut tree_builder);
            claim.mix_into(channel);
            tree_builder.commit(channel);
            let base = base_timer.elapsed();

            let interaction_timer = Instant::now();
            let interaction_pow = SimdBackend::grind(channel, INTERACTION_POW_BITS);
            channel.mix_u64(interaction_pow);
            let interaction_elements = CommonLookupElements::draw(channel);
            let mut tree_builder = commitment_scheme.tree_builder();
            let interaction_claim = interaction_generator
                .write_interaction_trace(&mut tree_builder, &interaction_elements);
            assert_eq!(
                lookup_sum(&claim, &interaction_elements, &interaction_claim),
                SecureField::zero()
            );
            interaction_claim.mix_into(channel);
            tree_builder.commit(channel);
            let interaction = interaction_timer.elapsed();

            let prove_timer = Instant::now();
            let component_builder = CairoComponents::new(
                &claim,
                &interaction_elements,
                &interaction_claim,
                &preprocessed_trace.ids(),
            );
            let components = cairo_provers(&component_builder);
            let proof = prove_ex::<SimdBackend, _>(
                &components,
                channel,
                commitment_scheme,
                include_all_preprocessed_columns,
            )
            .unwrap();
            let prove = prove_timer.elapsed();
            let total = total_timer.elapsed();

            let cairo_proof = CairoProof {
                claim,
                interaction_pow,
                interaction_claim,
                extended_stark_proof: proof,
                channel_salt,
                preprocessed_trace_variant: prover_params.preprocessed_trace,
            };
            verify_cairo_ex::<MC>(cairo_proof.into(), include_all_preprocessed_columns).unwrap();

            BenchmarkTimings {
                setup,
                preprocessed,
                base,
                interaction,
                prove,
                total,
            }
        }

        fn run_small_pie_cuda_benchmark() -> BenchmarkTimings {
            type MC = Blake2sMerkleChannel;

            let input = load_pie_10_transfers_input();
            let prover_params = small_pie_benchmark_params();
            let ProverParameters {
                channel_hash: _,
                channel_salt,
                pcs_config,
                preprocessed_trace,
                store_polynomials_coefficients,
                include_all_preprocessed_columns: _,
            } = prover_params;

            let total_timer = Instant::now();

            let setup_timer = Instant::now();
            let twiddles = CudaBackend::precompute_twiddles(
                CanonicCoset::new(simd_max_domain_size(prover_params))
                    .circle_domain()
                    .half_coset,
            );
            let channel = &mut <MC as MerkleChannel>::C::default();
            channel.mix_felts(&[channel_salt.into()]);
            pcs_config.mix_into(channel);
            let mut commitment_scheme =
                CommitmentSchemeProver::<CudaBackend, MC>::new(pcs_config, &twiddles);
            if store_polynomials_coefficients {
                commitment_scheme.set_store_polynomials_coefficients();
            }
            let setup = setup_timer.elapsed();

            let preprocessed_timer = Instant::now();
            let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
            let evals = gen_preprocessed_trace_cuda(&preprocessed_trace);
            let polys = interpolate_columns_batched(evals, &twiddles);
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_polys(polys);
            tree_builder.commit(channel);
            let preprocessed = preprocessed_timer.elapsed();

            let base_timer = Instant::now();
            let native_cuda_gen =
                crate::witness::cairo_cuda::create_native_cairo_cuda_claim_generator(
                    input,
                    preprocessed_trace.clone(),
                );
            let mut tree_builder = commitment_scheme.tree_builder();
            let (claim, interaction_generator) = native_cuda_gen.write_trace(&mut tree_builder);
            claim.mix_into(channel);
            tree_builder.commit(channel);
            let base = base_timer.elapsed();

            let interaction_timer = Instant::now();
            let interaction_pow = CudaBackend::grind(channel, INTERACTION_POW_BITS);
            channel.mix_u64(interaction_pow);
            let interaction_elements = CommonLookupElements::draw(channel);
            let mut tree_builder = commitment_scheme.tree_builder();
            let interaction_claim = interaction_generator
                .write_interaction_trace(&mut tree_builder, &interaction_elements);
            assert_eq!(
                lookup_sum(&claim, &interaction_elements, &interaction_claim),
                SecureField::zero()
            );
            interaction_claim.mix_into(channel);
            tree_builder.commit(channel);
            let interaction = interaction_timer.elapsed();

            let prove_timer = Instant::now();
            let component_builder = CairoComponents::new(
                &claim,
                &interaction_elements,
                &interaction_claim,
                &preprocessed_trace.ids(),
            );
            let components = component_builder.provers_cuda();
            let proof = prove::<CudaBackend, _>(&components, channel, commitment_scheme).unwrap();
            let prove = prove_timer.elapsed();
            let total = total_timer.elapsed();

            let cairo_proof = CairoProofCuda {
                claim,
                interaction_pow,
                interaction_claim,
                stark_proof: proof,
                channel_salt: Some(channel_salt as u64),
                preprocessed_trace_variant: prover_params.preprocessed_trace,
            };
            verify_cairo_ex::<MC>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();

            BenchmarkTimings {
                setup,
                preprocessed,
                base,
                interaction,
                prove,
                total,
            }
        }

        #[test]
        fn test_prove_verify_small_pie_cuda_once() {
            let input = load_pie_10_transfers_input();
            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig {
                    pow_bits: 26,
                    fri_config: FriConfig::new(0, 1, 70, 1),
                    lifting_log_size: None,
                },
                preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                channel_salt: 0,
                store_polynomials_coefficients: true,
                include_all_preprocessed_columns: false,
            };
            let timer = std::time::Instant::now();
            let cairo_proof =
                prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            println!("CUDA proof generation time: {:?}", timer.elapsed());
            verify_cairo_ex::<Blake2sMerkleChannel>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();
        }

        #[test]
        fn test_prove_verify_small_pie_cuda_multi() {
            let loop_count: usize = std::env::var("PROVE_LOOP_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            for run in 0..loop_count {
                println!("\n========== RUN {} ==========\n", run);
                let input = load_pie_10_transfers_input();
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig {
                        pow_bits: 26,
                        fri_config: FriConfig::new(0, 1, 70, 1),
                        lifting_log_size: None,
                    },
                    // pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                    channel_salt: 0,
                    store_polynomials_coefficients: true,
                    include_all_preprocessed_columns: false,
                };
                let timer = std::time::Instant::now();
                let cairo_proof =
                    prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
                let prove_time = timer.elapsed();
                verify_cairo_ex::<Blake2sMerkleChannel>(
                    cairo_proof.into(),
                    prover_params.include_all_preprocessed_columns,
                )
                .unwrap();
                println!("No.{} CUDA proof generation time: {:?}", run, prove_time);
            }
        }

        #[test]
        #[ignore = "manual SIMD baseline benchmark; run with --ignored"]
        fn test_prove_verify_small_pie_simd_once() {
            let input = load_pie_10_transfers_input();
            let prover_params = small_pie_benchmark_params();
            let timer = Instant::now();
            let cairo_proof = prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            println!("SIMD proof generation time: {:?}", timer.elapsed());
            verify_cairo_ex::<Blake2sMerkleChannel>(
                cairo_proof.into(),
                prover_params.include_all_preprocessed_columns,
            )
            .unwrap();
        }

        #[test]
        fn test_prove_verify_small_pie_simd_multi() {
            let loop_count: usize = std::env::var("PROVE_LOOP_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            for run in 0..loop_count {
                println!("\n========== RUN {} ==========\n", run);
                let input = load_pie_10_transfers_input();
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig {
                        pow_bits: 26,
                        fri_config: FriConfig::new(0, 1, 70, 1),
                        lifting_log_size: None,
                    },
                    // pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                    channel_salt: 0,
                    store_polynomials_coefficients: true,
                    include_all_preprocessed_columns: false,
                };
                let time = std::time::Instant::now();
                let cairo_proof =
                    prove_cairo::<Blake2sMerkleChannel>(input, prover_params).unwrap();
                let prove_time = time.elapsed();
                verify_cairo_ex::<Blake2sMerkleChannel>(
                    cairo_proof.into(),
                    prover_params.include_all_preprocessed_columns,
                )
                .unwrap();
                println!("No.{} SIMD proof generation time: {:?}", run, prove_time);
            }
        }
        #[test]
        #[ignore = "manual matched SIMD/CUDA stage benchmark; run with --ignored"]
        fn test_benchmark_small_pie_simd_vs_cuda_stages() {
            let simd = run_small_pie_simd_benchmark();
            simd.print_report("SIMD");

            let cuda_cold = run_small_pie_cuda_benchmark();
            cuda_cold.print_report("CUDA cold");

            let cuda_warm = run_small_pie_cuda_benchmark();
            cuda_warm.print_report("CUDA warm");

            print_speedups(&simd, &cuda_warm);
        }

        // ==================== sn_pie Tests (Large ~25M Steps) ====================

        #[test]
        #[ignore = "sn_pie exceeds GPU memory; needs streaming spill path or multi-GPU"]
        fn test_prove_verify_sn_pie_cuda_multi() {
            let loop_count: usize = std::env::var("PROVE_LOOP_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            for run in 0..loop_count {
                println!("\n========== RUN {} ==========\n", run);
                let input = load_sn_pie_input();
                let prover_params = ProverParameters {
                    channel_hash: ChannelHash::Blake2s,
                    pcs_config: PcsConfig::default(),
                    preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
                    channel_salt: 0,
                    store_polynomials_coefficients: false,
                    include_all_preprocessed_columns: false,
                };
                let timer = std::time::Instant::now();
                let cairo_proof =
                    prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
                println!(
                    "No.{} sn_pie CUDA proof generation time: {:?}",
                    run,
                    timer.elapsed()
                );
                verify_cairo_ex::<Blake2sMerkleChannel>(
                    cairo_proof.into(),
                    prover_params.include_all_preprocessed_columns,
                )
                .unwrap();
            }
        }

        // ==================== SIMD Memory Profile Test ====================

        /// Inline SIMD prove flow with memory profiling at every phase boundary.
        /// Measures host RSS to estimate GPU memory requirements.
        #[test]
        #[ignore = "sn_pie SIMD memory profiling; run manually with --ignored"]
        fn test_prove_verify_sn_pie_simd_mem_profile() {
            use std::sync::Arc;

            use cairo_air::air::CairoComponents;
            use cairo_air::claims::lookup_sum;
            use cairo_air::relations::CommonLookupElements;
            use cairo_air::verifier::{verify_cairo_ex, INTERACTION_POW_BITS};
            use cairo_air::PreProcessedTraceVariant;
            use num_traits::Zero;
            use stwo::core::channel::{Channel, MerkleChannel};
            use stwo::core::fields::qm31::SecureField;
            use stwo::core::pcs::PcsConfig;
            use stwo::core::poly::circle::CanonicCoset;
            use stwo::core::proof_of_work::GrindOps;
            use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
            use stwo::prover::backend::simd::SimdBackend;
            use stwo::prover::poly::circle::PolyOps;
            use stwo::prover::{prove_ex, CommitmentSchemeProver};

            use crate::mem_profile::MemProfiler;
            use crate::prover::{ChannelHash, ProverParameters, LOG_MAX_ROWS};
            use crate::utils::cairo_provers;
            use crate::witness::cairo::create_cairo_claim_generator;
            use crate::witness::preprocessed_trace::gen_trace;
            use crate::witness::utils::witness_trace_cells;

            type MC = Blake2sMerkleChannel;

            let mut mp = MemProfiler::new();
            mp.snap("START (before input load)");

            // Load sn_pie input.
            let input = load_sn_pie_input();
            mp.snap("After load_sn_pie_input()");

            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig::default(),
                preprocessed_trace: PreProcessedTraceVariant::Canonical,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let ProverParameters {
                channel_hash: _,
                channel_salt,
                pcs_config,
                preprocessed_trace,
                store_polynomials_coefficients: _,
                include_all_preprocessed_columns,
            } = prover_params;

            // Twiddles.
            let max_domain_size = {
                let cairo_air_log_degree_bound = 1;
                if let Some(lifting_log_size) = pcs_config.lifting_log_size {
                    lifting_log_size
                } else {
                    LOG_MAX_ROWS
                        + std::cmp::max(
                            cairo_air_log_degree_bound,
                            pcs_config.fri_config.log_blowup_factor,
                        )
                }
            };
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(max_domain_size)
                    .circle_domain()
                    .half_coset,
            );
            mp.snap("After precompute_twiddles()");

            // Setup commitment scheme.
            let channel = &mut <MC as MerkleChannel>::C::default();
            channel.mix_felts(&[channel_salt.into()]);
            pcs_config.mix_into(channel);
            let mut commitment_scheme =
                CommitmentSchemeProver::<SimdBackend, MC>::new(pcs_config, &twiddles);
            // Don't store coefficients for SIMD profiling (saves memory).
            let _ = &mut commitment_scheme;
            mp.snap("After CommitmentSchemeProver::new()");

            // Preprocessed trace.
            let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
            mp.snap("After to_preprocessed_trace()");

            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(gen_trace(preprocessed_trace.clone()));
            mp.snap("After preprocessed extend_evals()");

            tree_builder.commit(channel);
            mp.snap("After preprocessed commit()");

            // Create claim generator (runs Cairo VM adapter).
            let cairo_claim_generator =
                create_cairo_claim_generator(input, preprocessed_trace.clone());
            mp.snap("After create_cairo_claim_generator()");

            // Base trace.
            let mut tree_builder = commitment_scheme.tree_builder();
            let (claim, interaction_generator) =
                cairo_claim_generator.write_trace(&mut tree_builder);
            mp.snap("After base trace write_trace()");

            eprintln!(
                "Witness trace cells: {:?}",
                witness_trace_cells(&claim, &preprocessed_trace)
            );

            claim.mix_into(channel);
            tree_builder.commit(channel);
            mp.snap("After base trace commit()");

            // PoW grinding.
            let interaction_pow = SimdBackend::grind(channel, INTERACTION_POW_BITS);
            channel.mix_u64(interaction_pow);
            let interaction_elements = CommonLookupElements::draw(channel);
            mp.snap("After PoW grind + draw elements");

            // Interaction trace.
            let mut tree_builder = commitment_scheme.tree_builder();
            let interaction_claim = interaction_generator
                .write_interaction_trace(&mut tree_builder, &interaction_elements);
            mp.snap("After interaction write_interaction_trace()");

            debug_assert_eq!(
                lookup_sum(&claim, &interaction_elements, &interaction_claim),
                SecureField::zero()
            );

            interaction_claim.mix_into(channel);
            tree_builder.commit(channel);
            mp.snap("After interaction trace commit()");

            // Component provers.
            let component_builder = CairoComponents::new(
                &claim,
                &interaction_elements,
                &interaction_claim,
                &preprocessed_trace.ids(),
            );
            let components = cairo_provers(&component_builder);
            mp.snap("After component provers setup");

            // Prove STARK.
            let proof = prove_ex::<SimdBackend, _>(
                &components,
                channel,
                commitment_scheme,
                include_all_preprocessed_columns,
            )
            .unwrap();
            mp.snap("After prove_ex() (STARK proof done)");

            // Verify.
            let cairo_proof = cairo_air::CairoProof {
                claim,
                interaction_pow,
                interaction_claim,
                extended_stark_proof: proof,
                channel_salt,
                preprocessed_trace_variant: prover_params.preprocessed_trace,
            };
            verify_cairo_ex::<MC>(cairo_proof.into(), include_all_preprocessed_columns).unwrap();
            mp.snap("After verify (DONE)");

            mp.print_delta_report();
        }

        /// Same as sn_pie profiler but for pie_10_transfers.
        /// Run: cargo test --release -p stwo-cairo-prover test_prove_verify_pie10_simd_mem_profile
        /// -- \        --nocapture --test-threads=1 --ignored
        #[test]
        #[ignore = "pie_10_transfers SIMD memory profiling; run manually with --ignored"]
        fn test_prove_verify_pie10_simd_mem_profile() {
            use std::sync::Arc;

            use cairo_air::air::CairoComponents;
            use cairo_air::claims::lookup_sum;
            use cairo_air::relations::CommonLookupElements;
            use cairo_air::verifier::{verify_cairo_ex, INTERACTION_POW_BITS};
            use cairo_air::PreProcessedTraceVariant;
            use num_traits::Zero;
            use stwo::core::channel::{Channel, MerkleChannel};
            use stwo::core::fields::qm31::SecureField;
            use stwo::core::pcs::PcsConfig;
            use stwo::core::poly::circle::CanonicCoset;
            use stwo::core::proof_of_work::GrindOps;
            use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
            use stwo::prover::backend::simd::SimdBackend;
            use stwo::prover::poly::circle::PolyOps;
            use stwo::prover::{prove_ex, CommitmentSchemeProver};

            use crate::mem_profile::MemProfiler;
            use crate::prover::{ChannelHash, ProverParameters, LOG_MAX_ROWS};
            use crate::utils::cairo_provers;
            use crate::witness::cairo::create_cairo_claim_generator;
            use crate::witness::preprocessed_trace::gen_trace;
            use crate::witness::utils::witness_trace_cells;

            type MC = Blake2sMerkleChannel;

            let mut mp = MemProfiler::new();
            mp.snap("START (before input load)");

            let input = load_pie_10_transfers_input();
            mp.snap("After load_pie_10_transfers_input()");

            let prover_params = ProverParameters {
                channel_hash: ChannelHash::Blake2s,
                pcs_config: PcsConfig::default(),
                preprocessed_trace: PreProcessedTraceVariant::Canonical,
                channel_salt: 0,
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let ProverParameters {
                channel_hash: _,
                channel_salt,
                pcs_config,
                preprocessed_trace,
                store_polynomials_coefficients: _,
                include_all_preprocessed_columns,
            } = prover_params;

            let max_domain_size = {
                let cairo_air_log_degree_bound = 1;
                if let Some(lifting_log_size) = pcs_config.lifting_log_size {
                    lifting_log_size
                } else {
                    LOG_MAX_ROWS
                        + std::cmp::max(
                            cairo_air_log_degree_bound,
                            pcs_config.fri_config.log_blowup_factor,
                        )
                }
            };
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(max_domain_size)
                    .circle_domain()
                    .half_coset,
            );
            mp.snap("After precompute_twiddles()");

            let channel = &mut <MC as MerkleChannel>::C::default();
            channel.mix_felts(&[channel_salt.into()]);
            pcs_config.mix_into(channel);
            let mut commitment_scheme =
                CommitmentSchemeProver::<SimdBackend, MC>::new(pcs_config, &twiddles);
            mp.snap("After CommitmentSchemeProver::new()");

            let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
            mp.snap("After to_preprocessed_trace()");

            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(gen_trace(preprocessed_trace.clone()));
            mp.snap("After preprocessed extend_evals()");

            tree_builder.commit(channel);
            mp.snap("After preprocessed commit()");

            let cairo_claim_generator =
                create_cairo_claim_generator(input, preprocessed_trace.clone());
            mp.snap("After create_cairo_claim_generator()");

            let mut tree_builder = commitment_scheme.tree_builder();
            let (claim, interaction_generator) =
                cairo_claim_generator.write_trace(&mut tree_builder);
            mp.snap("After base trace write_trace()");

            eprintln!(
                "Witness trace cells: {:?}",
                witness_trace_cells(&claim, &preprocessed_trace)
            );

            claim.mix_into(channel);
            tree_builder.commit(channel);
            mp.snap("After base trace commit()");

            let interaction_pow = SimdBackend::grind(channel, INTERACTION_POW_BITS);
            channel.mix_u64(interaction_pow);
            let interaction_elements = CommonLookupElements::draw(channel);
            mp.snap("After PoW grind + draw elements");

            let mut tree_builder = commitment_scheme.tree_builder();
            let interaction_claim = interaction_generator
                .write_interaction_trace(&mut tree_builder, &interaction_elements);
            mp.snap("After interaction write_interaction_trace()");

            debug_assert_eq!(
                lookup_sum(&claim, &interaction_elements, &interaction_claim),
                SecureField::zero()
            );

            interaction_claim.mix_into(channel);
            tree_builder.commit(channel);
            mp.snap("After interaction trace commit()");

            let component_builder = CairoComponents::new(
                &claim,
                &interaction_elements,
                &interaction_claim,
                &preprocessed_trace.ids(),
            );
            let components = cairo_provers(&component_builder);
            mp.snap("After component provers setup");

            let proof = prove_ex::<SimdBackend, _>(
                &components,
                channel,
                commitment_scheme,
                include_all_preprocessed_columns,
            )
            .unwrap();
            mp.snap("After prove_ex() (STARK proof done)");

            let cairo_proof = cairo_air::CairoProof {
                claim,
                interaction_pow,
                interaction_claim,
                extended_stark_proof: proof,
                channel_salt,
                preprocessed_trace_variant: prover_params.preprocessed_trace,
            };
            verify_cairo_ex::<MC>(cairo_proof.into(), include_all_preprocessed_columns).unwrap();
            mp.snap("After verify (DONE)");

            mp.print_delta_report();
        }

        // ==================== GPU Memory Estimation Tool ====================

        /// Estimates GPU memory requirements from ProverInput.
        /// Uses the exact code path (write_trace + witness_trace_cells) for precise
        /// cell counting, then applies calibrated multiplier for GPU peak.
        ///
        /// Run: cargo test --release -p stwo-cairo-prover test_gpu_memory_estimator -- \
        ///        --nocapture --test-threads=1 --ignored
        #[test]
        #[ignore = "GPU memory estimation tool; run manually with --ignored"]
        fn test_gpu_memory_estimator() {
            use std::sync::Arc;

            use cairo_air::PreProcessedTraceVariant;
            use stwo::core::pcs::PcsConfig;
            use stwo::core::poly::circle::CanonicCoset;
            use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
            use stwo::prover::backend::simd::SimdBackend;
            use stwo::prover::poly::circle::PolyOps;
            use stwo::prover::CommitmentSchemeProver;
            use stwo_cairo_adapter::ExecutionResources;

            use crate::prover::LOG_MAX_ROWS;
            use crate::witness::cairo::create_cairo_claim_generator;
            use crate::witness::utils::witness_trace_cells;

            /// Load input, run write_trace to get CairoClaim, compute exact trace cells.
            fn estimate_gpu_memory(input: stwo_cairo_adapter::ProverInput, label: &str) -> f64 {
                let er = ExecutionResources::from_prover_input(&input);

                // Print ExecutionResources summary
                eprintln!("\n{}", "=".repeat(80));
                eprintln!("  {}", label);
                eprintln!("{}", "=".repeat(80));

                // Opcode counts
                eprintln!("\n--- Opcode Counts ---");
                let mut opcodes: Vec<_> = er.opcodes_instance_counter.iter().collect();
                opcodes.sort_by(|a, b| b.1.cmp(a.1));
                for (name, count) in &opcodes {
                    if **count > 0 {
                        eprintln!("  {:<40} {:>10}", name, count);
                    }
                }

                // Builtin counts
                eprintln!("\n--- Builtin Counts ---");
                let mut builtins: Vec<_> = er.builtin_instance_counter.iter().collect();
                builtins.sort_by(|a, b| b.1.cmp(a.1));
                for (name, count) in &builtins {
                    if **count > 0 {
                        eprintln!("  {:<40} {:>10}", name, count);
                    }
                }

                // Memory table sizes
                eprintln!("\n--- Memory Table Sizes ---");
                eprintln!(
                    "  memory_address_to_id: {:>12}",
                    er.memory_tables_sizes.memory_address_to_id
                );
                eprintln!(
                    "  memory_id_to_big:     {:>12}",
                    er.memory_tables_sizes.memory_id_to_big
                );
                eprintln!(
                    "  memory_id_to_small:   {:>12}",
                    er.memory_tables_sizes.memory_id_to_small
                );
                eprintln!("  verify_instruction:   {:>12}", er.verify_instruction);

                // Derived sizes for GPU resident data
                let big_count = er.memory_tables_sizes.memory_id_to_big;
                let small_count = er.memory_tables_sizes.memory_id_to_small;
                let big_size = std::cmp::max(big_count.next_multiple_of(16), 16);
                let small_size =
                    std::cmp::max(small_count.next_multiple_of(16), 16).next_power_of_two();
                let mem_id_to_big_bytes = (9 * big_size * 4 + small_size * 20) as f64 / 1e9;
                let addr_count = er.memory_tables_sizes.memory_address_to_id;
                let mem_addr_to_id_bytes = (2 * addr_count.saturating_sub(1) * 4) as f64 / 1e9;

                eprintln!("\n--- GPU Resident Data ---");
                eprintln!("  mem_id_to_big constructor: {:.2} GB", mem_id_to_big_bytes);
                eprintln!(
                    "  mem_addr_to_id data:       {:.2} GB",
                    mem_addr_to_id_bytes
                );
                eprintln!("  twiddles:                  0.26 GB");

                // Run exact trace cell computation
                let preprocessed_trace =
                    Arc::new(PreProcessedTraceVariant::Canonical.to_preprocessed_trace());
                let cairo_claim_generator =
                    create_cairo_claim_generator(input, preprocessed_trace.clone());

                let pcs_config = PcsConfig::default();
                let max_domain_size =
                    LOG_MAX_ROWS + std::cmp::max(1, pcs_config.fri_config.log_blowup_factor);
                let twiddles = SimdBackend::precompute_twiddles(
                    CanonicCoset::new(max_domain_size)
                        .circle_domain()
                        .half_coset,
                );
                let mut commitment_scheme = CommitmentSchemeProver::<
                    SimdBackend,
                    Blake2sMerkleChannel,
                >::new(pcs_config, &twiddles);
                let mut tree_builder = commitment_scheme.tree_builder();
                let (claim, _) = cairo_claim_generator.write_trace(&mut tree_builder);
                let cells = witness_trace_cells(&claim, &preprocessed_trace);

                let preprocessed = cells[0];
                let base = cells[1];
                let interaction = cells[2];
                let total = preprocessed + base + interaction;

                eprintln!("\n--- Exact Trace Cells ---");
                eprintln!(
                    "  Preprocessed: {:>14} ({:>6.1} GB raw)",
                    preprocessed,
                    preprocessed as f64 * 4.0 / 1e9
                );
                eprintln!(
                    "  Base trace:   {:>14} ({:>6.1} GB raw)",
                    base,
                    base as f64 * 4.0 / 1e9
                );
                eprintln!(
                    "  Interaction:  {:>14} ({:>6.1} GB raw)",
                    interaction,
                    interaction as f64 * 4.0 / 1e9
                );
                eprintln!(
                    "  Total:        {:>14} ({:>6.1} GB raw)",
                    total,
                    total as f64 * 4.0 / 1e9
                );

                // GPU peak estimation (calibrated from SIMD HWM / raw_data):
                //   Large workloads (>1B cells): 3.1× (sn_pie: 96.8 GB / 31.3 GB)
                //   Small workloads (<1B cells): 7.3× (pie_10_transfers: 23.9 GB / 3.3 GB)
                //   Preprocessed trace dominates small workloads → higher overhead ratio.
                let raw_gb = total as f64 * 4.0 / 1e9;
                let multiplier = if total > 1_000_000_000 { 3.1 } else { 7.3 };
                let estimated_peak = raw_gb * multiplier;

                eprintln!("\n--- GPU Peak Estimate ---");
                eprintln!("  Formula: total_cells × 4B × {:.1}", multiplier);
                eprintln!("  Estimated GPU peak: {:.1} GB", estimated_peak);

                // Fit assessment
                eprintln!("\n  Fit assessment:");
                if estimated_peak < 24.0 * 0.70 {
                    eprintln!("    → SAFE on RTX 4090 (24 GB) ✓");
                } else if estimated_peak < 24.0 * 0.95 {
                    eprintln!("    → TIGHT on RTX 4090 (24 GB), safe on RTX 5090 (32 GB)");
                } else if estimated_peak < 32.0 * 0.95 {
                    eprintln!("    → FITS on RTX 5090 (32 GB), needs streaming on 4090");
                } else if estimated_peak < 80.0 * 0.95 {
                    eprintln!("    → NEEDS streaming on consumer GPUs, may fit A100 (80 GB)");
                } else {
                    eprintln!("    → NEEDS streaming on all GPUs (peak > 80 GB)");
                }

                estimated_peak
            }

            // ---- Run both workloads ----
            let pie10_input = load_pie_10_transfers_input();
            let pie10_peak = estimate_gpu_memory(pie10_input, "pie_10_transfers");

            let snpie_input = load_sn_pie_input();
            let snpie_peak = estimate_gpu_memory(snpie_input, "sn_pie");

            // ---- Summary ----
            eprintln!("\n{}", "=".repeat(80));
            eprintln!("  CROSS-VALIDATION SUMMARY");
            eprintln!("{}", "=".repeat(80));
            eprintln!("{:<30} {:>18} {:>18}", "", "pie_10_transfers", "sn_pie");
            eprintln!("{}", "-".repeat(70));
            eprintln!(
                "{:<30} {:>18} {:>18}",
                "Estimated peak (GB)",
                format!("{:.1}", pie10_peak),
                format!("{:.1}", snpie_peak)
            );
            eprintln!(
                "{:<30} {:>18} {:>18}",
                "Measured SIMD HWM (GB)", "23.9", "96.8"
            );
            eprintln!(
                "{:<30} {:>18} {:>18}",
                "Estimate/Measured",
                format!("{:.2}", pie10_peak / 23.9),
                format!("{:.2}", snpie_peak / 96.8)
            );
        }
    }
}
