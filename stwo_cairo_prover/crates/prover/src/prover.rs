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
    let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(gen_trace(preprocessed_trace.clone()));
    tree_builder.commit(channel);

    // Run Cairo.
    let cairo_claim_generator = create_cairo_claim_generator(input, preprocessed_trace.clone());
    // Base trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    let span = span!(Level::INFO, "Base trace").entered();
    let (claim, interaction_generator) = cairo_claim_generator.write_trace(&mut tree_builder);
    span.exit();

    claim.mix_into(channel);
    tree_builder.commit(channel);

    // Draw interaction elements.
    let interaction_pow = SimdBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow);
    let interaction_elements = CommonLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction trace").entered();
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
    let span = span!(Level::INFO, "Prove STARKs").entered();
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
    // CUDA path uses Option<u64> salt semantics: only mix if non-zero.
    if channel_salt != 0 {
        channel.mix_u64(channel_salt as u64);
    }
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

    let cuda_channel_salt = if channel_salt != 0 {
        Some(channel_salt as u64)
    } else {
        None
    };

    Ok(CairoProofCuda {
        claim,
        interaction_pow,
        interaction_claim,
        stark_proof: proof,
        channel_salt: cuda_channel_salt,
        preprocessed_trace_variant: prover_params.preprocessed_trace,
    })
}

/// Native CUDA proving path with per-component SIMD→CUDA dispatch.
///
/// Unlike `prove_cairo_cuda_v0` (which batch-collects all SIMD traces then converts),
/// this path converts each component's trace to CUDA inline via `SimdToCudaBridge`,
/// enabling incremental upgrade to native CUDA kernels per-component.
///
/// Currently all components use SIMD fallback — functionally equivalent to V0 but
/// structured for per-component CUDA acceleration.
pub fn prove_cairo_cuda<MC: MerkleChannel>(
    input: ProverInput,
    prover_params: ProverParameters,
) -> Result<CairoProofCuda<MC::H>, ProvingError>
where
    stwo::prover::backend::cuda::CudaBackend: BackendForChannel<MC>,
    SimdBackend: BackendForChannel<MC>,
{
    use std::time::Instant;

    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::prove;

    // Poseidon workloads now use SIMD hybrid pipeline (poseidon_builtin + poseidon_aggregator
    // via SIMD, chain sub-components via CUDA). No V0 fallback needed.

    let _span = span!(Level::INFO, "prove_cairo_cuda").entered();
    let ProverParameters {
        channel_hash: _,
        channel_salt,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients,
        include_all_preprocessed_columns: _,
    } = prover_params;

    let total_start = Instant::now();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] START");

    let cairo_air_log_degree_bound = 1;
    let max_domain_size = LOG_MAX_ROWS
        + std::cmp::max(
            cairo_air_log_degree_bound,
            pcs_config.fri_config.log_blowup_factor,
        );

    // Twiddles for CUDA backend.
    let t = Instant::now();
    tracing::info!("[CUDA] Computing twiddles for CUDA backend");
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(max_domain_size)
            .circle_domain()
            .half_coset,
    );
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After twiddles");

    // Protocol setup.
    let channel = &mut MC::C::default();
    if channel_salt != 0 {
        channel.mix_u64(channel_salt as u64);
    }
    pcs_config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<CudaBackend, MC>::new(pcs_config, &twiddles);
    // Always store polynomial coefficients for the native CUDA path.
    // This enables fast batched GPU OODS evaluation via batch_eval_at_point()
    // (~15 kernel launches) instead of slow per-polynomial CPU barycentric weights
    // (~400 individual evaluations). +50% GPU memory for coefficients is acceptable
    // since this path already assumes all data fits in GPU memory.
    commitment_scheme.set_store_polynomials_coefficients();
    eprintln!("[PROFILE] twiddles + scheme:          {:>6}ms", t.elapsed().as_millis());

    // Preprocessed trace — generate on GPU.
    let t = Instant::now();
    tracing::info!("[CUDA] Generating preprocessed trace");
    let preprocessed_trace = Arc::new(preprocessed_trace.to_preprocessed_trace());
    let evals =
        crate::witness::preprocessed_trace_cuda::gen_preprocessed_trace_cuda(&preprocessed_trace);
    let t_gen = t.elapsed().as_millis();
    let t2 = Instant::now();
    let polys =
        crate::witness::preprocessed_trace_cuda::interpolate_columns_batched(evals, &twiddles);
    let t_interp = t2.elapsed().as_millis();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After preprocessed interpolation");
    let t3 = Instant::now();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(polys);
    tree_builder.commit(channel);
    let t_commit = t3.elapsed().as_millis();
    eprintln!("[PROFILE] preprocessed: gen            {:>6}ms", t_gen);
    eprintln!("[PROFILE] preprocessed: interp         {:>6}ms", t_interp);
    eprintln!("[PROFILE] preprocessed: commit         {:>6}ms", t_commit);
    eprintln!("[PROFILE] preprocessed: total          {:>6}ms", t.elapsed().as_millis());
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After preprocessed commit");

    // Base trace — native CUDA generation (per-component GPU kernels, no SIMD bridge).
    let t = Instant::now();
    tracing::info!("[CUDA] Generating base trace (native CUDA)");
    let native_cuda_gen = crate::witness::cairo_cuda::create_native_cairo_cuda_claim_generator(
        input,
        preprocessed_trace.clone(),
    );
    let t_ctor = t.elapsed().as_millis();
    eprintln!("[PROFILE] base_trace: ctor             {:>6}ms", t_ctor);
    let mut tree_builder = commitment_scheme.tree_builder();
    let t2 = Instant::now();
    let span = span!(Level::INFO, "Base trace (CUDA native)").entered();
    let (claim, interaction_generator) =
        native_cuda_gen.write_trace(&mut tree_builder);
    span.exit();
    let t_write = t2.elapsed().as_millis();
    eprintln!("[PROFILE] base_trace: write_trace      {:>6}ms", t_write);
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After base trace extend");

    claim.mix_into(channel);
    let t3 = Instant::now();
    tree_builder.commit(channel);
    let t_bt_commit = t3.elapsed().as_millis();
    eprintln!("[PROFILE] commit: base_trace           {:>6}ms", t_bt_commit);
    eprintln!("[PROFILE] base_trace: total            {:>6}ms", t.elapsed().as_millis());
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After base trace commit");

    // Draw interaction elements.
    let t = Instant::now();
    let interaction_pow = CudaBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow);
    eprintln!("[PROFILE] proof_of_work:               {:>6}ms", t.elapsed().as_millis());
    let interaction_elements = CommonLookupElements::draw(channel);

    // Interaction trace — native CUDA generation (per-component GPU kernels).
    let t = Instant::now();
    tracing::info!("[CUDA] Generating interaction trace (native CUDA)");
    let mut tree_builder = commitment_scheme.tree_builder();
    let span = span!(Level::INFO, "Interaction trace (CUDA native)").entered();
    let interaction_claim = interaction_generator
        .write_interaction_trace(&mut tree_builder, &interaction_elements);
    span.exit();
    let t_it_write = t.elapsed().as_millis();
    eprintln!("[PROFILE] interaction_trace:            {:>6}ms", t_it_write);
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
    let t2 = Instant::now();
    tree_builder.commit(channel);
    let t_it_commit = t2.elapsed().as_millis();
    eprintln!("[PROFILE] commit: interaction           {:>6}ms", t_it_commit);
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
    let t = Instant::now();
    let span = span!(Level::INFO, "Prove STARKs (CUDA native)").entered();
    stwo::stwo_cuda::print_cuda_memory("[CUDA] Before prove()");
    let proof = prove::<CudaBackend, _>(&components, channel, commitment_scheme)?;
    stwo::stwo_cuda::print_cuda_memory("[CUDA] After prove()");
    span.exit();
    eprintln!("[PROFILE] prove: stark                 {:>6}ms", t.elapsed().as_millis());
    eprintln!("[PROFILE] TOTAL                        {:>6}ms", total_start.elapsed().as_millis());

    event!(
        name: "component_info",
        Level::DEBUG,
        "Components: {}",
        component_builder
    );

    let cuda_channel_salt = if channel_salt != 0 {
        Some(channel_salt as u64)
    } else {
        None
    };

    Ok(CairoProofCuda {
        claim,
        interaction_pow,
        interaction_claim,
        stark_proof: proof,
        channel_salt: cuda_channel_salt,
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
        use cairo_air::verifier::verify_cairo_cuda;
        use cairo_air::PreProcessedTraceVariant;
        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
        use stwo_cairo_adapter::ProverInput;
        use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
        use test_log::test;

        use crate::prover::{
            prove_cairo_cuda, prove_cairo_cuda_v0, ChannelHash, ProverParameters,
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

            let bootloader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("proving-utils/crates/cairo-program-runner-lib/resources/compiled_programs/bootloaders/simple_bootloader_compiled.json");
            let bootloader_program =
                Program::from_file(bootloader_path.as_path(), Some("main"))
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
            verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
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
            verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
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
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
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
                store_polynomials_coefficients: false,
                include_all_preprocessed_columns: false,
            };
            let cairo_proof =
                prove_cairo_cuda::<Blake2sMerkleChannel>(input, prover_params).unwrap();
            verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
        }

        // ==================== PIE File Tests ====================

        fn load_pie_10_transfers_input() -> ProverInput {
            load_pie_input(std::path::Path::new(
                "/root/starkware/opensource/cairo_pie_10_transfers_with_6_ecop.zip",
            ))
        }

        fn load_sn_pie_input() -> ProverInput {
            load_pie_input(std::path::Path::new("/root/starkware/opensource/sn_pie"))
        }

        #[test]
        fn test_prove_verify_pie_cuda_once() {
            let input = load_pie_10_transfers_input();
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
            println!("CUDA proof generation time: {:?}", timer.elapsed());
            verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
        }

        #[test]
        fn test_prove_verify_pie_cuda_multi() {
            let loop_count: usize = std::env::var("PROVE_LOOP_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            for run in 0..loop_count {
                println!("\n========== RUN {} ==========\n", run);
                let input = load_pie_10_transfers_input();
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
                let prove_time = timer.elapsed();
                verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
                println!("No.{} CUDA proof generation time: {:?}", run, prove_time);
            }
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
                verify_cairo_cuda::<Blake2sMerkleChannel>(cairo_proof).unwrap();
            }
        }
    }
}
