//! Estimates GPU memory requirements for a Cairo workload without running the prover.
//!
//! Supports two input modes:
//! - `--pie`: CairoPie file (zip or directory) → run through bootloader → ProverInput
//! - `--prover_input`: Pre-serialized ProverInput JSON file
//!
//! ### Examples
//!
//! ```bash
//! # From a PIE file
//! cargo run --release --bin estimate_gpu_memory -- --pie /path/to/my.pie.zip
//!
//! # From a serialized ProverInput JSON
//! cargo run --release --bin estimate_gpu_memory -- --prover_input /path/to/input.json
//!
//! # Quick mode (skip exact computation, use heuristic only)
//! cargo run --release --bin estimate_gpu_memory -- --pie /path/to/my.pie.zip --quick
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use cairo_air::PreProcessedTraceVariant;
use clap::Parser;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::CommitmentSchemeProver;
use stwo_cairo_adapter::{ExecutionResources, ProverInput};
use stwo_cairo_prover::witness::cairo::create_cairo_claim_generator;
use stwo_cairo_prover::witness::utils::witness_trace_cells;

const LOG_MAX_ROWS: u32 = 27;

/// GPU memory estimation tool for stwo-cairo workloads.
#[derive(Parser, Debug)]
#[command(name = "estimate_gpu_memory")]
#[command(about = "Estimate GPU memory requirements for a Cairo workload")]
struct Args {
    /// Path to a CairoPie file (zip or directory).
    #[arg(long = "pie")]
    pie_path: Option<PathBuf>,

    /// Path to a pre-serialized ProverInput JSON file.
    #[arg(long = "prover_input")]
    prover_input_path: Option<PathBuf>,

    /// Quick mode: skip exact trace computation, use heuristic estimate only.
    /// Faster (~1s) but less accurate than exact mode (~15-30s).
    #[arg(long)]
    quick: bool,

    /// Output JSON with structured results (for piping to other tools).
    #[arg(long)]
    json: bool,

    /// Path to simple_bootloader_compiled.json.
    /// If not set, uses BOOTLOADER_PATH env var, then falls back to auto-detection
    /// relative to the source tree.
    #[arg(long = "bootloader")]
    bootloader_path: Option<PathBuf>,
}

const BOOTLOADER_RELATIVE: &str = "proving-utils/crates/cairo-program-runner-lib/resources/compiled_programs/bootloaders/simple_bootloader_compiled.json";

/// Resolve the bootloader path. Priority:
/// 1. --bootloader CLI arg
/// 2. BOOTLOADER_PATH env var
/// 3. Auto-detect relative to CARGO_MANIFEST_DIR (source tree)
fn resolve_bootloader_path(cli_override: Option<&PathBuf>) -> Result<PathBuf> {
    // 1. CLI arg
    if let Some(p) = cli_override {
        if p.exists() {
            return Ok(p.clone());
        }
        bail!(
            "Bootloader not found at --bootloader path: {}\n\
             Provide a valid path to simple_bootloader_compiled.json",
            p.display()
        );
    }

    // 2. Env var
    if let Ok(env_path) = std::env::var("BOOTLOADER_PATH") {
        let p = PathBuf::from(&env_path);
        if p.exists() {
            return Ok(p);
        }
        bail!(
            "Bootloader not found at BOOTLOADER_PATH={}\n\
             Provide a valid path to simple_bootloader_compiled.json",
            env_path
        );
    }

    // 3. Auto-detect: walk up from CARGO_MANIFEST_DIR looking for proving-utils/
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dir = manifest_dir.as_path();
    for _ in 0..6 {
        let candidate = dir.join(BOOTLOADER_RELATIVE);
        if candidate.exists() {
            return Ok(candidate);
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }

    bail!(
        "Could not find simple_bootloader_compiled.json.\n\
         Searched up from: {}\n\n\
         Fix: set BOOTLOADER_PATH env var or pass --bootloader <path>\n\
         Example:\n  \
           export BOOTLOADER_PATH=/path/to/simple_bootloader_compiled.json\n  \
           cargo run --release --bin estimate_gpu_memory -- --pie my.pie.zip",
        manifest_dir.display()
    );
}

/// Load a CairoPie file and produce ProverInput via bootloader execution.
fn load_pie_input(
    pie_path: &std::path::Path,
    bootloader_override: Option<&PathBuf>,
) -> Result<ProverInput> {
    use std::rc::Rc;

    use cairo_program_runner_lib::tasks::create_pie_task;
    use cairo_program_runner_lib::types::{HashFunc, RunMode};
    use cairo_program_runner_lib::{
        cairo_run_program, ProgramInput, SimpleBootloaderInput, TaskSpec,
    };
    use cairo_vm::types::layout_name::LayoutName;
    use cairo_vm::types::program::Program;
    use stwo_cairo_adapter::adapter::adapt;

    // Check PIE file exists first
    if !pie_path.exists() {
        bail!(
            "PIE file not found: {}\n\
             Check the path and try again.",
            pie_path.display()
        );
    }

    let task = create_pie_task(pie_path)
        .with_context(|| format!("Failed to load CairoPie from: {}", pie_path.display()))?;
    let task_spec = TaskSpec {
        task: Rc::new(task),
        program_hash_function: HashFunc::Blake,
    };
    let bootloader_input = SimpleBootloaderInput {
        fact_topologies_path: None,
        single_page: true,
        tasks: vec![task_spec],
    };

    let bootloader_path = resolve_bootloader_path(bootloader_override)?;
    let bootloader_program = Program::from_file(bootloader_path.as_path(), Some("main"))
        .with_context(|| {
            format!(
                "Failed to load bootloader from: {}",
                bootloader_path.display()
            )
        })?;

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
    .context("Failed to run PIE through bootloader")?;

    adapt(&runner).context("Failed to adapt runner to ProverInput")
}

struct EstimationResult {
    label: String,
    // ExecutionResources summary
    n_steps: usize,
    memory_address_to_id: usize,
    memory_id_to_big: usize,
    memory_id_to_small: usize,
    verify_instruction: usize,
    // GPU resident data
    mem_id_to_big_gb: f64,
    mem_addr_to_id_gb: f64,
    // Trace cells (exact or heuristic)
    preprocessed_cells: u64,
    base_cells: u64,
    interaction_cells: u64,
    total_cells: u64,
    exact: bool,
    // Peak estimate
    raw_gb: f64,
    multiplier: f64,
    estimated_peak_gb: f64,
}

fn estimate_gpu_memory(input: ProverInput, label: &str, quick: bool) -> EstimationResult {
    let er = ExecutionResources::from_prover_input(&input);

    // Compute n_steps from opcode counts
    let n_steps: usize = er.opcodes_instance_counter.values().sum();

    // GPU resident data sizes
    let big_count = er.memory_tables_sizes.memory_id_to_big;
    let small_count = er.memory_tables_sizes.memory_id_to_small;
    let big_size = std::cmp::max(big_count.next_multiple_of(16), 16);
    let small_size = std::cmp::max(small_count.next_multiple_of(16), 16).next_power_of_two();
    let mem_id_to_big_gb = (9 * big_size * 4 + small_size * 20) as f64 / 1e9;
    let addr_count = er.memory_tables_sizes.memory_address_to_id;
    let mem_addr_to_id_gb = (2 * addr_count.saturating_sub(1) * 4) as f64 / 1e9;

    let (preprocessed_cells, base_cells, interaction_cells, exact) = if quick {
        // Heuristic: rough estimate based on n_steps
        // Calibrated from measured data:
        //   pie_10_transfers: 593K steps → 813M total cells (1370 cells/step)
        //   sn_pie: 31.9M steps → 7815M total cells (245 cells/step)
        // Use conservative per-step estimate + fixed preprocessed overhead
        let preprocessed = 543_000_000u64; // ~543M (constant for canonical preprocessed)
        let base = (n_steps as f64 * 130.0) as u64; // ~130 base cells/step
        let interaction = (n_steps as f64 * 95.0) as u64; // ~95 interaction cells/step
        (preprocessed, base, interaction, false)
    } else {
        // Exact: run write_trace to get precise cell counts
        let preprocessed_trace =
            Arc::new(PreProcessedTraceVariant::Canonical.to_preprocessed_trace());
        let cairo_claim_generator = create_cairo_claim_generator(input, preprocessed_trace.clone());

        let pcs_config = PcsConfig::default();
        let max_domain_size =
            LOG_MAX_ROWS + std::cmp::max(1, pcs_config.fri_config.log_blowup_factor);
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(max_domain_size)
                .circle_domain()
                .half_coset,
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);
        let mut tree_builder = commitment_scheme.tree_builder();
        let (claim, _) = cairo_claim_generator.write_trace(&mut tree_builder);
        let cells = witness_trace_cells(&claim, &preprocessed_trace);

        (cells[0], cells[1], cells[2], true)
    };

    let total_cells = preprocessed_cells + base_cells + interaction_cells;
    let raw_gb = total_cells as f64 * 4.0 / 1e9;

    // Calibrated multiplier (SIMD HWM / raw_data):
    //   Large workloads (>1B cells): 3.1× (sn_pie: 96.8 GB / 31.3 GB)
    //   Small workloads (<1B cells): 7.3× (pie_10_transfers: 23.9 GB / 3.3 GB)
    //   Preprocessed trace dominates small workloads → higher overhead ratio.
    let multiplier = if total_cells > 1_000_000_000 {
        3.1
    } else {
        7.3
    };
    let estimated_peak_gb = raw_gb * multiplier;

    EstimationResult {
        label: label.to_string(),
        n_steps,
        memory_address_to_id: er.memory_tables_sizes.memory_address_to_id,
        memory_id_to_big: er.memory_tables_sizes.memory_id_to_big,
        memory_id_to_small: er.memory_tables_sizes.memory_id_to_small,
        verify_instruction: er.verify_instruction,
        mem_id_to_big_gb,
        mem_addr_to_id_gb,
        preprocessed_cells,
        base_cells,
        interaction_cells,
        total_cells,
        exact,
        raw_gb,
        multiplier,
        estimated_peak_gb,
    }
}

fn print_result(r: &EstimationResult) {
    println!("\n{}", "=".repeat(72));
    println!("  GPU Memory Estimate: {}", r.label);
    println!("{}", "=".repeat(72));

    println!("\n--- Workload Summary ---");
    println!("  Total opcode steps:          {:>14}", r.n_steps);
    println!(
        "  memory_address_to_id count:  {:>14}",
        r.memory_address_to_id
    );
    println!("  memory_id_to_big count:      {:>14}", r.memory_id_to_big);
    println!(
        "  memory_id_to_small count:    {:>14}",
        r.memory_id_to_small
    );
    println!(
        "  verify_instruction count:    {:>14}",
        r.verify_instruction
    );

    println!("\n--- GPU Resident Data ---");
    println!(
        "  mem_id_to_big constructor:   {:>10.2} GB",
        r.mem_id_to_big_gb
    );
    println!(
        "  mem_addr_to_id data:         {:>10.2} GB",
        r.mem_addr_to_id_gb
    );
    println!("  twiddles (fixed):            {:>10.2} GB", 0.26);

    let mode = if r.exact { "EXACT" } else { "HEURISTIC" };
    println!("\n--- Trace Cells ({}) ---", mode);
    println!(
        "  Preprocessed: {:>14}  ({:>6.1} GB raw)",
        r.preprocessed_cells,
        r.preprocessed_cells as f64 * 4.0 / 1e9
    );
    println!(
        "  Base trace:   {:>14}  ({:>6.1} GB raw)",
        r.base_cells,
        r.base_cells as f64 * 4.0 / 1e9
    );
    println!(
        "  Interaction:  {:>14}  ({:>6.1} GB raw)",
        r.interaction_cells,
        r.interaction_cells as f64 * 4.0 / 1e9
    );
    println!(
        "  Total:        {:>14}  ({:>6.1} GB raw)",
        r.total_cells,
        r.total_cells as f64 * 4.0 / 1e9
    );

    println!("\n--- GPU Peak Estimate ---");
    println!("  Formula: total_cells x 4B x {:.1}", r.multiplier);
    println!("  Estimated GPU peak: {:.1} GB", r.estimated_peak_gb);

    // Fit assessment for common GPUs
    println!("\n--- Hardware Fit ---");
    let gpus = [
        ("RTX 4090", 24.0),
        ("RTX 5090", 32.0),
        ("A100 40GB", 40.0),
        ("A100 80GB", 80.0),
        ("H100 80GB", 80.0),
    ];
    for (name, vram) in &gpus {
        let ratio = r.estimated_peak_gb / vram;
        let status = if ratio < 0.70 {
            "SAFE"
        } else if ratio < 0.90 {
            "TIGHT"
        } else if ratio < 1.0 {
            "RISKY (fragmentation may cause OOM)"
        } else {
            "NEEDS STREAMING"
        };
        println!("  {:<14} {:>5.0} GB VRAM  ->  {}", name, vram, status);
    }

    if r.estimated_peak_gb > 32.0 {
        println!("\n  Streaming spill reduces GPU peak to ~20 GB (fits RTX 5090/4090).");
        println!("  Use --cuda-spill flag when proving.");
    }
}

fn print_json(r: &EstimationResult) {
    println!("{{");
    println!("  \"label\": \"{}\",", r.label);
    println!("  \"n_steps\": {},", r.n_steps);
    println!("  \"memory_address_to_id\": {},", r.memory_address_to_id);
    println!("  \"memory_id_to_big\": {},", r.memory_id_to_big);
    println!("  \"memory_id_to_small\": {},", r.memory_id_to_small);
    println!("  \"verify_instruction\": {},", r.verify_instruction);
    println!("  \"exact\": {},", r.exact);
    println!("  \"preprocessed_cells\": {},", r.preprocessed_cells);
    println!("  \"base_cells\": {},", r.base_cells);
    println!("  \"interaction_cells\": {},", r.interaction_cells);
    println!("  \"total_cells\": {},", r.total_cells);
    println!("  \"raw_data_gb\": {:.2},", r.raw_gb);
    println!("  \"multiplier\": {:.1},", r.multiplier);
    println!("  \"estimated_peak_gb\": {:.1},", r.estimated_peak_gb);
    println!("  \"fits_rtx4090\": {},", r.estimated_peak_gb < 24.0 * 0.90);
    println!("  \"fits_rtx5090\": {},", r.estimated_peak_gb < 32.0 * 0.90);
    println!(
        "  \"fits_a100_80gb\": {},",
        r.estimated_peak_gb < 80.0 * 0.90
    );
    println!("  \"needs_streaming\": {}", r.estimated_peak_gb > 32.0);
    println!("}}");
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.pie_path.is_none() && args.prover_input_path.is_none() {
        bail!("Must specify either --pie <path> or --prover_input <path>");
    }

    let (input, label) = if let Some(pie_path) = &args.pie_path {
        let label = pie_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "pie".to_string());
        if !args.json {
            eprintln!("Loading PIE from: {}", pie_path.display());
        }
        (
            load_pie_input(pie_path, args.bootloader_path.as_ref())?,
            label,
        )
    } else {
        let path = args.prover_input_path.as_ref().unwrap();
        let label = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "prover_input".to_string());
        if !args.json {
            eprintln!("Loading ProverInput from: {}", path.display());
        }
        let file = std::fs::File::open(path)?;
        let input: ProverInput = serde_json::from_reader(file)?;
        (input, label)
    };

    if !args.json && !args.quick {
        eprintln!("Running exact trace cell computation (this takes ~15-30s)...");
    }

    let result = estimate_gpu_memory(input, &label, args.quick);

    if args.json {
        print_json(&result);
    } else {
        print_result(&result);
    }

    Ok(())
}
