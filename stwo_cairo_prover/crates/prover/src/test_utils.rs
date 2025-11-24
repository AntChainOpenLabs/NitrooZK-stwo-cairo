//! Test utilities for the prover crate.
//!
//! This module provides helper functions for creating test inputs and running tests.

use cairo_lang_casm::instructions::Instruction;
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor;
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::types::program::Program;
use cairo_vm::types::relocatable::MaybeRelocatable;
use cairo_vm::vm::runners::cairo_runner::CairoRunner;
use itertools::Itertools;
use stwo_cairo_adapter::adapter::adapter;
use stwo_cairo_adapter::ProverInput;
use std::fs::read_to_string;
use std::path::PathBuf;

/// Converts a list of CASM instructions into a Cairo VM program.
/// Returns both the program and its length.
pub fn program_from_casm(casm: Vec<Instruction>) -> (Program, usize) {
    let felt_code = casm
        .into_iter()
        .flat_map(|instruction| instruction.assemble().encode())
        .map(|felt| MaybeRelocatable::Int(felt.into()))
        .collect_vec();
    let program_len = felt_code.len();
    let program = Program::new_for_proof(
        vec![],
        felt_code,
        0,
        1,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .expect("Program creation failed");
    (program, program_len)
}

/// Translates plain CASM instructions into a ProverInput by running the program
/// and extracting the memory and state transitions.
pub fn input_from_plain_casm(casm: Vec<Instruction>) -> ProverInput {
    let (program, program_len) = program_from_casm(casm);

    let mut runner =
        CairoRunner::new(&program, LayoutName::all_cairo_stwo, None, true, true, true)
            .expect("Runner creation failed");
    runner.initialize(true).expect("Initialization failed");
    runner
        .run_until_pc(
            (runner.program_base.unwrap() + program_len).unwrap(),
            &mut BuiltinHintProcessor::new_empty(),
        )
        .expect("Run failed");

    adapter(&runner)
}

/// Gets the path to a prover input test file.
pub fn get_prover_input_path(test_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/")
        .join(test_name)
        .join("prover_input.json")
}

/// Loads a ProverInput from a compiled Cairo program test file.
pub fn prover_input_from_compiled_cairo_program(test_name: &str) -> ProverInput {
    let path = get_prover_input_path(test_name);
    let json_content = read_to_string(path).expect("Failed to read test file");
    serde_json::from_str(&json_content).expect("Failed to parse ProverInput from JSON")
}
