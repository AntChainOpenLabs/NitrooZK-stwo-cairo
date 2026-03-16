use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Run `cargo metadata` to locate the cairo-program-runner-lib package.
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version=1"])
        .output()
        .expect("Failed to run `cargo metadata`");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("Failed to parse cargo metadata JSON");

    let packages = metadata["packages"]
        .as_array()
        .expect("packages field missing");

    let manifest_path = packages
        .iter()
        .find(|p| p["name"].as_str() == Some("cairo-program-runner-lib"))
        .expect("cairo-program-runner-lib not found in cargo metadata")["manifest_path"]
        .as_str()
        .expect("manifest_path is not a string");

    let pkg_dir = PathBuf::from(manifest_path)
        .parent()
        .expect("manifest_path has no parent")
        .to_path_buf();

    let bootloader_path = pkg_dir.join(
        "resources/compiled_programs/bootloaders/simple_bootloader_compiled.json",
    );

    println!(
        "cargo:rustc-env=BOOTLOADER_JSON_PATH={}",
        bootloader_path.display()
    );

    // Re-run if the dependency tree changes.
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
}
