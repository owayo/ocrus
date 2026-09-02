//! Checks that the installed `.ocnn` model reproduces the outputs its converter recorded.
//!
//! This is the guard against the failure this project actually hit: a model file built by
//! an older converter loaded without complaint and answered confidently wrong on every
//! character. Structural validation cannot catch that, so the converter records what the
//! reference runtime produced and this test replays it.
//!
//! Skips itself when no `.ocnn` model is installed, so a fresh clone and CI stay green.

use std::path::PathBuf;

use ocrus_nn::ocnn::exec::Executor;
use ocrus_nn::ocnn::{format::Model, verify};

fn model_path() -> PathBuf {
    if let Ok(dir) = std::env::var("OCRUS_MODEL_DIR") {
        return PathBuf::from(dir).join("rec.ocnn");
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".ocrus/models/rec.ocnn")
}

#[test]
fn installed_model_matches_its_golden_records() {
    let path = model_path();
    if !path.is_file() {
        eprintln!("skip: no .ocnn model at {}", path.display());
        return;
    }
    if cfg!(debug_assertions) {
        eprintln!("skip: debug build is too slow for a full inference; run with --release");
        return;
    }

    let model = Model::load(&path).expect("the model failed to load");
    println!(
        "{}: {} nodes, {} tensors, {} golden record(s), converted by {}",
        path.display(),
        model.meta.nodes.len(),
        model.meta.tensors.len(),
        model.meta.golden.len(),
        model.meta.converter,
    );

    model
        .verify_checksums()
        .expect("the tensor payload is corrupt");

    let report = verify(&Executor::new(model)).expect("verification could not run");
    assert!(
        report.checked > 0,
        "the model carries no golden records, so it cannot be verified: re-convert it"
    );
    for failure in &report.failures {
        eprintln!("  {failure}");
    }
    assert!(
        report.ok(),
        "{} of {} golden record(s) did not match: this model does not compute what the \
         converter measured. Re-convert from rec.onnx.",
        report.failures.len(),
        report.checked,
    );
    println!(
        "golden records: {}/{} matched",
        report.checked, report.checked
    );
}
