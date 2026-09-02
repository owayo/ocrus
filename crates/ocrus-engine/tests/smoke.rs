//! Short end-to-end checks for the recognition pipeline.
//!
//! These exist to answer one question quickly: *does OCR still work at all?* A dependency
//! bump, a preprocessing tweak or a model swap can silently turn recognition into an empty
//! page, and the full `char_accuracy` suite takes half an hour — far too long to run after
//! every change. This runs in seconds.
//!
//! It is deliberately not an accuracy benchmark. It prints the numbers it measures so a
//! human can compare runs, but it only fails on catastrophic breakage (nothing recognized
//! at all). Set `OCRUS_SMOKE_MIN_ACCURACY` to enforce a floor.
//!
//! ```bash
//! cargo test -p ocrus-engine --release --test smoke -- --nocapture
//! ```
//!
//! Both tests skip themselves when the model or `test_images/` is absent, so a fresh clone
//! and CI stay green without the 80 MB model.

use std::path::{Path, PathBuf};

use ocrus_core::EngineConfigBuilder;
use ocrus_engine::{OcrEngine, models_ready};

/// Categories the smoke test samples, and how many characters from each.
const SAMPLE_CATEGORIES: &[&str] = &["hiragana", "katakana", "halfwidth_alnum"];
const SAMPLES_PER_CATEGORY: usize = 8;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate> should have a workspace root above it")
        .to_path_buf()
}

/// Build an engine, or explain why the test is being skipped.
fn engine_or_skip() -> Option<OcrEngine> {
    // Inference is pure Rust with no optimizations in a debug build: a single image takes
    // minutes instead of a fraction of a second. Running there would turn `cargo test`
    // into an hours-long command, so the check only runs in release.
    if cfg!(debug_assertions) {
        eprintln!(
            "skip: debug build. Run with --release (`cargo test -p ocrus-engine --release --test smoke -- --nocapture`)"
        );
        return None;
    }

    let config = EngineConfigBuilder::new().build();
    if !models_ready(&config.model_dir) {
        eprintln!(
            "skip: no model in {} (run `python models/download.py` and convert to .ocnn)",
            config.model_dir.display()
        );
        return None;
    }
    Some(OcrEngine::new(config).expect("model files exist but the engine failed to load"))
}

#[test]
fn recognizes_the_sample_page() {
    let Some(engine) = engine_or_skip() else {
        return;
    };

    let sample = workspace_root().join("testdata/sample_ja.png");
    if !sample.is_file() {
        eprintln!("skip: {} not found", sample.display());
        return;
    }

    let result = engine
        .recognize_path(&sample)
        .expect("recognizing the sample page failed");

    let page = result.pages.first().expect("no page in result");
    assert_eq!((page.width, page.height), (600, 200), "page size changed");
    assert!(!page.lines.is_empty(), "layout analysis found no lines");

    let text = result.full_text();
    println!("sample_ja.png -> {} line(s): {text:?}", page.lines.len());
    assert!(
        text.chars().any(|c| !c.is_whitespace()),
        "every line came back empty: recognition is broken, not merely inaccurate"
    );
}

#[test]
fn recognizes_single_characters() {
    let Some(engine) = engine_or_skip() else {
        return;
    };

    let root = workspace_root();
    let images_dir = root.join("test_images");
    let Some(font) = first_font(&images_dir) else {
        eprintln!(
            "skip: no fonts in {} (run `cargo test -p ocrus-cli --test generate_test_images \
             -- --ignored --nocapture`)",
            images_dir.display()
        );
        return;
    };

    println!("font: {font}");
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut non_empty = 0usize;

    for category in SAMPLE_CATEGORIES {
        let Some(chars) = category_chars(&root, category) else {
            continue;
        };

        // Evenly spaced samples rather than the first N: the head of each file is
        // alphabetically or codepoint-ordered, so the first N are unrepresentative.
        let step = (chars.len() / SAMPLES_PER_CATEGORY).max(1);
        let mut cat_total = 0usize;
        let mut cat_correct = 0usize;

        for ch in chars.iter().step_by(step).take(SAMPLES_PER_CATEGORY) {
            let path = images_dir
                .join(&font)
                .join(category)
                .join(format!("U+{:04X}.png", *ch as u32));
            let Ok(result) = engine.recognize_path(&path) else {
                continue;
            };
            cat_total += 1;
            let got = result.full_text().chars().find(|c| !c.is_whitespace());
            if got.is_some() {
                non_empty += 1;
            }
            if got.is_some_and(|got| got == *ch) {
                cat_correct += 1;
            }
        }

        if cat_total > 0 {
            let pct = cat_correct as f64 / cat_total as f64 * 100.0;
            println!("  {category:<18} {cat_correct:>2}/{cat_total:<2} ({pct:.0}%)");
        }
        total += cat_total;
        correct += cat_correct;
    }

    if total == 0 {
        eprintln!("skip: no test images matched the sampled characters");
        return;
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    println!("  {:<18} {correct:>2}/{total:<2} ({accuracy:.0}%)", "total");
    println!("  non-empty output : {non_empty}/{total}");
    println!(
        "  24 images are far too few to judge accuracy — `--test accuracy` measures the same \
         pipeline over 845. This number only catches a collapse. Baseline 2026-09-03: 12/24 \
         correct, 24/24 non-empty. A drop to 0 usually means the model file and the executor \
         disagree, which `--test ocnn_golden` reports directly."
    );

    // Only total breakage fails the run. Asserting an accuracy number here would either be
    // flaky or freeze today's (known bad) production accuracy in place; the number is
    // printed so a human can compare it across runs instead.
    assert!(
        non_empty > 0,
        "every single-character image came back empty: recognition is broken"
    );

    if let Ok(floor) = std::env::var("OCRUS_SMOKE_MIN_ACCURACY") {
        let floor: f64 = floor
            .parse()
            .expect("OCRUS_SMOKE_MIN_ACCURACY must be a number");
        assert!(
            accuracy >= floor,
            "accuracy {accuracy:.1}% is below the required {floor:.1}%"
        );
    }
}

/// The alphabetically first font directory, so runs are comparable with each other.
fn first_font(images_dir: &Path) -> Option<String> {
    let mut fonts: Vec<String> = std::fs::read_dir(images_dir)
        .ok()?
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    fonts.sort();
    fonts.into_iter().next()
}

fn category_chars(root: &Path, category: &str) -> Option<Vec<char>> {
    let path = root.join("data/test_chars").join(format!("{category}.txt"));
    let content = std::fs::read_to_string(path).ok()?;
    let chars: Vec<char> = content.chars().filter(|c| !c.is_whitespace()).collect();
    (!chars.is_empty()).then_some(chars)
}
