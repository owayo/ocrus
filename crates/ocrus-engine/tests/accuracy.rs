//! Character accuracy of the **production** pipeline.
//!
//! `crates/ocrus-cli/tests/char_accuracy.rs` measures a separate research path: it feeds
//! normalized crops straight to the model and merges greedy with TLA. That number says how
//! good the model is, not what a user of `ocrus recognize` actually gets — which goes
//! through quality gating, orientation detection, layout analysis and the production
//! decoder. The two have never been compared, and improvements have been judged on the
//! path users do not run.
//!
//! This test closes that gap: same images, same denominators as `char_accuracy_step2`, but
//! through `OcrEngine`.
//!
//! ```bash
//! cargo test -p ocrus-engine --release --test accuracy -- --ignored --nocapture
//! ```
//!
//! `OCRUS_ACC_CATEGORIES` (default `hiragana,katakana`) and `OCRUS_ACC_FONTS` (default: all)
//! narrow the run while iterating.

use std::path::{Path, PathBuf};

use ocrus_core::EngineConfigBuilder;
use ocrus_engine::{OcrEngine, models_ready};
use rayon::prelude::*;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate> should have a workspace root above it")
        .to_path_buf()
}

/// Characters of one category, in file order.
fn category_chars(root: &Path, category: &str) -> Vec<char> {
    let path = root.join("data/test_chars").join(format!("{category}.txt"));
    std::fs::read_to_string(path)
        .map(|s| s.chars().filter(|c| !c.is_whitespace()).collect())
        .unwrap_or_default()
}

/// Compare like `char_accuracy` does, so the two numbers mean the same thing.
fn matches(got: char, expected: char) -> bool {
    use unicode_normalization::UnicodeNormalization;
    let norm = |c: char| c.to_string().nfkc().next().unwrap_or(c);
    got == expected || norm(got) == norm(expected)
}

#[test]
#[ignore = "runs the whole production pipeline over ~850 images"]
fn production_pipeline_character_accuracy() {
    let config = EngineConfigBuilder::new().build();
    if !models_ready(&config.model_dir) {
        eprintln!("skip: no model in {}", config.model_dir.display());
        return;
    }
    let root = workspace_root();
    let images_dir = root.join("test_images");
    if !images_dir.is_dir() {
        eprintln!("skip: {} not found", images_dir.display());
        return;
    }

    let categories: Vec<String> = std::env::var("OCRUS_ACC_CATEGORIES")
        .unwrap_or_else(|_| "hiragana,katakana".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut fonts: Vec<String> = std::fs::read_dir(&images_dir)
        .expect("test_images is unreadable")
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    fonts.sort();
    if let Ok(wanted) = std::env::var("OCRUS_ACC_FONTS") {
        let wanted: Vec<&str> = wanted.split(',').map(str::trim).collect();
        fonts.retain(|f| wanted.contains(&f.as_str()));
    }

    let engine = OcrEngine::new(config).expect("the engine failed to load");
    println!("fonts: {}", fonts.join(", "));

    let started = std::time::Instant::now();
    let mut grand_correct = 0usize;
    let mut grand_total = 0usize;

    for category in &categories {
        let chars = category_chars(&root, category);
        if chars.is_empty() {
            println!("  {category:<20} (no data)");
            continue;
        }

        // One task per image: the pipeline itself is parallel inside, but at this size
        // running images concurrently keeps every core busy.
        let results: Vec<(bool, bool)> = fonts
            .iter()
            .flat_map(|font| chars.iter().map(move |ch| (font, *ch)))
            .par_bridge()
            .filter_map(|(font, ch)| {
                let path = images_dir
                    .join(font)
                    .join(category)
                    .join(format!("U+{:04X}.png", ch as u32));
                let result = engine.recognize_path(&path).ok()?;
                let got = result.full_text().chars().find(|c| !c.is_whitespace());
                Some((got.is_some_and(|g| matches(g, ch)), got.is_none()))
            })
            .collect();

        let total = results.len();
        let correct = results.iter().filter(|(ok, _)| *ok).count();
        let empty = results.iter().filter(|(_, e)| *e).count();
        let pct = if total > 0 {
            correct as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        println!("  {category:<20} {correct:>5}/{total:<5} ({pct:.1}%)  空出力 {empty}");
        grand_correct += correct;
        grand_total += total;
    }

    let pct = if grand_total > 0 {
        grand_correct as f64 / grand_total as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  {:<20} {grand_correct:>5}/{grand_total:<5} ({pct:.1}%)",
        "合計"
    );
    println!("  {:.1}s", started.elapsed().as_secs_f64());

    assert!(grand_total > 0, "no test images were read");
}
