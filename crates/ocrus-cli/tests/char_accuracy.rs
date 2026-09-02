use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use std::sync::atomic::AtomicUsize;

use image::{DynamicImage, GrayImage, Luma};
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};
use log::info;
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

#[derive(Serialize)]
struct CharFailure {
    character: char,
    category: String,
    font_name: String,
    expected: String,
    recognized: String,
}

use ocrus_layout::detect_lines_projection;
use ocrus_nn::{Executor, Model, NdTensor};
use ocrus_preproc::{binarize_adaptive, normalize_line_scaled, to_grayscale};
use ocrus_recognizer::{charset::Charset, ctc_greedy_decode, ctc_tla_decode};

// Step definitions: name -> categories
const STEPS: &[(&str, &[&str])] = &[
    (
        "step1",
        &[
            "halfwidth_alnum",
            "halfwidth_symbols",
            "fullwidth_alnum",
            "fullwidth_symbols",
        ],
    ),
    ("step2", &["hiragana", "katakana"]),
    ("step3_joyo", &["joyo_kanji"]),
    ("step3_jis1", &["jis_level1"]),
    ("step3_jis2", &["jis_level2"]),
    ("step3_jis3", &["jis_level3"]),
    ("step3_jis4", &["jis_level4"]),
];

/// Load font names from test_images/ directory.
/// Each subdirectory under test_images/ represents a font.
fn load_font_names(workspace_root: &Path) -> Vec<String> {
    let images_dir = workspace_root.join("test_images");
    assert!(
        images_dir.is_dir(),
        "test_images/ not found. Run: cargo test -p ocrus-cli --test generate_test_images -- --ignored --nocapture"
    );

    let mut names: Vec<String> = std::fs::read_dir(&images_dir)
        .expect("Failed to read test_images/")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_dir() {
                Some(entry.file_name().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    names.sort();
    names
}

/// Load a pre-rendered test image for a specific font, category, and character.
fn load_test_image(
    workspace_root: &Path,
    font_name: &str,
    category: &str,
    ch: char,
) -> Option<DynamicImage> {
    let codepoint = ch as u32;
    let filename = format!("U+{codepoint:04X}.png");
    let path = workspace_root
        .join("test_images")
        .join(font_name)
        .join(category)
        .join(&filename);
    image::open(&path).ok()
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/owa"))
}

fn load_category_chars(workspace_root: &Path, category: &str) -> Option<Vec<char>> {
    let path = workspace_root
        .join("data/test_chars")
        .join(format!("{category}.txt"));
    let content = std::fs::read_to_string(&path).ok()?;
    let chars: Vec<char> = content.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() { None } else { Some(chars) }
}

fn straug_tta_enabled() -> bool {
    matches!(
        std::env::var("OCRUS_CHAR_STR_AUG_TTA").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn autocontrast_gray(gray: &Array2<u8>) -> Array2<u8> {
    let mut min_px = u8::MAX;
    let mut max_px = u8::MIN;
    for &px in gray {
        min_px = min_px.min(px);
        max_px = max_px.max(px);
    }
    if min_px >= max_px {
        return gray.clone();
    }

    let scale = 255.0f32 / (max_px as f32 - min_px as f32);
    gray.mapv(|px| {
        (((px.saturating_sub(min_px)) as f32) * scale)
            .round()
            .clamp(0.0, 255.0) as u8
    })
}

fn sharpen_gray(gray: &Array2<u8>) -> Array2<u8> {
    let (h, w) = (gray.nrows(), gray.ncols());
    if h < 3 || w < 3 {
        return gray.clone();
    }

    let mut out = gray.clone();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let val = 5i32 * gray[[y, x]] as i32
                - gray[[y - 1, x]] as i32
                - gray[[y + 1, x]] as i32
                - gray[[y, x - 1]] as i32
                - gray[[y, x + 1]] as i32;
            out[[y, x]] = val.clamp(0, 255) as u8;
        }
    }
    out
}

fn ndarray_to_gray_image(gray: &Array2<u8>) -> GrayImage {
    let (h, w) = (gray.nrows() as u32, gray.ncols() as u32);
    GrayImage::from_fn(w, h, |x, y| Luma([gray[[y as usize, x as usize]]]))
}

fn gray_image_to_ndarray(img: &GrayImage) -> Array2<u8> {
    let (w, h) = img.dimensions();
    Array2::from_shape_fn((h as usize, w as usize), |(y, x)| {
        img.get_pixel(x as u32, y as u32)[0]
    })
}

fn rotate_gray(gray: &Array2<u8>, degrees: f32) -> Array2<u8> {
    let img = ndarray_to_gray_image(gray);
    let rotated = rotate_about_center(
        &img,
        degrees.to_radians(),
        Interpolation::Bilinear,
        Border::Constant(Luma([255u8])),
    );
    gray_image_to_ndarray(&rotated)
}

fn straug_tta_variants(gray: &Array2<u8>) -> Vec<Array2<u8>> {
    vec![
        gray.clone(),
        autocontrast_gray(gray),
        sharpen_gray(gray),
        rotate_gray(gray, -2.0),
        rotate_gray(gray, 2.0),
    ]
}

fn vote_texts(texts: &[String]) -> String {
    use std::collections::HashMap;

    let mut counts = HashMap::<String, usize>::new();
    for text in texts {
        *counts.entry(text.clone()).or_default() += 1;
    }

    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.len().cmp(&a.0.len())))
        .map(|(text, _)| text)
        .unwrap_or_default()
}

fn recognize_bbox_base(
    exec: &Executor,
    charset: &Charset,
    gray: &Array2<u8>,
    bbox: &ocrus_core::BBox,
) -> String {
    let tensor = normalize_line_scaled(gray, bbox, 1.0);
    let shape = tensor.shape().to_vec();
    let input = NdTensor::from_vec(tensor.into_raw_vec_and_offset().0, &shape);
    if let Ok(output) = exec.run(input) {
        let timesteps = output.shape[1];
        let num_classes = output.shape[2];

        let (ctc_text, _ctc_conf) =
            ctc_greedy_decode(&output.data, timesteps, num_classes, charset);
        let (tla_text, _tla_conf) = ctc_tla_decode(&output.data, timesteps, num_classes, charset);

        if ctc_text == tla_text {
            return ctc_text;
        }
        if ctc_text.is_empty() && !tla_text.is_empty() {
            return tla_text;
        }
        if ctc_text.chars().count() > 1 && tla_text.chars().count() == 1 {
            return tla_text;
        }
        return ctc_text;
    }
    String::new()
}

/// Recognize a single bbox using both CTC greedy and TLA, picking the better result.
fn recognize_bbox_ensemble(
    exec: &Executor,
    charset: &Charset,
    gray: &Array2<u8>,
    bbox: &ocrus_core::BBox,
) -> String {
    if !straug_tta_enabled() {
        return recognize_bbox_base(exec, charset, gray, bbox);
    }

    let variants = straug_tta_variants(gray);
    let mut texts = Vec::with_capacity(variants.len());
    for variant in &variants {
        texts.push(recognize_bbox_base(exec, charset, variant, bbox));
    }

    let voted = vote_texts(&texts);
    if !voted.is_empty() {
        return voted;
    }

    texts.into_iter().next().unwrap_or_default()
}

fn recognize_image(exec: &Executor, charset: &Charset, img: &DynamicImage) -> String {
    let gray = to_grayscale(img);
    let binary = binarize_adaptive(&gray);
    let lines = detect_lines_projection(&binary);

    let mut recognized = String::new();
    if lines.is_empty() {
        let bbox = ocrus_core::BBox::new(0, 0, gray.ncols() as u32, gray.nrows() as u32);
        recognized = recognize_bbox_ensemble(exec, charset, &gray, &bbox);
    } else {
        for line in &lines {
            let text = recognize_bbox_ensemble(exec, charset, &gray, line);
            recognized.push_str(&text);
        }
    }

    recognized
}

/// Normalize a char via NFKC (e.g. fullwidth 'Ａ' -> halfwidth 'A').
fn normalize_char(c: char) -> char {
    let s: String = c.to_string().nfkc().collect();
    s.chars().next().unwrap_or(c)
}

/// Compare chars with NFKC normalization (fullwidth/halfwidth differences are tolerated).
fn chars_match(a: char, b: char) -> bool {
    a == b || normalize_char(a) == normalize_char(b)
}

/// Resolve which categories to test based on step name.
/// Returns None if step name is invalid.
fn resolve_categories(step: &str) -> Option<Vec<&'static str>> {
    // "all" runs everything
    if step == "all" {
        return Some(
            STEPS
                .iter()
                .flat_map(|(_, cats)| cats.iter().copied())
                .collect(),
        );
    }
    // "step3" runs all kanji steps
    if step == "step3" {
        return Some(
            STEPS
                .iter()
                .filter(|(name, _)| name.starts_with("step3"))
                .flat_map(|(_, cats)| cats.iter().copied())
                .collect(),
        );
    }
    // Exact match (step1, step2, step3_joyo, step3_jis1, ...)
    for (name, cats) in STEPS {
        if *name == step {
            return Some(cats.to_vec());
        }
    }
    None
}

/// Core test runner shared by all step-specific tests.
fn run_accuracy_test(categories: &[&str], step_label: &str) {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let model_dir = std::env::var("OCRUS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join(".ocrus/models"));

    let model_path = model_dir.join("rec.ocnn");
    let dict_path = model_dir.join("dict.txt");

    assert!(
        model_path.exists(),
        "Model not found at {}. Run models/download.sh or convert to .ocnn first.",
        model_path.display()
    );
    assert!(
        dict_path.exists(),
        "Dict not found at {}",
        dict_path.display()
    );

    // Setup log file in logs/ directory
    let logs_dir = workspace_root.join("logs");
    std::fs::create_dir_all(&logs_dir).ok();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let log_path = logs_dir.join(format!("char_accuracy_{step_label}_{now}.log"));
    let log_file = std::fs::File::create(&log_path).expect("Failed to create log file");

    use simplelog::{
        ColorChoice, CombinedLogger, ConfigBuilder, LevelFilter, TermLogger, TerminalMode,
        WriteLogger,
    };
    let log_config = ConfigBuilder::new()
        .set_time_offset_to_local()
        .unwrap_or_else(|b| b)
        .build();
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Info,
            log_config.clone(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(LevelFilter::Info, log_config, log_file),
    ])
    .expect("Failed to init logger");

    info!("Step: {step_label}");
    info!("Categories: {}", categories.join(", "));
    info!("Log file: {}", log_path.display());

    let exec = Executor::new(Model::load(&model_path).expect("Failed to load model"));
    let charset = Charset::from_file(&dict_path).expect("Failed to load charset");

    // Optionally load quantized model for A/B comparison
    let quantized_model_path = std::env::var("OCRUS_QUANTIZED_MODEL")
        .map(PathBuf::from)
        .ok();
    let quantized_model = quantized_model_path.as_ref().map(|qpath| {
        assert!(
            qpath.exists(),
            "Quantized model not found at {}",
            qpath.display()
        );
        info!("A/B test enabled: FP32 vs INT8 ({})", qpath.display());
        Executor::new(Model::load(qpath).expect("Failed to load quantized model"))
    });

    let font_names = load_font_names(&workspace_root);
    assert!(
        !font_names.is_empty(),
        "No fonts found in test_images/ directory"
    );

    let mut overall: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    let mut overall_q: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    let all_failures: Arc<Mutex<Vec<CharFailure>>> = Arc::new(Mutex::new(Vec::new()));
    let test_start = std::time::Instant::now();

    // Setup results directory and Ctrl+C handler
    let results_dir = workspace_root.join("test_results");
    std::fs::create_dir_all(&results_dir).ok();
    let failures_path = results_dir.join(format!("failures_{step_label}.json"));

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        let all_failures = Arc::clone(&all_failures);
        let failures_path = failures_path.clone();
        ctrlc::set_handler(move || {
            interrupted.store(true, Ordering::SeqCst);
            let failures = all_failures.lock().unwrap();
            if !failures.is_empty() {
                let json = serde_json::to_string_pretty(&*failures).unwrap();
                std::fs::write(&failures_path, &json).ok();
                eprintln!(
                    "\nInterrupted. Saved {} failures to {}",
                    failures.len(),
                    failures_path.display()
                );
            }
            std::process::exit(1);
        })
        .expect("Failed to set Ctrl+C handler");
    }

    for font_name in &font_names {
        info!("=== Font: {} ===", font_name);

        for &category in categories {
            let chars = match load_category_chars(&workspace_root, category) {
                Some(c) => c,
                None => {
                    info!("  {category}: (no data)");
                    continue;
                }
            };

            let total_chars = chars.len();
            let cat_correct = AtomicUsize::new(0);
            let cat_total = AtomicUsize::new(0);
            let cat_correct_q = AtomicUsize::new(0);
            let cat_total_q = AtomicUsize::new(0);
            let processed_chars = AtomicUsize::new(0);
            let cat_start = std::time::Instant::now();

            chars.par_iter().for_each(|&exp_c| {
                let img = match load_test_image(&workspace_root, font_name, category, exp_c) {
                    Some(img) => img,
                    None => return,
                };

                // FP32 inference
                let recognized = recognize_image(&exec, &charset, &img);

                let rec_c = recognized.chars().find(|c| !c.is_whitespace());
                let is_correct = rec_c.is_some_and(|r| chars_match(r, exp_c));
                if is_correct {
                    cat_correct.fetch_add(1, Ordering::Relaxed);
                } else {
                    all_failures.lock().unwrap().push(CharFailure {
                        character: exp_c,
                        category: category.to_string(),
                        font_name: font_name.clone(),
                        expected: exp_c.to_string(),
                        recognized: rec_c.map(|c| c.to_string()).unwrap_or_default(),
                    });
                }
                cat_total.fetch_add(1, Ordering::Relaxed);

                // Progress log every 100 chars
                let done = processed_chars.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(100) || done == total_chars {
                    let elapsed_secs = cat_start.elapsed().as_secs_f64();
                    let pct_done = done as f64 / total_chars as f64 * 100.0;
                    let spc = elapsed_secs / done as f64;
                    let remaining = spc * (total_chars - done) as f64;
                    info!(
                        "    {category}: {done}/{total_chars} ({pct_done:.0}%) {elapsed_secs:.0}s elapsed, {spc:.2}s/char, ETA {remaining:.0}s",
                    );
                }

                // INT8 inference (A/B comparison)
                if let Some(q_exec) = quantized_model.as_ref() {
                    let recognized_q = recognize_image(q_exec, &charset, &img);

                    let rec_c_q = recognized_q.chars().find(|c| !c.is_whitespace());
                    let is_correct_q = rec_c_q.is_some_and(|r| chars_match(r, exp_c));
                    if is_correct_q {
                        cat_correct_q.fetch_add(1, Ordering::Relaxed);
                    }
                    cat_total_q.fetch_add(1, Ordering::Relaxed);
                }
            });

            let cat_correct = cat_correct.load(Ordering::Relaxed);
            let cat_total = cat_total.load(Ordering::Relaxed);
            let cat_correct_q = cat_correct_q.load(Ordering::Relaxed);
            let cat_total_q = cat_total_q.load(Ordering::Relaxed);

            let pct = if cat_total > 0 {
                cat_correct as f64 / cat_total as f64 * 100.0
            } else {
                0.0
            };

            if quantized_model.is_some() {
                let pct_q = if cat_total_q > 0 {
                    cat_correct_q as f64 / cat_total_q as f64 * 100.0
                } else {
                    0.0
                };
                let diff = pct_q - pct;
                info!(
                    "  {category:20} FP32={cat_correct:>5}/{cat_total:<5} ({pct:.1}%)  \
                     INT8={cat_correct_q:>5}/{cat_total_q:<5} ({pct_q:.1}%)  \
                     diff={diff:+.1}pp"
                );

                let entry_q = overall_q.entry(category.to_string()).or_insert((0, 0));
                entry_q.0 += cat_correct_q;
                entry_q.1 += cat_total_q;
            } else {
                info!("  {category:20} {cat_correct:>5}/{cat_total:<5} ({pct:.1}%)");
            }

            let entry = overall.entry(category.to_string()).or_insert((0, 0));
            entry.0 += cat_correct;
            entry.1 += cat_total;

            // Save failures incrementally after each category
            {
                let failures = all_failures.lock().unwrap();
                let json = serde_json::to_string_pretty(&*failures).unwrap();
                std::fs::write(&failures_path, &json).ok();
                info!(
                    "  -> Saved {} failures to {}",
                    failures.len(),
                    failures_path.display()
                );
            }
        }
    }

    let failures = all_failures.lock().unwrap();
    let failures_json = serde_json::to_string_pretty(&*failures).unwrap();
    std::fs::write(&failures_path, &failures_json).unwrap();
    info!(
        "Exported {} failures to {}",
        failures.len(),
        failures_path.display()
    );

    info!("=== Overall Accuracy ({step_label}) ===");
    for &category in categories {
        if let Some(&(correct, total)) = overall.get(category) {
            let pct = if total > 0 {
                correct as f64 / total as f64 * 100.0
            } else {
                0.0
            };

            if let Some(&(correct_q, total_q)) = overall_q.get(category) {
                let pct_q = if total_q > 0 {
                    correct_q as f64 / total_q as f64 * 100.0
                } else {
                    0.0
                };
                let diff = pct_q - pct;
                info!(
                    "  {category:20} FP32={correct:>5}/{total:<5} ({pct:.1}%)  \
                     INT8={correct_q:>5}/{total_q:<5} ({pct_q:.1}%)  \
                     diff={diff:+.1}pp"
                );
            } else {
                info!("  {category:20} {correct:>5}/{total:<5} ({pct:.1}%)");
            }
        }
    }

    let total_elapsed = test_start.elapsed();
    info!("=== Total time: {:.1}s ===", total_elapsed.as_secs_f64());
    info!("Log saved to: {}", log_path.display());
}

// --- Step-specific test functions ---

/// Step 1: Half-width & full-width alphanumeric + symbols
#[test]
#[ignore]
fn char_accuracy_step1() {
    let cats = resolve_categories("step1").unwrap();
    run_accuracy_test(&cats, "step1");
}

/// Step 2: Hiragana & Katakana
#[test]
#[ignore]
fn char_accuracy_step2() {
    let cats = resolve_categories("step2").unwrap();
    run_accuracy_test(&cats, "step2");
}

/// Step 3 (Joyo): Joyo kanji (2,136 chars)
#[test]
#[ignore]
fn char_accuracy_step3_joyo() {
    let cats = resolve_categories("step3_joyo").unwrap();
    run_accuracy_test(&cats, "step3_joyo");
}

/// Step 3 (JIS1): JIS X 0208 Level 1 kanji (2,965 chars)
#[test]
#[ignore]
fn char_accuracy_step3_jis1() {
    let cats = resolve_categories("step3_jis1").unwrap();
    run_accuracy_test(&cats, "step3_jis1");
}

/// Step 3 (JIS2): JIS X 0208 Level 2 kanji (3,390 chars)
#[test]
#[ignore]
fn char_accuracy_step3_jis2() {
    let cats = resolve_categories("step3_jis2").unwrap();
    run_accuracy_test(&cats, "step3_jis2");
}

/// Step 3 (JIS3): JIS X 0213 Level 3 kanji (1,233 chars)
#[test]
#[ignore]
fn char_accuracy_step3_jis3() {
    let cats = resolve_categories("step3_jis3").unwrap();
    run_accuracy_test(&cats, "step3_jis3");
}

/// Step 3 (JIS4): JIS X 0213 Level 4 kanji (7,960 chars)
#[test]
#[ignore]
fn char_accuracy_step3_jis4() {
    let cats = resolve_categories("step3_jis4").unwrap();
    run_accuracy_test(&cats, "step3_jis4");
}

/// All categories (full test)
#[test]
#[ignore]
fn char_accuracy_all() {
    let cats = resolve_categories("all").unwrap();
    run_accuracy_test(&cats, "all");
}
