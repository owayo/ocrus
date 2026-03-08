use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ab_glyph::{FontRef, PxScale};
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use log::info;
use serde::Serialize;

#[derive(Serialize)]
struct CharFailure {
    character: char,
    category: String,
    font_name: String,
    expected: String,
    recognized: String,
}

use ocrus_layout::detect_lines_projection;
use ocrus_nn::{NnEngine, Tensor};
use ocrus_preproc::{binarize_adaptive, normalize_line, to_grayscale};
use ocrus_recognizer::{charset::Charset, ctc_greedy_decode};

const BATCH_SIZE: usize = 15;
const FONT_SIZE: f32 = 48.0;
const PADDING: u32 = 20;

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

struct FontEntry {
    name: String,
    data: Vec<u8>,
    index: u32,
}

fn discover_fonts() -> Vec<FontEntry> {
    let known_fonts: Vec<(&str, &str, u32)> = vec![
        (
            "HiraginoKakuGothic W3",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            0,
        ),
        (
            "HiraginoMincho ProN",
            "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
            0,
        ),
        ("BIZ-UDGothic", "/Library/Fonts/BIZ-UDGothicR.ttc", 0),
        ("BIZ-UDMincho", "/Library/Fonts/BIZ-UDMinchoM.ttc", 0),
        (
            "UDEVGothic",
            "/Users/owa/Library/Fonts/UDEVGothic-Regular.ttf",
            0,
        ),
    ];

    let mut fonts = Vec::new();

    for (name, path, index) in &known_fonts {
        let p = Path::new(path);
        if p.exists()
            && let Ok(data) = std::fs::read(p)
        {
            fonts.push(FontEntry {
                name: name.to_string(),
                data,
                index: *index,
            });
        }
    }

    let scan_dirs = [
        PathBuf::from("/System/Library/Fonts"),
        PathBuf::from("/Library/Fonts"),
        dirs_home().join("Library/Fonts"),
    ];

    let known_paths: Vec<PathBuf> = known_fonts
        .iter()
        .map(|(_, p, _)| PathBuf::from(p))
        .collect();

    for dir in &scan_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if known_paths.contains(&path) {
                    continue;
                }
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if matches!(ext.as_str(), "ttf" | "otf" | "ttc")
                    && let Ok(data) = std::fs::read(&path)
                {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    fonts.push(FontEntry {
                        name,
                        data,
                        index: 0,
                    });
                }
            }
        }
    }

    fonts
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
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

fn render_text_image(font: &FontRef<'_>, text: &str) -> DynamicImage {
    let scale = PxScale::from(FONT_SIZE);
    let (text_w, text_h) = text_size(scale, font, text);
    let img_w = text_w + PADDING * 2;
    let img_h = text_h + PADDING * 2;
    let mut img = RgbImage::from_pixel(img_w, img_h, Rgb([255u8, 255, 255]));
    draw_text_mut(
        &mut img,
        Rgb([0u8, 0, 0]),
        PADDING as i32,
        PADDING as i32,
        scale,
        font,
        text,
    );
    DynamicImage::ImageRgb8(img)
}

fn recognize_image(
    engine: &NnEngine,
    model: &ocrus_nn::model::OcnnModel,
    charset: &Charset,
    img: &DynamicImage,
) -> String {
    let gray = to_grayscale(img);
    let binary = binarize_adaptive(&gray);
    let lines = detect_lines_projection(&binary);

    let mut recognized = String::new();
    if lines.is_empty() {
        let bbox = ocrus_core::BBox::new(0, 0, gray.ncols() as u32, gray.nrows() as u32);
        let tensor = normalize_line(&gray, &bbox);
        let shape = tensor.shape().to_vec();
        let input = Tensor::new(tensor.into_raw_vec_and_offset().0, shape);
        if let Ok(outputs) = engine.run(model, &[input])
            && let Some(output) = outputs.first()
        {
            let timesteps = output.shape[1];
            let num_classes = output.shape[2];
            let (text, _conf) = ctc_greedy_decode(&output.data, timesteps, num_classes, charset);
            recognized.push_str(&text);
        }
    } else {
        for line in &lines {
            let tensor = normalize_line(&gray, line);
            let shape = tensor.shape().to_vec();
            let input = Tensor::new(tensor.into_raw_vec_and_offset().0, shape);
            if let Ok(outputs) = engine.run(model, &[input])
                && let Some(output) = outputs.first()
            {
                let timesteps = output.shape[1];
                let num_classes = output.shape[2];
                let (text, _conf) =
                    ctc_greedy_decode(&output.data, timesteps, num_classes, charset);
                recognized.push_str(&text);
            }
        }
    }

    recognized
}

fn char_accuracy(expected: &str, recognized: &str) -> (usize, usize) {
    let exp_chars: Vec<char> = expected.chars().collect();
    let rec_chars: Vec<char> = recognized.chars().filter(|c| !c.is_whitespace()).collect();
    let total = exp_chars.len();
    let mut correct = 0;
    let mut rec_idx = 0;
    for &exp_c in &exp_chars {
        if rec_idx < rec_chars.len() && rec_chars[rec_idx] == exp_c {
            correct += 1;
            rec_idx += 1;
        } else if rec_idx < rec_chars.len() {
            let search_start = rec_idx.saturating_sub(2);
            let search_end = (rec_idx + 3).min(rec_chars.len());
            if search_start < search_end
                && let Some(pos) = rec_chars[search_start..search_end]
                    .iter()
                    .position(|&c| c == exp_c)
            {
                correct += 1;
                rec_idx = search_start + pos + 1;
                continue;
            }
            rec_idx += 1;
        }
    }
    (correct, total)
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

    let engine = NnEngine::new().expect("Failed to create NnEngine");
    let model = engine
        .load_model(&model_path)
        .expect("Failed to load model");
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
        engine
            .load_model(qpath)
            .expect("Failed to load quantized model")
    });

    let fonts = discover_fonts();
    assert!(!fonts.is_empty(), "No fonts found on this system");

    let mut overall: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    let mut overall_q: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    let all_failures: Arc<Mutex<Vec<CharFailure>>> = Arc::new(Mutex::new(Vec::new()));
    let mut total_fp32_elapsed = std::time::Duration::ZERO;
    let mut total_int8_elapsed = std::time::Duration::ZERO;
    let mut inference_count = 0u64;

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

    for font_entry in &fonts {
        let font = match FontRef::try_from_slice_and_index(&font_entry.data, font_entry.index) {
            Ok(f) => f,
            Err(_) => {
                log::warn!("Skipping font {} (failed to load)", font_entry.name);
                continue;
            }
        };

        info!("=== Font: {} ===", font_entry.name);

        for &category in categories {
            let chars = match load_category_chars(&workspace_root, category) {
                Some(c) => c,
                None => {
                    info!("  {category}: (no data)");
                    continue;
                }
            };

            let mut cat_correct = 0usize;
            let mut cat_total = 0usize;
            let mut cat_correct_q = 0usize;
            let mut cat_total_q = 0usize;

            let total_chars = chars.len();
            let mut processed_chars = 0usize;
            let cat_start = std::time::Instant::now();

            for batch in chars.chunks(BATCH_SIZE) {
                let batch_str: String = batch.iter().collect();
                let img = render_text_image(&font, &batch_str);

                // FP32 inference
                let t0 = std::time::Instant::now();
                let recognized = recognize_image(&engine, &model, &charset, &img);
                total_fp32_elapsed += t0.elapsed();
                inference_count += 1;

                let (correct, total) = char_accuracy(&batch_str, &recognized);
                cat_correct += correct;
                cat_total += total;

                // Progress log every 100 chars
                processed_chars += batch.len();
                if processed_chars % 100 < BATCH_SIZE || processed_chars == total_chars {
                    let elapsed_secs = cat_start.elapsed().as_secs_f64();
                    let pct_done = processed_chars as f64 / total_chars as f64 * 100.0;
                    let spc = if processed_chars > 0 {
                        elapsed_secs / processed_chars as f64
                    } else {
                        0.0
                    };
                    let eta = if spc > 0.0 {
                        let remaining = spc * (total_chars - processed_chars) as f64;
                        format!("ETA {:.0}s", remaining)
                    } else {
                        "ETA --".to_string()
                    };
                    info!(
                        "    {category}: {processed_chars}/{total_chars} ({pct_done:.0}%) {:.0}s elapsed, {spc:.2}s/char, {eta}",
                        elapsed_secs
                    );
                }

                let rec_chars: Vec<char> =
                    recognized.chars().filter(|c| !c.is_whitespace()).collect();
                for (i, &exp_c) in batch.iter().enumerate() {
                    let rec_c = rec_chars.get(i).copied();
                    if rec_c != Some(exp_c) {
                        all_failures.lock().unwrap().push(CharFailure {
                            character: exp_c,
                            category: category.to_string(),
                            font_name: font_entry.name.clone(),
                            expected: exp_c.to_string(),
                            recognized: rec_c.map(|c| c.to_string()).unwrap_or_default(),
                        });
                    }
                }

                // INT8 inference (A/B comparison)
                if let Some(ref q_model) = quantized_model {
                    let t0 = std::time::Instant::now();
                    let recognized_q = recognize_image(&engine, q_model, &charset, &img);
                    total_int8_elapsed += t0.elapsed();

                    let (correct_q, total_q) = char_accuracy(&batch_str, &recognized_q);
                    cat_correct_q += correct_q;
                    cat_total_q += total_q;
                }
            }

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

    // Print timing comparison when A/B test is active
    if quantized_model.is_some() && inference_count > 0 {
        let fp32_ms = total_fp32_elapsed.as_secs_f64() * 1000.0;
        let int8_ms = total_int8_elapsed.as_secs_f64() * 1000.0;
        let fp32_avg = fp32_ms / inference_count as f64;
        let int8_avg = int8_ms / inference_count as f64;
        let speedup = if int8_ms > 0.0 {
            fp32_ms / int8_ms
        } else {
            f64::NAN
        };
        info!("=== Inference Timing ({inference_count} batches) ===");
        info!("  FP32 total: {fp32_ms:.1}ms  avg: {fp32_avg:.2}ms/batch");
        info!("  INT8 total: {int8_ms:.1}ms  avg: {int8_avg:.2}ms/batch");
        info!("  Speedup:    {speedup:.2}x");
    }

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
