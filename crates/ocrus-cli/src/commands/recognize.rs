use std::fs::File;

use anyhow::{Context, Result};
use memmap2::Mmap;
use ndarray::s;
use rayon::prelude::*;

use ocrus_core::{
    CharsetMode, EngineConfig, EngineConfigBuilder, OcrMode, OcrResult, Page, RubyAnnotation,
    TextLine,
};
use ocrus_layout::{
    TextOrientation, assess_quality, detect_columns_vertical, detect_lines_ccl,
    detect_lines_projection, detect_orientation, separate_ruby, should_use_fast_path,
};
use ocrus_preproc::{binarize_adaptive, normalize_line, normalize_line_vertical, to_grayscale};
use ocrus_recognizer::charset::Charset;
use ocrus_recognizer::{
    CascadeRecognizer, DictCorrector, GlyphCache, ctc_beam_decode, ctc_greedy_decode,
    ctc_greedy_decode_masked,
};
use ocrus_runtime::{InferenceBackend, ModelOptions, OrtBackend, Tensor};

use super::{CliCharset, CliMode, OutputFormat, RecognizeArgs};
use crate::output;

pub fn run(args: RecognizeArgs) -> Result<()> {
    let engine_config = build_engine_config(&args);

    let file = File::open(&args.input)
        .with_context(|| format!("Failed to open image: {}", args.input.display()))?;
    // SAFETY: Standard mmap pattern; file is not modified by other processes during OCR.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("Failed to mmap image: {}", args.input.display()))?;
    let img = image::load_from_memory(&mmap)
        .with_context(|| format!("Failed to decode image: {}", args.input.display()))?;

    let (width, height) = (img.width(), img.height());

    // Preprocessing
    let gray = to_grayscale(&img);
    let binary = binarize_adaptive(&gray);

    // Quality Gate
    let quality = assess_quality(&binary);
    let use_fast_path = match engine_config.mode {
        OcrMode::Fastest => true,
        OcrMode::Accurate => false,
        OcrMode::Auto => should_use_fast_path(&quality),
    };

    // Orientation detection and layout analysis
    let orientation = detect_orientation(&binary);
    let line_bboxes = match orientation {
        TextOrientation::Vertical => detect_columns_vertical(&binary),
        TextOrientation::Horizontal if use_fast_path => detect_lines_projection(&binary),
        _ => {
            // Accurate mode or Mixed: try CCL first, fall back to projection
            let ccl_lines = detect_lines_ccl(&binary);
            if ccl_lines.is_empty() {
                detect_lines_projection(&binary)
            } else {
                ccl_lines
            }
        }
    };

    // Ruby separation: detect and separate ruby annotations from body text
    let (line_bboxes, ruby_info) = if engine_config.ruby_separation {
        let mut new_bboxes = Vec::new();
        let mut ruby_map: Vec<Vec<ocrus_core::BBox>> = Vec::new();
        for bbox in &line_bboxes {
            let sep = separate_ruby(&binary, bbox, orientation);
            new_bboxes.push(sep.body_bbox);
            ruby_map.push(sep.ruby_bboxes);
        }
        (new_bboxes, Some(ruby_map))
    } else {
        (line_bboxes, None)
    };

    if line_bboxes.is_empty() {
        let result = OcrResult {
            pages: vec![Page {
                width,
                height,
                lines: vec![],
            }],
        };
        return output_result(&result, &args.format);
    }

    // Load model and charset
    let rec_model_path = engine_config.model_dir.join("rec.onnx");
    let charset_path = engine_config.model_dir.join("dict.txt");

    if !rec_model_path.exists() {
        anyhow::bail!(
            "Recognition model not found: {}\nRun `models/download.sh` to download models.",
            rec_model_path.display()
        );
    }

    let charset = Charset::from_file(&charset_path)
        .with_context(|| format!("Failed to load charset: {}", charset_path.display()))?;

    // Build logit mask for JIS charset if requested
    let logit_mask = match engine_config.charset {
        CharsetMode::Jis => {
            let data_dir =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/test_chars");
            let jis = Charset::from_jis_strict(&data_dir);
            Some(jis.logit_mask(&charset))
        }
        CharsetMode::Full => None,
    };

    // Load dictionary corrector if specified
    let dict_corrector = match &engine_config.dict_path {
        Some(path) => Some(
            DictCorrector::from_file(path)
                .with_context(|| format!("Failed to load dictionary: {}", path.display()))?,
        ),
        None => None,
    };

    let backend = OrtBackend::new().context("Failed to initialize ONNX Runtime")?;
    let model_opts = ModelOptions {
        num_threads: engine_config.num_threads,
        ..Default::default()
    };
    let model = backend
        .load_model(&rec_model_path, &model_opts)
        .context("Failed to load recognition model")?;

    // Cascade recognition (if cascade model specified)
    let _cascade = engine_config.cascade_model_path.as_ref().map(|_path| {
        // TODO: Load cascade classifier model when trained
        // For now, create cascade recognizer that always falls back to CTC
        CascadeRecognizer::new(engine_config.cascade_threshold)
    });

    // Normalize all lines (rotate vertical columns 90° for the horizontal-input model)
    let is_vertical = orientation == TextOrientation::Vertical;
    let line_tensors: Vec<Tensor> = line_bboxes
        .par_iter()
        .map(|bbox| {
            let line_img = if is_vertical {
                normalize_line_vertical(&gray, bbox)
            } else {
                normalize_line(&gray, bbox)
            };
            let shape = line_img.shape().to_vec();
            let data = line_img.into_raw_vec_and_offset().0;
            Tensor::new(data, shape)
        })
        .collect();

    // Check glyph cache for each line before inference
    let mut glyph_cache = GlyphCache::new(2);
    let mut cached_results: Vec<Option<(String, f32)>> = Vec::with_capacity(line_bboxes.len());
    let mut uncached_indices: Vec<usize> = Vec::new();
    let mut uncached_tensors: Vec<Tensor> = Vec::new();

    for (i, (bbox, tensor)) in line_bboxes.iter().zip(line_tensors.iter()).enumerate() {
        let line_gray = gray.slice(s![
            bbox.y as usize..(bbox.y + bbox.height) as usize,
            bbox.x as usize..(bbox.x + bbox.width) as usize
        ]);
        let hash = line_gray
            .as_slice()
            .and_then(|s| glyph_cache.compute_hash(s, bbox.width, bbox.height));

        if let Some(ref h) = hash
            && let Some((text, conf)) = glyph_cache.lookup(h)
        {
            cached_results.push(Some((text.to_string(), conf)));
            continue;
        }

        cached_results.push(None);
        uncached_indices.push(i);
        uncached_tensors.push(tensor.clone());
    }

    // Run inference only on uncached lines
    let outputs = if uncached_tensors.is_empty() {
        Vec::new()
    } else if use_fast_path && uncached_tensors.len() > 1 {
        backend
            .run_batch(&model, &uncached_tensors)
            .context("Batch inference failed")?
    } else {
        uncached_tensors
            .iter()
            .map(|t| backend.run(&model, std::slice::from_ref(t)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Inference failed")?
    };

    // Merge cached and inferred results
    let mut lines = Vec::with_capacity(line_bboxes.len());
    let mut inference_idx = 0;

    for (i, bbox) in line_bboxes.iter().enumerate() {
        let ruby_annotations = ruby_info
            .as_ref()
            .map(|info| {
                info[i]
                    .iter()
                    .map(|rb| RubyAnnotation {
                        ruby_text: String::new(),
                        bbox: *rb,
                        confidence: 0.0,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if let Some((text, confidence)) = &cached_results[i] {
            let mut text = text.clone();
            if let Some(ref corrector) = dict_corrector {
                text = corrector.correct(&text);
            }
            lines.push(TextLine {
                text,
                bbox: *bbox,
                confidence: *confidence,
                ruby: ruby_annotations,
            });
        } else {
            if let Some(output_tensors) = outputs.get(inference_idx)
                && let Some(output) = output_tensors.first()
            {
                let timesteps = output.shape.get(1).copied().unwrap_or(0);
                let num_classes = output.shape.get(2).copied().unwrap_or(0);

                let (mut text, confidence) = match &logit_mask {
                    Some(mask) => ctc_greedy_decode_masked(
                        &output.data,
                        timesteps,
                        num_classes,
                        &charset,
                        mask,
                    ),
                    None => ctc_greedy_decode(&output.data, timesteps, num_classes, &charset),
                };

                // Beam search fallback for low-confidence lines
                if confidence < engine_config.confidence_threshold
                    && engine_config.beam_width > 1
                    && logit_mask.is_none()
                {
                    let (beam_text, beam_conf) = ctc_beam_decode(
                        &output.data,
                        timesteps,
                        num_classes,
                        &charset,
                        engine_config.beam_width,
                    );
                    if beam_conf > confidence {
                        text = beam_text;
                    }
                }

                // Apply dictionary correction if available
                if let Some(ref corrector) = dict_corrector {
                    text = corrector.correct(&text);
                }

                // Insert into glyph cache
                let line_gray = gray.slice(s![
                    bbox.y as usize..(bbox.y + bbox.height) as usize,
                    bbox.x as usize..(bbox.x + bbox.width) as usize
                ]);
                if let Some(hash) = line_gray
                    .as_slice()
                    .and_then(|s| glyph_cache.compute_hash(s, bbox.width, bbox.height))
                {
                    glyph_cache.insert(hash, text.clone(), confidence);
                }

                lines.push(TextLine {
                    text,
                    bbox: *bbox,
                    confidence,
                    ruby: ruby_annotations,
                });
            }
            inference_idx += 1;
        }
    }

    let result = OcrResult {
        pages: vec![Page {
            width,
            height,
            lines,
        }],
    };

    output_result(&result, &args.format)
}

fn build_engine_config(args: &RecognizeArgs) -> EngineConfig {
    let mut builder = EngineConfigBuilder::new();

    if let Some(ref dir) = args.model_dir {
        builder = builder.model_dir(dir.clone());
    }
    if let Some(threads) = args.threads {
        builder = builder.num_threads(threads);
    }

    builder = builder.mode(match args.mode {
        CliMode::Auto => OcrMode::Auto,
        CliMode::Fastest => OcrMode::Fastest,
        CliMode::Accurate => OcrMode::Accurate,
    });

    builder = builder.charset(match args.charset {
        CliCharset::Full => CharsetMode::Full,
        CliCharset::Jis => CharsetMode::Jis,
    });

    if let Some(ref path) = args.dict {
        builder = builder.dict_path(path.clone());
    }

    builder = builder.ruby_separation(args.ruby);

    if let Some(ref path) = args.cascade {
        builder = builder.cascade_model_path(path.clone());
    }

    builder.build()
}

fn output_result(result: &OcrResult, format: &OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Text => output::text::print(result),
        OutputFormat::Json => output::json::print(result),
    }
}
