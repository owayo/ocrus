//! OCR pipeline as a reusable library.
//!
//! Everything above the individual algorithm crates lives here: preprocessing, layout,
//! inference and decoding are wired together in one place so the CLI, the Python bindings
//! and tests all exercise the *same* pipeline. Callers that only need "image in, text out"
//! should use this crate rather than reimplementing the wiring.
//!
//! The model is loaded once in [`OcrEngine::new`] and reused for every image. Loading is
//! the expensive part (an 80 MB `.ocnn` mmap plus an 18k-entry charset), so keep one engine
//! alive instead of constructing it per image.
//!
//! ```no_run
//! use ocrus_core::EngineConfigBuilder;
//! use ocrus_engine::OcrEngine;
//!
//! let engine = OcrEngine::new(EngineConfigBuilder::new().build())?;
//! let result = engine.recognize_path(std::path::Path::new("page.png"))?;
//! println!("{}", result.full_text());
//! # Ok::<(), ocrus_core::OcrusError>(())
//! ```

use std::fs::File;
use std::path::{Path, PathBuf};

use image::DynamicImage;
use memmap2::Mmap;
use ndarray::s;
use rayon::prelude::*;

use ocrus_core::error::Result;
use ocrus_core::{
    CharsetMode, EngineConfig, OcrMode, OcrResult, OcrusError, Page, RubyAnnotation, TextLine,
};
use ocrus_layout::{
    TextOrientation, assess_quality, detect_columns_vertical, detect_lines_ccl,
    detect_lines_projection, detect_orientation, separate_ruby, should_use_fast_path,
};
use ocrus_nn::{Executor, Model, NdTensor};
use ocrus_preproc::{binarize_adaptive, normalize_line, normalize_line_vertical, to_grayscale};
use ocrus_recognizer::charset::Charset;
use ocrus_recognizer::{
    DictCorrector, GlyphCache, ctc_beam_decode, ctc_greedy_decode, ctc_greedy_decode_masked,
    ctc_tla_decode,
};

/// Recognition model file name inside the model directory.
pub const REC_MODEL_FILE: &str = "rec.ocnn";
/// Character dictionary file name inside the model directory.
pub const DICT_FILE: &str = "dict.txt";

/// A loaded OCR engine: model, charset and post-processing, ready to recognize images.
pub struct OcrEngine {
    config: EngineConfig,
    executor: Executor,
    charset: Charset,
    /// Allowed-class mask for [`CharsetMode::Jis`]; `None` means every class is allowed.
    logit_mask: Option<Vec<bool>>,
    dict: Option<DictCorrector>,
}

impl OcrEngine {
    /// Load the model, charset and optional correction dictionary described by `config`.
    ///
    /// Fails with a message naming the missing file when the model directory has not been
    /// populated — that is by far the most common setup mistake, and the recovery
    /// (`python models/download.py`) is worth spelling out in the error itself.
    pub fn new(config: EngineConfig) -> Result<Self> {
        let model_path = config.model_dir.join(REC_MODEL_FILE);
        let dict_path = config.model_dir.join(DICT_FILE);

        if !model_path.exists() {
            return Err(missing_model_error(&config.model_dir, &model_path));
        }
        if !dict_path.exists() {
            return Err(missing_model_error(&config.model_dir, &dict_path));
        }

        let charset = Charset::from_file(&dict_path).map_err(|e| {
            OcrusError::Model(format!(
                "failed to load charset {}: {e}",
                dict_path.display()
            ))
        })?;

        // The JIS mask is built from the charset embedded in ocrus-recognizer, so it works
        // without the repository's data/ directory on disk.
        let logit_mask = match config.charset {
            CharsetMode::Jis => Some(Charset::from_jis_embedded().logit_mask(&charset)),
            CharsetMode::Full => None,
        };

        let dict = match &config.dict_path {
            Some(path) => Some(DictCorrector::from_file(path).map_err(|e| {
                OcrusError::Config(format!("failed to load dictionary {}: {e}", path.display()))
            })?),
            None => None,
        };

        let executor = Executor::new(Model::load(&model_path)?);

        Ok(Self {
            config,
            executor,
            charset,
            logit_mask,
            dict,
        })
    }

    /// The configuration this engine was built with.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Recognize an image file.
    ///
    /// The file is memory-mapped rather than read, so large scans do not cost a full copy.
    pub fn recognize_path(&self, path: &Path) -> Result<OcrResult> {
        let file = File::open(path)
            .map_err(|e| OcrusError::Image(format!("failed to open {}: {e}", path.display())))?;
        // SAFETY: standard mmap pattern; the file is not modified by others during OCR.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| OcrusError::Image(format!("failed to mmap {}: {e}", path.display())))?;
        let img = image::load_from_memory(&mmap)
            .map_err(|e| OcrusError::Image(format!("failed to decode {}: {e}", path.display())))?;
        self.recognize_image(&img)
    }

    /// Recognize an encoded image (PNG, JPEG, ...) held in memory.
    pub fn recognize_bytes(&self, data: &[u8]) -> Result<OcrResult> {
        let img = image::load_from_memory(data)
            .map_err(|e| OcrusError::Image(format!("failed to decode image bytes: {e}")))?;
        self.recognize_image(&img)
    }

    /// Recognize raw, already-decoded pixels.
    ///
    /// `channels` is 1 (grayscale), 3 (RGB) or 4 (RGBA), and `data` must be
    /// `width * height * channels` bytes in row-major order. This is the entry point for
    /// callers that already hold pixels — a numpy array on the Python side, a frame from a
    /// camera — and would otherwise have to re-encode a PNG just to hand it over.
    pub fn recognize_raw(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<OcrResult> {
        let expected = width as usize * height as usize * channels as usize;
        if data.len() != expected {
            return Err(OcrusError::Image(format!(
                "raw image size mismatch: got {} bytes, expected {expected} \
                 ({width}x{height}x{channels})",
                data.len(),
            )));
        }

        let img = match channels {
            1 => image::GrayImage::from_raw(width, height, data.to_vec()).map(DynamicImage::from),
            3 => image::RgbImage::from_raw(width, height, data.to_vec()).map(DynamicImage::from),
            4 => image::RgbaImage::from_raw(width, height, data.to_vec()).map(DynamicImage::from),
            other => {
                return Err(OcrusError::Image(format!(
                    "unsupported channel count {other}: expected 1 (gray), 3 (RGB) or 4 (RGBA)"
                )));
            }
        };

        let img = img.ok_or_else(|| OcrusError::Image("failed to build image".to_string()))?;
        self.recognize_image(&img)
    }

    /// Recognize an already-decoded image.
    pub fn recognize_image(&self, img: &DynamicImage) -> Result<OcrResult> {
        let (width, height) = (img.width(), img.height());

        let gray = to_grayscale(img);
        let binary = binarize_adaptive(&gray);

        // NOTE: the quality gate no longer selects a layout algorithm — projection won on
        // measurement (see below) — so `OcrMode` currently does not change the pipeline.
        // Deciding what it should mean, or dropping it, is tracked in todo.md.
        let quality = assess_quality(&binary);
        let _use_fast_path = match self.config.mode {
            OcrMode::Fastest => true,
            OcrMode::Accurate => false,
            OcrMode::Auto => should_use_fast_path(&quality),
        };

        // Vertical layout needs evidence, not a coin flip. `detect_orientation` compares how
        // sharp the row and column projections are, which is meaningless for a single glyph:
        // both profiles look alike and the answer comes out arbitrary. Guessing "vertical"
        // is not a harmless mistake — the crop then gets rotated 90°, which destroys it. So
        // require either more than one column, or a region clearly taller than wide.
        let mut orientation = detect_orientation(&binary);
        if orientation == TextOrientation::Vertical && !looks_vertical(width, height) {
            orientation = TextOrientation::Horizontal;
        }

        // Projection first, connected components only as a fallback.
        //
        // Measured on the 845-image kana benchmark: projection scores 72.5% where CCL scores
        // 54.6%. CCL groups components into lines by vertical overlap, which splits any
        // character whose strokes do not overlap vertically (ニ, 三, ー) into several
        // "lines" that are then recognized separately. It still earns its place when
        // projection finds nothing, but it should not be the default.
        let line_bboxes = match orientation {
            TextOrientation::Vertical => detect_columns_vertical(&binary),
            _ => {
                let lines = detect_lines_projection(&binary);
                if lines.is_empty() {
                    detect_lines_ccl(&binary)
                } else {
                    lines
                }
            }
        };

        // Ruby separation shrinks each line bbox to its body and records the ruby boxes.
        let (line_bboxes, ruby_info) = if self.config.ruby_separation {
            let mut bodies = Vec::with_capacity(line_bboxes.len());
            let mut ruby_map: Vec<Vec<ocrus_core::BBox>> = Vec::with_capacity(line_bboxes.len());
            for bbox in &line_bboxes {
                let sep = separate_ruby(&binary, bbox, orientation);
                bodies.push(sep.body_bbox);
                ruby_map.push(sep.ruby_bboxes);
            }
            (bodies, Some(ruby_map))
        } else {
            (line_bboxes, None)
        };

        if line_bboxes.is_empty() {
            return Ok(OcrResult {
                pages: vec![Page {
                    width,
                    height,
                    lines: vec![],
                }],
            });
        }

        // Vertical columns are rotated 90° because the model only takes horizontal lines.
        let is_vertical = orientation == TextOrientation::Vertical;
        let line_tensors: Vec<NdTensor<f32>> = line_bboxes
            .par_iter()
            .map(|bbox| {
                let line_img = if is_vertical {
                    normalize_line_vertical(&gray, bbox)
                } else {
                    normalize_line(&gray, bbox)
                };
                let shape = line_img.shape().to_vec();
                let data = line_img.into_raw_vec_and_offset().0;
                NdTensor::from_vec(data, &shape)
            })
            .collect();

        // Identical glyphs within one image are recognized once. The cache is per call:
        // it is cheap to rebuild and keeping it in the engine would make `&self` mutable
        // for every caller, including the Python bindings that share one engine.
        let mut glyph_cache = GlyphCache::new(2);
        let mut cached_results: Vec<Option<(String, f32)>> = Vec::with_capacity(line_bboxes.len());
        let mut uncached_tensors: Vec<NdTensor<f32>> = Vec::new();

        for (bbox, tensor) in line_bboxes.iter().zip(line_tensors.iter()) {
            let hash = gray
                .slice(s![
                    bbox.y as usize..(bbox.y + bbox.height) as usize,
                    bbox.x as usize..(bbox.x + bbox.width) as usize
                ])
                .as_slice()
                .and_then(|s| glyph_cache.compute_hash(s, bbox.width, bbox.height));
            if let Some(ref h) = hash
                && let Some((text, conf)) = glyph_cache.lookup(h)
            {
                cached_results.push(Some((text.to_string(), conf)));
                continue;
            }
            cached_results.push(None);
            uncached_tensors.push(tensor.clone());
        }

        // One image at a time: batching only ever amortized the old engine's per-call
        // overhead, and the executor now spends its time inside parallel kernels.
        let outputs = uncached_tensors
            .iter()
            .map(|t| self.executor.run(t.clone()))
            .collect::<Result<Vec<_>>>()?;

        let mut lines = Vec::with_capacity(line_bboxes.len());
        let mut inference_idx = 0;

        for (i, bbox) in line_bboxes.iter().enumerate() {
            let ruby = ruby_info
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
                lines.push(TextLine {
                    text: self.correct(text.clone()),
                    bbox: *bbox,
                    confidence: *confidence,
                    ruby,
                });
                continue;
            }

            if let Some(output) = outputs.get(inference_idx) {
                let (text, confidence) = self.decode(output);
                let text = self.correct(text);

                let hash = gray
                    .slice(s![
                        bbox.y as usize..(bbox.y + bbox.height) as usize,
                        bbox.x as usize..(bbox.x + bbox.width) as usize
                    ])
                    .as_slice()
                    .and_then(|s| glyph_cache.compute_hash(s, bbox.width, bbox.height));
                if let Some(hash) = hash {
                    glyph_cache.insert(hash, text.clone(), confidence);
                }

                lines.push(TextLine {
                    text,
                    bbox: *bbox,
                    confidence,
                    ruby,
                });
            }
            inference_idx += 1;
        }

        Ok(OcrResult {
            pages: vec![Page {
                width,
                height,
                lines,
            }],
        })
    }

    /// CTC-decode one model output, falling back to beam search for low-confidence lines.
    fn decode(&self, output: &NdTensor<f32>) -> (String, f32) {
        let timesteps = output.shape.get(1).copied().unwrap_or(0);
        let num_classes = output.shape.get(2).copied().unwrap_or(0);

        let (text, confidence) = match &self.logit_mask {
            Some(mask) => {
                ctc_greedy_decode_masked(&output.data, timesteps, num_classes, &self.charset, mask)
            }
            None => ctc_greedy_decode(&output.data, timesteps, num_classes, &self.charset),
        };

        // Greedy takes the argmax at each timestep, so a line whose evidence is spread
        // thinly across timesteps collapses to all-blank and comes back empty. Aggregating
        // the per-class probabilities over time recovers those. Only used when greedy found
        // nothing at all: where greedy did read something, it is the more precise answer.
        if text.chars().all(char::is_whitespace) {
            let (tla_text, tla_conf) =
                ctc_tla_decode(&output.data, timesteps, num_classes, &self.charset);
            if !tla_text.chars().all(char::is_whitespace) {
                return (tla_text, tla_conf.max(confidence));
            }
        }

        // Beam search is only worth its cost when greedy is unsure. It is skipped under a
        // logit mask because the beam decoder does not apply the mask.
        if confidence < self.config.confidence_threshold
            && self.config.beam_width > 1
            && self.logit_mask.is_none()
        {
            let (beam_text, beam_conf) = ctc_beam_decode(
                &output.data,
                timesteps,
                num_classes,
                &self.charset,
                self.config.beam_width,
            );
            if beam_conf > confidence {
                return (beam_text, confidence);
            }
        }

        (text, confidence)
    }

    /// Apply the correction dictionary when one is configured.
    fn correct(&self, text: String) -> String {
        match &self.dict {
            Some(corrector) => corrector.correct(&text),
            None => text,
        }
    }
}

/// Whether the evidence really supports reading this image as vertical text.
///
/// Whether this image should be read as vertical text.
///
/// `detect_orientation` compares how sharp the row and column projections are. For a single
/// glyph that comparison is meaningless — the strokes of い or け look exactly like two
/// columns — and on the 845-image kana benchmark it calls 128 images vertical, every one of
/// them wrong. A mistaken "vertical" rotates the crop 90°, so the character is lost: the
/// measured cost is 3.7 points.
///
/// Column shape cannot break the tie either, because `detect_columns_vertical` returns
/// full-height columns whatever the ink looks like. What remains is the region itself:
/// vertical Japanese text is written in tall columns, so a region that is not taller than
/// it is wide is read horizontally.
///
/// This is deliberately one-sided evidence. There is no vertical-text benchmark in the
/// repository, so only the false positives are measurable; see todo.md.
const VERTICAL_ASPECT: f32 = 1.5;

fn looks_vertical(width: u32, height: u32) -> bool {
    height as f32 >= width as f32 * VERTICAL_ASPECT
}

fn missing_model_error(model_dir: &Path, missing: &Path) -> OcrusError {
    OcrusError::Model(format!(
        "model file not found: {}\n\
         model directory: {}\n\
         run `python models/download.py`, or point OCRUS_MODEL_DIR / EngineConfig::model_dir \
         at a directory containing {REC_MODEL_FILE} and {DICT_FILE}",
        missing.display(),
        model_dir.display(),
    ))
}

/// Whether a model directory holds everything [`OcrEngine::new`] needs.
///
/// Useful for telling a user what to download before paying the cost of loading a model.
pub fn models_ready(model_dir: &Path) -> bool {
    model_dir.join(REC_MODEL_FILE).is_file() && model_dir.join(DICT_FILE).is_file()
}

/// The model directory used when the caller does not specify one.
///
/// `OCRUS_MODEL_DIR` wins, otherwise `~/.ocrus/models`.
pub fn default_model_dir() -> PathBuf {
    EngineConfig::default().model_dir
}
