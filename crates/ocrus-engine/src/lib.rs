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
use ocrus_preproc::{
    binarize_adaptive, normalize_line, normalize_line_vertical, thicken, thin, to_grayscale,
};
use ocrus_recognizer::charset::Charset;
use ocrus_recognizer::{
    DictCorrector, ctc_beam_decode, ctc_greedy_decode, ctc_greedy_decode_masked, ctc_tla_decode,
};

/// Recognition model file name inside the model directory.
pub const REC_MODEL_FILE: &str = "rec.ocnn";
/// Character dictionary file name inside the model directory.
pub const DICT_FILE: &str = "dict.txt";

/// A detected line and the exact model inputs used by recognition.
pub struct PreparedLine {
    pub bbox: ocrus_core::BBox,
    /// NCHW float32 inputs, including stroke variants in accurate mode.
    pub inputs: Vec<NdTensor<f32>>,
    pub ruby_bboxes: Vec<ocrus_core::BBox>,
}

/// Layout and normalized inputs for a decoded image, without running a model.
pub struct PreparedImage {
    pub width: u32,
    pub height: u32,
    pub lines: Vec<PreparedLine>,
}

/// Prepare an image through the same layout and normalization as [`OcrEngine`].
///
/// Training and diagnostics can consume these inputs without duplicating the OCR
/// pipeline or loading a recognition model. Blank images produce no lines.
pub fn prepare_image(img: &DynamicImage, config: &EngineConfig) -> PreparedImage {
    let (width, height) = (img.width(), img.height());

    let gray = to_grayscale(img);
    let binary = binarize_adaptive(&gray);

    // NOTE: the quality gate no longer selects a layout algorithm — projection won on
    // measurement (see below) — so `OcrMode` currently does not change the pipeline.
    // Deciding what it should mean, or dropping it, is tracked in todo.md.
    let quality = assess_quality(&binary);
    let _use_fast_path = match config.mode {
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
    let (line_bboxes, ruby_info) = if config.ruby_separation {
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

    // Strokes are not lines. Characters whose upper stroke stands clear of the body
    // (う, こ, き, ふ, え) split at that gap: 74 of the 845 benchmark images came back as
    // two or three "lines", were recognized as separate fragments and concatenated into
    // nonsense. In a square-ish frame that the ink fills, everything found is one
    // character; a page of text has a frame much wider than it is tall.
    let line_bboxes = merge_strokes_of_one_glyph(line_bboxes, width, height);
    let line_bboxes = keep_frame_for_tiny_mark(line_bboxes, width, height);

    if line_bboxes.is_empty() {
        return PreparedImage {
            width,
            height,
            lines: vec![],
        };
    }

    // In accurate mode the same crop is read three ways — as rendered, with strokes
    // thickened, and with them thinned — and the answers are voted on. A glyph whose
    // strokes are too fine or too heavy for its size reads differently under each.
    let variants: Vec<ndarray::Array2<u8>> = if matches!(config.mode, OcrMode::Accurate) {
        vec![gray.clone(), thicken(&gray), thin(&gray)]
    } else {
        vec![gray.clone()]
    };

    // Vertical columns are rotated 90° because the model only takes horizontal lines.
    let is_vertical = orientation == TextOrientation::Vertical;
    let line_tensors: Vec<Vec<NdTensor<f32>>> = line_bboxes
        .par_iter()
        .map(|bbox| {
            variants
                .iter()
                .map(|src| {
                    let line_img = if is_vertical {
                        normalize_line_vertical(src, bbox)
                    } else {
                        normalize_line(src, bbox)
                    };
                    let shape = line_img.shape().to_vec();
                    let data = line_img.into_raw_vec_and_offset().0;
                    NdTensor::from_vec(data, &shape)
                })
                .collect()
        })
        .collect();

    let lines = line_bboxes
        .into_iter()
        .zip(line_tensors)
        .enumerate()
        .map(|(i, (bbox, inputs))| PreparedLine {
            bbox,
            inputs,
            ruby_bboxes: ruby_info
                .as_ref()
                .map(|info| info[i].clone())
                .unwrap_or_default(),
        })
        .collect();
    PreparedImage {
        width,
        height,
        lines,
    }
}

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
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|size| size.checked_mul(channels as usize))
            .ok_or_else(|| OcrusError::Image("raw image dimensions overflow usize".into()))?;
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
        let prepared = prepare_image(img, &self.config);
        let (width, height) = (prepared.width, prepared.height);

        // 行ごとに推論する。各行の前処理バリアントは同じ順序で保持する。
        let outputs = prepared
            .lines
            .iter()
            .map(|line| {
                line.inputs
                    .iter()
                    .map(|t| self.executor.run(t.clone()))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let mut lines = Vec::with_capacity(prepared.lines.len());
        for (line, output) in prepared.lines.iter().zip(&outputs) {
            let bbox = &line.bbox;
            let ruby = line
                .ruby_bboxes
                .iter()
                .map(|rb| RubyAnnotation {
                    ruby_text: String::new(),
                    bbox: *rb,
                    confidence: 0.0,
                })
                .collect();

            let (text, confidence) = self.decode_voted(output);
            let text = correct_small_kana(&text, bbox, height).unwrap_or(text);
            let text = self.correct(text);

            lines.push(TextLine {
                text,
                bbox: *bbox,
                confidence,
                ruby,
            });
        }

        Ok(OcrResult {
            pages: vec![Page {
                width,
                height,
                lines,
            }],
        })
    }

    /// Decode every variant of a line and take the answer they agree on.
    ///
    /// Agreement is the useful signal here: CTC confidences are not comparable between
    /// differently preprocessed inputs, but two variants landing on the same string is
    /// evidence in a way one variant's score is not. With a single variant this is just
    /// [`Self::decode`].
    fn decode_voted(&self, outputs: &[NdTensor<f32>]) -> (String, f32) {
        let decoded: Vec<(String, f32)> = outputs.iter().map(|o| self.decode(o)).collect();
        let Some((first_text, first_conf)) = decoded.first().cloned() else {
            return (String::new(), 0.0);
        };
        if decoded.len() == 1 {
            return (first_text, first_conf);
        }

        let mut best: Option<(String, f32, usize)> = None;
        for (text, conf) in &decoded {
            let votes = decoded.iter().filter(|(t, _)| t == text).count();
            let better = match &best {
                None => true,
                Some((_, best_conf, best_votes)) => {
                    votes > *best_votes || (votes == *best_votes && conf > best_conf)
                }
            };
            if better {
                best = Some((text.clone(), *conf, votes));
            }
        }
        // A tie between three different answers falls back to the unmodified crop, which is
        // the one the model was trained to expect.
        match best {
            Some((text, conf, votes)) if votes > 1 => (text, conf),
            _ => (first_text, first_conf),
        }
    }

    /// モデル出力を CTC でデコードし、空出力や低信頼度の行には補助経路を使う。
    fn decode(&self, output: &NdTensor<f32>) -> (String, f32) {
        let timesteps = output.shape.get(1).copied().unwrap_or(0);
        let num_classes = output.shape.get(2).copied().unwrap_or(0);

        if timesteps == 0 || num_classes == 0 {
            return (String::new(), 0.0);
        }

        let (text, confidence) = match &self.logit_mask {
            Some(mask) => {
                ctc_greedy_decode_masked(&output.data, timesteps, num_classes, &self.charset, mask)
            }
            None => ctc_greedy_decode(&output.data, timesteps, num_classes, &self.charset),
        };

        // 各時刻の argmax がすべて blank でも、文字への支持を時刻間で集計すると
        // 読み取れる場合がある。この補助経路は greedy が空出力のときだけ使う。
        if text.chars().all(char::is_whitespace) {
            // 空出力からの復旧でも、greedy と同じ文字集合の制限を適用する。
            let masked_logits = self.logit_mask.as_ref().map(|mask| {
                output
                    .data
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| {
                        if mask[index % num_classes] {
                            value
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<_>>()
            });
            let logits = masked_logits.as_deref().unwrap_or(&output.data);
            let (tla_text, tla_conf) =
                ctc_tla_decode(logits, timesteps, num_classes, &self.charset);
            if !tla_text.chars().all(char::is_whitespace) {
                return (tla_text, tla_conf.max(confidence));
            }
        }

        // 低信頼度の行だけ beam search を試す。beam 側はマスクに対応していないため、
        // 文字集合の制限がある場合は実行しない。
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

/// The small (捨て仮名) counterpart of a full-size kana, if it has one.
fn small_kana_variant(c: char) -> Option<char> {
    Some(match c {
        'あ' => 'ぁ',
        'い' => 'ぃ',
        'う' => 'ぅ',
        'え' => 'ぇ',
        'お' => 'ぉ',
        'つ' => 'っ',
        'や' => 'ゃ',
        'ゆ' => 'ゅ',
        'よ' => 'ょ',
        'わ' => 'ゎ',
        'か' => 'ゕ',
        'け' => 'ゖ',
        'ア' => 'ァ',
        'イ' => 'ィ',
        'ウ' => 'ゥ',
        'エ' => 'ェ',
        'オ' => 'ォ',
        'ツ' => 'ッ',
        'ヤ' => 'ャ',
        'ユ' => 'ュ',
        'ヨ' => 'ョ',
        'ワ' => 'ヮ',
        'カ' => 'ヵ',
        'ケ' => 'ヶ',
        _ => return None,
    })
}

/// Where the ink sits vertically, as a fraction of the frame. Small kana are drawn in the
/// lower part of the em box, which is the one cue that survives the crop.
const SMALL_KANA_CENTER: f32 = 0.525;

/// Rewrite a single recognized kana to its small form when the ink sits low in the frame.
///
/// The model resizes every crop to the same height, so あ and ぁ arrive looking identical —
/// half of all remaining errors on the kana benchmark are exactly this confusion. Size does
/// not separate them (the ranges overlap: small reaches 0.41 of the frame, full-size starts
/// at 0.07), but vertical position does: measured over 200 rendered glyphs, small kana sit
/// at 0.52–0.59 of the frame height and full-size ones at 0.30–0.52. Sweeping the
/// threshold over the benchmark gives a plateau at 0.52–0.53 (82.4% / 82.2%) that falls
/// away sharply on both sides (0.51 → 80.2%, 0.55 → 77.5%), so the value sits in the
/// middle of that plateau rather than on the measured peak.
///
/// Only applied to a single-character result. Within a line of text the crop covers many
/// characters, so there is no per-character geometry to read — and there the model has the
/// neighbouring characters to judge size against, which is why it gets those right anyway.
fn correct_small_kana(text: &str, bbox: &ocrus_core::BBox, frame_height: u32) -> Option<String> {
    let mut chars = text.chars();
    let only = chars.next()?;
    if chars.next().is_some() || frame_height == 0 {
        return None;
    }
    let small = small_kana_variant(only)?;
    let center = (bbox.y as f32 + bbox.height as f32 / 2.0) / frame_height as f32;
    (center >= SMALL_KANA_CENTER).then(|| small.to_string())
}

/// Ink this much smaller than its frame is a punctuation mark rather than a character.
const TINY_MARK_COVERAGE: f32 = 0.20;

/// Keep the frame when the ink is a tiny mark.
///
/// A comma or a period is a few pixels of ink. Cropping to that ink and resizing it to the
/// model's fixed height blows it up into a blob that reads as a letter — `.` comes back as
/// `a`, `,` as `l`. Keeping the frame preserves what actually identifies these marks: how
/// small they are and where they sit.
fn keep_frame_for_tiny_mark(
    boxes: Vec<ocrus_core::BBox>,
    width: u32,
    height: u32,
) -> Vec<ocrus_core::BBox> {
    if boxes.len() != 1 || height == 0 {
        return boxes;
    }
    let frame_aspect = width as f32 / height as f32;
    if !(1.0 / GLYPH_FRAME_ASPECT..=GLYPH_FRAME_ASPECT).contains(&frame_aspect) {
        return boxes;
    }
    let b = boxes[0];
    if (b.height as f32) / height as f32 <= TINY_MARK_COVERAGE {
        vec![ocrus_core::BBox::new(b.x, 0, b.width, height)]
    } else {
        boxes
    }
}

/// Frame shapes that can only hold a single character rather than lines of text.
const GLYPH_FRAME_ASPECT: f32 = 2.0;
/// How much of the frame the ink must cover before it is read as one character.
const GLYPH_INK_COVERAGE: f32 = 0.4;

/// Merge separately detected strokes back into the single character they belong to.
///
/// Row projection cannot tell a gap between text lines from the gap under the top stroke of
/// う. What separates the two cases is the frame: a page holding several lines is much wider
/// than one line is tall, while an isolated glyph sits in a roughly square box that its ink
/// fills. Only that second case is merged.
fn merge_strokes_of_one_glyph(
    boxes: Vec<ocrus_core::BBox>,
    width: u32,
    height: u32,
) -> Vec<ocrus_core::BBox> {
    if boxes.len() < 2 || height == 0 {
        return boxes;
    }
    let frame_aspect = width as f32 / height as f32;
    if !(1.0 / GLYPH_FRAME_ASPECT..=GLYPH_FRAME_ASPECT).contains(&frame_aspect) {
        return boxes;
    }

    let top = boxes.iter().map(|b| b.y).min().unwrap_or(0);
    let bottom = boxes.iter().map(|b| b.y + b.height).max().unwrap_or(0);
    let left = boxes.iter().map(|b| b.x).min().unwrap_or(0);
    let right = boxes.iter().map(|b| b.x + b.width).max().unwrap_or(0);

    if (bottom - top) as f32 / height as f32 >= GLYPH_INK_COVERAGE {
        vec![ocrus_core::BBox::new(left, top, right - left, bottom - top)]
    } else {
        boxes
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ocrus_nn::ocnn::format::{ALIGN, HEADER_LEN, MAGIC, VERSION_MAJOR};

    #[test]
    fn preparation_needs_no_model_and_preserves_blank_pages() {
        let img =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(96, 32, image::Luma([255])));
        for mode in [OcrMode::Auto, OcrMode::Accurate] {
            let config = EngineConfig {
                mode,
                ..EngineConfig::default()
            };
            let prepared = prepare_image(&img, &config);
            assert_eq!((prepared.width, prepared.height), (96, 32));
            assert!(prepared.lines.is_empty());
        }
    }

    #[test]
    fn preparation_keeps_line_order_and_variant_order() {
        let mut pixels = image::GrayImage::from_pixel(160, 64, image::Luma([255]));
        for (top, bottom) in [(8, 20), (40, 52)] {
            for y in top..bottom {
                for x in 10..140 {
                    if x % 20 < 9 {
                        pixels.put_pixel(x, y, image::Luma([0]));
                    }
                }
            }
        }
        let img = DynamicImage::ImageLuma8(pixels);
        let config = EngineConfig {
            mode: OcrMode::Accurate,
            ..EngineConfig::default()
        };
        let prepared = prepare_image(&img, &config);
        assert_eq!(prepared.lines.len(), 2);
        assert!(prepared.lines[0].bbox.y < prepared.lines[1].bbox.y);
        let gray = to_grayscale(&img);
        for line in prepared.lines {
            assert_eq!(line.inputs.len(), 3);
            for (src, input) in [gray.clone(), thicken(&gray), thin(&gray)]
                .iter()
                .zip(line.inputs)
            {
                let expected = normalize_line(src, &line.bbox);
                assert_eq!(input.shape, expected.shape());
                assert_eq!(input.data, expected.into_raw_vec_and_offset().0);
            }
        }
    }

    fn test_engine(mask: Option<Vec<bool>>) -> OcrEngine {
        // 恒等グラフだけのモデルで、実際の重みなしに入力検証とデコードを検証する。
        let metadata = br#"{"converter":"test","source":{"file":"test.onnx","sha256":""},"values":[{"name":"input"}],"tensors":[],"nodes":[],"inputs":[0],"outputs":[0]}"#;
        let len = (HEADER_LEN + metadata.len()).div_ceil(ALIGN as usize) * ALIGN as usize;
        let mut mmap = memmap2::MmapMut::map_anon(len).unwrap();
        mmap[..4].copy_from_slice(MAGIC);
        mmap[4..8].copy_from_slice(&VERSION_MAJOR.to_le_bytes());
        mmap[16..24].copy_from_slice(&(HEADER_LEN as u64).to_le_bytes());
        mmap[24..32].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
        mmap[32..40].copy_from_slice(&(len as u64).to_le_bytes());
        mmap[HEADER_LEN..HEADER_LEN + metadata.len()].copy_from_slice(metadata);
        let model = Model::from_mmap(mmap.make_read_only().unwrap()).unwrap();
        OcrEngine {
            config: EngineConfig::default(),
            executor: Executor::new(model),
            charset: Charset::from_chars(&['あ', '非']),
            logit_mask: mask,
            dict: None,
        }
    }

    #[test]
    fn tla_fallback_respects_the_charset_mask() {
        let output = NdTensor::from_vec(vec![0.6, 0.01, 0.39], &[1, 1, 3]);
        assert_eq!(test_engine(None).decode(&output).0, "非");
        let masked = test_engine(Some(vec![true, true, false]));
        assert_ne!(masked.decode(&output).0, "非");
    }

    #[test]
    fn raw_image_dimensions_cannot_overflow() {
        let engine = test_engine(None);
        assert!(engine.recognize_raw(&[], u32::MAX, u32::MAX, 4).is_err());
    }

    #[test]
    fn raw_image_rejects_invalid_size_and_channels() {
        let engine = test_engine(None);
        assert!(engine.recognize_raw(&[], 2, 2, 1).is_err());
        assert!(engine.recognize_raw(&[0; 8], 2, 2, 2).is_err());
    }

    #[test]
    fn masked_decode_handles_empty_or_short_outputs() {
        let engine = test_engine(Some(vec![true, true, false]));
        for output in [
            NdTensor::from_vec(vec![0.5], &[1]),
            NdTensor::from_vec(vec![], &[1, 0, 3]),
            NdTensor::from_vec(vec![], &[1, 1, 0]),
        ] {
            assert_eq!(engine.decode(&output), (String::new(), 0.0));
        }
    }
}
