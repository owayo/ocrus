# ocrus - Lightning-fast Japanese OCR

## Architecture

Cargo workspace with 7 crates:

| Crate | Role |
|-------|------|
| `ocrus-core` | Data models, config, errors, EngineConfig API |
| `ocrus-preproc` | Image preprocessing (SIMD grayscale, Otsu/Sauvola binarize, normalize) |
| `ocrus-layout` | Layout analysis (projection, CCL, vertical, quality gate, ruby separation) |
| `ocrus-recognizer` | CTC recognition (greedy + beam search, JIS charset, dict correction, cascade) |
| `ocrus-nn` | Pure Rust inference engine (.ocnn format, SIMD ops, mmap model loading) |
| `ocrus-dataset` | Training data generation (font rendering, augmentation, font style filtering) |
| `ocrus-cli` | CLI entry point |

## Dependency Graph

```
ocrus-cli → ocrus-core, ocrus-preproc, ocrus-layout, ocrus-recognizer, ocrus-nn, memmap2, rayon
ocrus-preproc → ocrus-core, wide, image
ocrus-layout → ocrus-core, wide, serde, imageproc, image
ocrus-recognizer → ocrus-core, ocrus-nn, daachorse, img_hash
ocrus-nn → ocrus-core, memmap2, wide
```

## Pipeline

```
image(mmap) → grayscale(SIMD) → binarize(Otsu/Sauvola adaptive) → quality gate
  → orientation detect → layout(projection/CCL/vertical) → ruby separation(optional)
  → normalize(SIMD+rayon) → batch inference(ocrus-nn) → cascade(optional)
  → CTC decode(greedy + beam fallback + logit mask) → dict correction → output
```

## CLI Usage

```bash
ocrus recognize image.png                      # Basic recognition
ocrus recognize image.png --charset jis        # JIS X 0208 charset (fewer false positives)
ocrus recognize image.png --dict corrections.txt  # Dictionary-based post-correction
ocrus recognize image.png --mode fastest       # Skip quality gate, use batch inference
ocrus recognize image.png --mode accurate      # Full quality pipeline
ocrus recognize image.png --ruby               # Ruby (furigana) separation
ocrus recognize image.png --cascade model.ocnn # Cascade recognition
ocrus bench image.png                          # Run benchmarks
```

## Key Features

- **SIMD preprocessing**: `wide` crate for grayscale, binarize, normalize, projection (8-16x parallel)
- **Quality Gate**: Automatic image quality assessment (contrast, binarization, skew) for adaptive pipeline
- **Ruby separation**: CCL-based furigana detection and separation from body text
- **Cascade recognition**: Character segmentation → classifier → CTC fallback for speed
- **Custom inference**: Pure Rust `ocrus-nn` engine with .ocnn mmap model format
- **Batch inference**: Multiple lines in single run (padded to max width)
- **JIS X 0208 charset**: Logit masking for Japanese-specific character set
- **Dictionary correction**: Aho-Corasick based post-processing via `daachorse`
- **Zero-copy I/O**: `memmap2` for memory-mapped image loading
- **Parallel preprocessing**: `rayon` for multi-threaded line normalization
- **Softmax-free decode**: CTC greedy decode uses raw logit argmax (no softmax overhead)
- **CCL layout**: Connected component labeling for irregular layouts via `imageproc`
- **Vertical text**: Auto-detection of text orientation with right-to-left column ordering
- **Sauvola binarization**: Adaptive local thresholding with integral image acceleration
- **CTC beam search**: Prefix beam search for low-confidence lines (conditional LM)
- **EngineConfig API**: Builder pattern for library-level OCR configuration
- **Glyph cache**: Perceptual hash (`img_hash`) for caching recognized characters
- **JPEG detection**: Fast JPEG magic byte detection and format-specific decoding

## Training Data Generation

`ocrus-dataset` crate generates training images from system fonts with augmentation:

```bash
# Generate training data (all categories, all fonts)
ocrus dataset generate --output ./training_data \
  --categories hiragana,katakana,joyo_kanji,jis_level1 \
  --samples-per-char 5

# Filter by font style
ocrus dataset generate --output ./training_data \
  --categories hiragana,katakana --font-styles mincho,gothic

# Generate from failure list (re-train weak characters)
ocrus dataset from-failures --failures ./test_results/failures.json \
  --output ./training_data --samples 10
```

## Fine-tuning (Python, requires PaddlePaddle)

Fine-tune PP-OCRv5 recognition model to improve accuracy on weak characters.
**Requires Python 3.12** (PaddlePaddle does not support 3.13+).

```bash
cd scripts

# Install training dependencies
uv pip install -e '.[train]'

# 1. Generate training data (Rust, fast)
ocrus dataset generate --output /tmp/training_data \
  --categories hiragana,katakana,halfwidth_alnum,fullwidth_alnum

# 2. Fine-tune PP-OCRv5 (PaddleOCR tools/train.py)
#    Pretrained weights: models/pretrained/PP-OCRv5_server_rec_pretrained.pdparams
#    Config template: configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml
PYTHONPATH=/path/to/PaddleOCR python3 tools/train.py -c config.yml

# 3. Export to ONNX
uv run export-onnx --model ./output/best_accuracy --output rec_finetuned.onnx --install

# 4. (Optional) INT8 quantization
uv run quantize --input rec_finetuned.onnx --output rec_int8.onnx
```

## Accuracy Testing

```bash
# Run character accuracy test across all fonts (slow, ~10min)
cargo test -p ocrus-cli --test char_accuracy -- --ignored

# A/B test FP32 vs INT8 quantized model
OCRUS_QUANTIZED_MODEL=path/to/rec_int8.onnx \
  cargo test -p ocrus-cli --test char_accuracy -- --ignored
```

Results are exported to `test_results/failures.json` for targeted re-training.

## Development

```bash
cargo build          # Build all crates
cargo test           # Run all tests
cargo clippy         # Lint
cargo bench          # Benchmarks
```

## Models

Models are not included in the repo. Run `models/download.sh` to download.
Default model directory: `~/.ocrus/models/` (override with `OCRUS_MODEL_DIR`)

- `rec.ocnn` - PP-OCRv5 recognition model (.ocnn format, pure Rust inference)
- `dict.txt` - Character dictionary (18,383 chars)
- Input shape: `(1, 3, 48, W)`, normalize: `(px/255 - 0.5) / 0.5`
- ONNX→.ocnn conversion: `scripts/src/ocrus_scripts/convert_to_ocnn.py`
- Source: huggingface.co/monkt/paddleocr-onnx
- Pretrained weights for fine-tuning: `models/pretrained/PP-OCRv5_server_rec_pretrained.pdparams` (214MB)

## Conventions

- Edition: Rust 2024
- Error handling: `thiserror` for library crates, `anyhow` for CLI
- Serialization: `serde` + `serde_json`
- CLI: `clap` derive API
- SIMD: `wide` crate (stable Rust compatible)
- Testing: unit tests in each crate, E2E tests in `ocrus-cli`
