# ocrus - Lightning-fast Japanese OCR

## Concept

**Pure Rust で完結する高速日本語 OCR。外部ランタイム依存ゼロ。**

- ONNX Runtime (ort) などの外部推論ライブラリに依存しない
- 自作推論エンジン `ocrus-nn` で PP-OCRv5 モデルを純 Rust で実行
- `.ocnn` バイナリモデルフォーマット（mmap、型付きグラフ、f16、ゴールデン検証つき）
- SIMD (`wide` crate) による前処理・推論の高速化
- ビルドに C/C++ コンパイラや cmake を必要としない

## Architecture

Cargo workspace with 9 crates:

| Crate | Role |
|-------|------|
| `ocrus-core` | Data models, config, errors, EngineConfig API |
| `ocrus-preproc` | Image preprocessing (SIMD grayscale, Otsu/Sauvola binarize, normalize) |
| `ocrus-layout` | Layout analysis (projection, CCL, vertical, quality gate, ruby separation) |
| `ocrus-recognizer` | CTC recognition (greedy + beam search, JIS charset, dict correction, cascade) |
| `ocrus-nn` | Pure Rust inference engine (.ocnn format, SIMD ops, mmap model loading) |
| `ocrus-engine` | **OCR pipeline** (preproc → layout → inference → decode). CLI and Python both call this |
| `ocrus-dataset` | Training data generation (font rendering, pre-rendered images, augmentation, font style filtering) |
| `ocrus-cli` | CLI entry point (thin wrapper over `ocrus-engine`) |
| `ocrus-python` | PyO3 bindings, built as the `ocrus` wheel with maturin (`python/`) |

## Dependency Graph

```
ocrus-cli → ocrus-engine, ocrus-core, ocrus-dataset, clap
ocrus-python → ocrus-engine, ocrus-core, pyo3
ocrus-engine → ocrus-core, ocrus-preproc, ocrus-layout, ocrus-recognizer, ocrus-nn, memmap2, rayon, image
ocrus-preproc → ocrus-core, wide, image
ocrus-layout → ocrus-core, wide, serde, imageproc, image
ocrus-recognizer → ocrus-core, ocrus-nn, daachorse, img_hash
ocrus-nn → ocrus-core, memmap2, wide
```

**Recognition logic lives only in `ocrus-engine`.** The CLI and the Python bindings are thin
translation layers, so both always produce the same result. Do not add pipeline logic to
either of them.

Note: `crates/ocrus-cli/tests/char_accuracy.rs` runs its *own* path (`normalize_line_scaled`
+ TLA decode), which is not what production uses. Its numbers are not comparable with the
engine's output.

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

## Python Bindings

Same pipeline, exposed as the `ocrus` package (PyO3 + maturin, abi3 wheels).

```python
import ocrus

engine = ocrus.OcrEngine(mode="accurate", charset="jis")   # load the model once
result = engine.recognize("page.png")                      # path / bytes / numpy / PIL

print(result.full_text())
result.to_json()      # identical to `ocrus recognize --format json`
```

```bash
cd python && uvx maturin build --release --out ../target/wheels   # build the wheel
uv run --with ./target/wheels/<wheel> --with pytest python -m pytest python/tests -q
```

- Source layout: `crates/ocrus-python` (Rust) + `python/ocrus` (wrapper, stubs)
- numpy is optional: arrays are read through the buffer protocol
- Run pytest from the repo root; from inside `python/` the source package shadows the wheel
- `pyo3/extension-module` must stay out of the crate's default features, otherwise
  `cargo test --workspace` fails to link on Linux/macOS

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
- **Robust CTC decoding**: Handles short logits and NaN inputs without panicking
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

# Generate from failure list using pre-rendered test images (recommended)
ocrus dataset from-failures --failures ./test_results/failures.json \
  --test-images ./test_images --output ./training_data --samples 10

# Generate from failure list by re-rendering from system fonts
ocrus dataset from-failures --failures ./test_results/failures.json \
  --output ./training_data --samples 10 --all-fonts
```

## Fine-tuning (Python, requires PaddlePaddle)

Fine-tune PP-OCRv5 recognition model to improve accuracy on weak characters.
**Requires Python 3.12** (PaddlePaddle does not support 3.13+).

```bash
cd scripts

# Install training dependencies
uv sync --extra train

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
cargo test -p ocrus-cli --test char_accuracy -- --ignored --nocapture

# A/B test FP32 vs INT8 quantized model
OCRUS_QUANTIZED_MODEL=path/to/rec_int8.onnx \
  cargo test -p ocrus-cli --test char_accuracy -- --ignored --nocapture
```

Results are exported to:
- `logs/char_accuracy_*.log` — Full test log (accuracy per font/category, timing)
- `test_results/failures.json` — Failed characters for targeted re-training

## AI Agent Rules

- 長時間かかるコマンド（E2Eテスト、モデル変換など）はAI側で実行せず、ユーザーに実行を依頼すること。AIセッション終了時にコマンドも終了してしまうため。
- 依存更新・ビルド確認・コミットまでの定期保守は `.claude/skills/ocrus-maintenance`、
  精度改善（char_accuracy の解析と実験）は `.claude/skills/ocrus-model-improvement` に手順がある。
  「メンテナンスして」「精度を上げて」の依頼ではまずそれぞれのスキルを読むこと。

## Development

```bash
cargo build          # Build all crates
cargo test           # Run all tests (no model required; OCR tests self-skip)
cargo clippy         # Lint
cargo bench          # Benchmarks

# End-to-end OCR smoke check (~20s, needs the model)
cargo test -p ocrus-engine --release --test smoke -- --nocapture
```

## Model Format (.ocnn)

The current format is `.ocnn`; see [docs/ocnn-format.md](docs/ocnn-format.md). Highlights:

- **Typed, named op parameters** (the old format packed them into an anonymous `[u32; 10]`)
- **SSA values**; shape-only subgraphs folded into `(W*mul + add)/div` dimension expressions
- **f16 weights by default**: 80.5 MB → 40.2 MB, same golden argmax, ~16% faster
- **Embedded golden outputs**: `cargo test -p ocrus-nn --release --test v3_golden` proves a
  model file computes what its converter measured. A stale artifact used to load happily and
  answer confidently wrong; that class of bug is now caught.

```bash
# ONNX -> .ocnn
uv run --with onnx --with onnxruntime --with numpy \n  python scripts/src/ocrus_scripts/convert_to_ocnn.py \n  ~/.ocrus/models/rec.onnx -o ~/.ocrus/models/rec.ocnn --dtype f16
```

**No backward compatibility.** The version lives in the header; a model from an older
format is rejected with "re-convert it" rather than read by a second code path. The
v1/v2 loader and executor have been deleted.

## Models

Models are not included in the repo. Run `models/download.sh` to download.
Default model directory: `~/.ocrus/models/` (override with `OCRUS_MODEL_DIR`)

- `rec.ocnn` - PP-OCRv5 recognition model (.ocnn format, f16, 40 MB)
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
