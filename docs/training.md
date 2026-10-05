# Fine-tuning

Fine-tune the PP-OCRv5 recognition model to improve accuracy on specific characters. The pipeline consists of 4 steps: data generation (Rust) → training (Python/PaddleOCR) → ONNX export → model replacement.

> **Note**: Training requires **Python 3.12** (PaddlePaddle 3.3 does not support 3.13+). The `scripts/` directory uses [uv](https://docs.astral.sh/uv/) for Python environment management.

### Prerequisites

```bash
# Clone PaddleOCR (training scripts)
git clone https://github.com/PaddlePaddle/PaddleOCR.git /tmp/PaddleOCR

# Install Python training dependencies
uv sync --project scripts --extra train
```

### Step 1: Generate Training Data

`ocrus-dataset` crate generates text images with augmentation (rotation, blur, noise, contrast). Data generation runs in Rust with rayon parallelism.

Two methods are available:
- **`--test-images` (recommended)**: Uses pre-rendered images from `test_images/`. No font installation needed, works cross-platform
- **Font re-rendering**: Renders from system fonts in real-time. Use `--all-fonts` to use all available fonts

```bash
# Generate training data for target character categories (from fonts)
ocrus dataset generate \
  --output /tmp/ocrus_training_data \
  --categories hiragana,katakana,halfwidth_alnum,fullwidth_alnum \
  --samples-per-char 5

# Generate focused data from test failure results (using pre-rendered images, recommended)
ocrus dataset from-failures \
  --failures ./test_results/failures_step1.json \
  --test-images ./test_images \
  --output /tmp/ocrus_training_data \
  --samples-per-char 10

# Generate from failures by re-rendering from system fonts
ocrus dataset from-failures \
  --failures ./test_results/failures_step1.json \
  --output /tmp/ocrus_training_data \
  --samples-per-char 10 --all-fonts
```

Available character categories:

| Category | Content | Count |
|----------|---------|-------|
| `halfwidth_alnum` | Half-width alphanumeric (A-Z, a-z, 0-9) | 62 |
| `halfwidth_symbols` | Half-width symbols (!@#$%&... etc.) | 32 |
| `fullwidth_alnum` | Full-width alphanumeric | 62 |
| `fullwidth_symbols` | Full-width symbols, Japanese punctuation | 63 |
| `hiragana` | Hiragana | 83 |
| `katakana` | Katakana | 86 |
| `joyo_kanji` | Joyo kanji (2010 revision) | 2,136 |
| `jis_level1` | JIS X 0208 Level 1 kanji | 2,965 |
| `jis_level2` | JIS X 0208 Level 2 kanji | 3,390 |
| `jis_level3` | JIS X 0213 Level 3 kanji | 1,233 |
| `jis_level4` | JIS X 0213 Level 4 kanji | 7,960 |

Available font styles (for `--font-styles`):

| Style | Description | Match patterns |
|-------|-------------|----------------|
| `mincho` | Mincho / Serif | mincho, serif, song, batang |
| `gothic` | Gothic / Sans-serif | gothic, sans, kaku, maru |
| `script` | Script / Brush | script, brush, gyosho, kaisho |
| `monospace` | Monospace | mono, courier, consolas, menlo |
| `other` | Unclassified | (default) |

Output format:
```text
/tmp/ocrus_training_data/
  manifest.json      # Metadata (fonts, categories, augment config)
  labels.tsv         # filename \t ground_truth \t category \t font \t augment
  samples/           # Rendered PNG images (height 48px)
    000000.png
    000001.png
    ...
```

The generator writes `labels.tsv` and `samples/`. The `finetune` command creates PaddleOCR `train.txt` / `val.txt` lists during conversion. Prepare the training and validation lists separately when invoking `tools/train.py` directly.

### Use production preprocessing for training inputs

`ocrus_engine::prepare_image` runs the same layout analysis and normalization as recognition without loading a model. It returns each line's bounding box and NCHW tensors: one input in normal mode, or the original, thickened and thinned variants in accurate mode.

The example accepts a JSON array of image paths and writes inputs to a new directory:

```bash
cargo run -p ocrus-engine --release --example prepare_inputs -- images.json inputs
```

`inputs/manifest.json` records source images, line boxes and tensor shapes. Each `.f32` contains little-endian float32 values. Give multi-line images separate line labels or exclude them from single-line training. Keep evaluation texts and fonts out of training and compare candidates through the production recognition pipeline before adoption.

The `recognize_batch` example reuses one `OcrEngine` for evaluation. Pass a model directory and thread count, then provide a JSON array of image paths on stdin. It returns recognition results in input order, using normal mode and the full character set.

```bash
cargo run -p ocrus-engine --release --example recognize_batch -- model-dir 2 < images.json
```

### Step 2: Download Pretrained Weights

```bash
# Download PP-OCRv5 server rec pretrained weights (~214MB)
mkdir -p models/pretrained
curl -L -o models/pretrained/PP-OCRv5_server_rec_pretrained.pdparams \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

### Step 3: Fine-tune with PaddleOCR

#### CPU vs GPU

| | CPU | GPU (e.g. RTX 4060 Super) |
|---|---|---|
| PaddlePaddle | `paddlepaddle-gpu==3.3.0` (CPU execution) | `paddlepaddle-gpu==3.3.0` |
| Config: `use_gpu` | `false` | `true` |
| Config: `batch_size_per_card` | 32 | 128 |
| Config: `num_workers` | 0 | 4-8 |
| Speed | ~18s/batch | ~0.4s/batch |
| 5 epochs (276k images) | ~8 days | ~4 hours |

```bash
# Training dependencies (Linux CUDA distribution; use_gpu: false runs on CPU)
uv sync --project scripts --extra train
```

#### Training Config

Create a YAML config file (based on `PP-OCRv5_server_rec.yml`). Below is a CPU example — for GPU, change `use_gpu`, `batch_size_per_card`, and `num_workers` as shown above.

```yaml
Global:
  model_name: PP-OCRv5_server_rec
  use_gpu: false
  epoch_num: 5
  save_model_dir: /tmp/ocrus_finetune_output
  pretrained_model: ./models/pretrained/PP-OCRv5_server_rec_pretrained
  character_dict_path: /tmp/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt
  max_text_length: &max_text_length 25
  eval_batch_step: [500, 1000]

Optimizer:
  name: Adam
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 1

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /tmp/ocrus_training_data/
    label_file_list:
    - /tmp/ocrus_training_data/train_list.txt
  loader:
    batch_size_per_card: 32
    num_workers: 0
```

See the full config reference at `PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`.

#### Run Training

```bash
PYTHONPATH=/tmp/PaddleOCR:$PYTHONPATH \
  uv run --project scripts --python 3.12 python3 -u /tmp/PaddleOCR/tools/train.py \
  -c /path/to/your_config.yml
```

Training outputs:
```text
/tmp/ocrus_finetune_output/
  train.log              # Training log
  config.yml             # Saved config
  best_accuracy/         # Best model checkpoint
    best_accuracy.pdparams
  latest/                # Latest checkpoint (for resume)
```

To resume from a checkpoint, add to config:
```yaml
Global:
  checkpoints: /tmp/ocrus_finetune_output/latest
```

### Step 4: Export to ONNX

```bash
# Export best model to ONNX
uv run --project scripts export-onnx \
  --model /tmp/ocrus_finetune_output/best_accuracy \
  --output rec_finetuned.onnx

# Install as the default model
uv run --project scripts export-onnx \
  --model /tmp/ocrus_finetune_output/best_accuracy \
  --output rec_finetuned.onnx \
  --install   # Copies to ~/.ocrus/models/rec.onnx

# Convert to .ocnn format
uv run --project scripts --extra convert python scripts/src/ocrus_scripts/convert_to_ocnn.py \
  rec_finetuned.onnx -o ~/.ocrus/models/rec.ocnn
```

### Step 5 (Optional): INT8 Quantization

```bash
uv sync --project scripts --extra quantize

uv run --project scripts quantize \
  --input rec_finetuned.onnx \
  --output rec_int8.onnx
```

### Accuracy Testing

Run character accuracy tests to evaluate the model and identify weak characters. Tests are split into steps so you can iterate incrementally:

| Step | Target | Char count |
|------|--------|------------|
| `step1` | Half/full-width alphanumeric & symbols | ~220 |
| `step2` | Hiragana & Katakana | ~170 |
| `step3_joyo` | Joyo kanji | 2,136 |
| `step3_jis1` | JIS Level 1 kanji | 2,965 |
| `step3_jis2` | JIS Level 2 kanji | 3,390 |
| `step3_jis3` | JIS Level 3 kanji | 1,233 |
| `step3_jis4` | JIS Level 4 kanji | 7,960 |

```bash
# Step 1: Half/full-width alphanumeric & symbols (about 36 minutes, varies by environment)
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture

# Step 2: Hiragana & Katakana
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step2 --release -- --ignored --nocapture

# Step 3: Kanji (run each level separately)
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_joyo --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis1 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis2 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis3 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis4 --release -- --ignored --nocapture

# All steps at once
cargo test -p ocrus-cli --test char_accuracy char_accuracy_all --release -- --ignored --nocapture

# A/B test against a quantized model (combinable with any step)
OCRUS_QUANTIZED_MODEL=rec_int8.onnx \
  cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture
```

Test results are exported to:
- `logs/char_accuracy_{step}_{timestamp}.log` — Full test log (accuracy per font/category, speed, ETA)
- `test_results/failures_{step}.json` — Failed characters (can be fed back into Step 1 `from-failures` for targeted retraining)

Failures are saved incrementally after each category completes. If interrupted with Ctrl+C, results up to that point are also saved.
