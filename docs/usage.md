# Usage

## Recognition

```bash
# Basic recognition
ocrus recognize image.png

# JIS X 0208 charset (fewer false positives for Japanese)
ocrus recognize image.png --charset jis

# Dictionary-based post-correction
ocrus recognize image.png --dict corrections.txt

# Fastest mode (skip quality gate, batch inference)
ocrus recognize image.png --mode fastest

# Accurate mode (full quality pipeline)
ocrus recognize image.png --mode accurate

# Ruby (furigana) separation
ocrus recognize image.png --ruby

# Cascade recognition (requires cascade classifier model)
ocrus recognize image.png --cascade path/to/cascade_model.ocnn
```

## From Python

```python
import ocrus

engine = ocrus.OcrEngine()              # load the model once and reuse it
result = engine.recognize("page.png")   # path / bytes / numpy array / PIL Image

print(result.full_text())
for line in result.pages[0].lines:
    print(line.bbox.as_tuple(), round(line.confidence, 3), line.text)
```

It calls the same pipeline as the CLI, so `to_json()` matches `--format json`. numpy is optional (arrays are read through `shape` and `tobytes()`). See [../python/README.md](../python/README.md).

```bash
make wheel   # build the wheel
```

## Interactive TUI

```bash
ocrus tui
```

Provides a terminal UI menu for common operations:

- E2E Accuracy Test
- Download Models
- ONNX → .ocnn Convert
- Dataset Generate
- Fine-tune
- Export ONNX
- Quantize (INT8)
- Benchmark

## Benchmarks

```bash
ocrus bench image.png -n 100
```

The iteration count `-n` must be at least 1. Dataset generation requires `--chars-per-image` greater than zero and a finite `--val-ratio` between 0 and 1, including both endpoints. Invalid values return an error.

## Training Data Generation

```bash
# Generate training data from system fonts
ocrus dataset generate --output ./training_data --categories hiragana,katakana

# Filter by font style
ocrus dataset generate --output ./training_data \
  --categories hiragana,katakana --font-styles mincho,gothic

# Generate small text using only fonts in the specified directory
ocrus dataset generate --output ./training_data/small-text \
  --font-dirs ./fonts/screen --no-system-fonts \
  --font-styles gothic,monospace --render-heights 12,16,20,24,32,48

# Generate from test failure results (using pre-rendered test images, recommended)
ocrus dataset from-failures --failures ./test_results/failures_step1.json \
  --test-images ./test_images --output ./training_data

# Generate from test failure results (re-render from system fonts)
ocrus dataset from-failures --failures ./failures.json --output ./training_data --all-fonts
```

Font directories are searched recursively, including every face in TTC / OTC collections. Characters missing from a font are excluded before rendering. `--render-heights` sets native canvas heights (default: 48); images are saved at their original resolution and resized by training preprocessing. See [small-text training and evaluation](training-small-text.md) for general font selection criteria and the evaluation plan. These data-generation changes require retraining before they can affect recognition accuracy.
