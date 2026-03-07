#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${OCRUS_MODEL_DIR:-$HOME/.ocrus/models}"
mkdir -p "$MODEL_DIR"

echo "Downloading OCR models to: $MODEL_DIR"

# PP-OCRv5 Chinese/Japanese recognition model (ONNX converted)
# Source: https://huggingface.co/monkt/paddleocr-onnx
REC_MODEL_URL="${OCRUS_REC_MODEL_URL:-https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/rec.onnx}"
DICT_URL="${OCRUS_DICT_URL:-https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/dict.txt}"

# Optional: lightweight model override (e.g., SVTR-Tiny)
# Set OCRUS_REC_MODEL_URL to use a different model
# Example: export OCRUS_REC_MODEL_URL="https://example.com/svtr_tiny.onnx"

if [ ! -f "$MODEL_DIR/rec.onnx" ]; then
    echo "Downloading recognition model (~80MB)..."
    curl -L -o "$MODEL_DIR/rec.onnx" "$REC_MODEL_URL"
else
    echo "Recognition model already exists, skipping."
fi

if [ ! -f "$MODEL_DIR/dict.txt" ]; then
    echo "Downloading dictionary..."
    curl -L -o "$MODEL_DIR/dict.txt" "$DICT_URL"
else
    echo "Dictionary already exists, skipping."
fi

# Download lightweight model if requested
LITE_MODEL_URL="${OCRUS_LITE_MODEL_URL:-}"
if [ -n "$LITE_MODEL_URL" ]; then
    if [ ! -f "$MODEL_DIR/rec_lite.onnx" ]; then
        echo "Downloading lightweight recognition model..."
        curl -L -o "$MODEL_DIR/rec_lite.onnx" "$LITE_MODEL_URL"
    else
        echo "Lightweight model already exists, skipping."
    fi
fi

# Convert ONNX → .ocnn format
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
if [ ! -f "$MODEL_DIR/rec.ocnn" ] && [ -f "$MODEL_DIR/rec.onnx" ]; then
    echo "Converting rec.onnx → rec.ocnn..."
    cd "$SCRIPT_DIR"
    uv run python src/ocrus_scripts/convert_to_ocnn.py \
        "$MODEL_DIR/rec.onnx" \
        -o "$MODEL_DIR/rec.ocnn"
    echo "Conversion complete."
elif [ -f "$MODEL_DIR/rec.ocnn" ]; then
    echo "rec.ocnn already exists, skipping conversion."
fi

echo ""
echo "Done. Models installed to: $MODEL_DIR"
echo ""
echo "Files:"
ls -lh "$MODEL_DIR"
