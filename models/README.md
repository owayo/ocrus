# Models

OCR models are not included in the repository. Run `download.sh` to download them.

## Required Models

- `rec.onnx` - PP-OCRv5 Chinese/Japanese recognition model (ONNX converted, ~80MB)
- `dict.txt` - Character dictionary (18,383 characters, one per line)

Source: [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx)

## Model Specs

- Input: `(1, 3, 48, W)` — RGB, height=48, dynamic width
- Output: `(1, T, 18385)` — CTC logits (T=timesteps, 18385=num_classes including blank)
- Preprocessing: normalize by `(pixel / 255.0 - 0.5) / 0.5`

## Download

```bash
./models/download.sh
```

Models will be installed to `~/.ocrus/models/` by default.
Set `OCRUS_MODEL_DIR` to override the location.
