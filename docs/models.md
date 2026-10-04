# Model Setup

Conversion needs Python 3.12 and uv. Run `make setup` in the repository checkout to install the pinned tools, then download the OCR model and convert it to `.ocnn`:

```bash
./models/download.sh
```

Models are installed to `~/.ocrus/models/` by default. Override with `OCRUS_MODEL_DIR`.

- `rec.ocnn` — PP-OCRv5 recognition model (pure Rust inference, no ONNX Runtime)
- `dict.txt` — Character dictionary (18,383 chars)

To convert an ONNX model to `.ocnn` format:

```bash
uv run --project scripts --extra convert python scripts/src/ocrus_scripts/convert_to_ocnn.py rec.onnx -o ~/.ocrus/models/rec.ocnn
```

## `.ocnn` Format

`.ocnn` (**Oc**rus **N**eural **N**etwork) is a custom binary model format designed for OCRus's pure Rust inference engine (`ocrus-nn`). It eliminates the dependency on ONNX Runtime while enabling zero-copy model loading via `mmap`.

Key characteristics:

- **mmap-friendly**: Tensor payloads are mapped; JSON metadata is parsed and validated
- **Conv+BN+ReLU fusion**: Batch normalization is fused into convolution weights at conversion time
- **Typed graph**: Named parameters, SSA value references, and dynamic dimension expressions
- **f16 weights**: The default model is about 40MB; every half-precision bit pattern is unit tested
- **Golden outputs**: Compare execution results with outputs recorded during conversion

See [the model format](ocnn-format.md) for details.
