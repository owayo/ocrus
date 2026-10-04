# `.ocnn` model format

`ocrus-nn` stores models as compiled execution artifacts. ONNX is the input format; `.ocnn` is the distribution and execution format.

## Design background

An outdated converter once produced a model that loaded successfully but returned incorrect answers with confidence 0.9. Recognition accuracy fell to 0% on 2026-09-02. The format lacked a way to detect the mismatch.

Measurements made before the redesign:

| Item | v2 measurement | Implication |
|---|---|---|
| Model loading | 117 µs (mmap) | Loading was already fast |
| Inference (W=104) | 592 ms | Inference dominated runtime |
| Size | 80.5 MB (all f32) | Convolution accounted for 87%, classifier for 10% |
| Detecting invalid artifacts | Impossible | Detection was the first priority |

The format therefore supports correctness checks, kernel-friendly layouts and smaller files.

## File structure

```text
[0, 64)                 Fixed 64-byte header
[meta_offset, +meta_len) Metadata (JSON, UTF-8)
[data_offset, +data_len) Tensor payloads (64-byte alignment)
```

All header fields use little-endian byte order:

| Offset | Type | Contents |
|---|---|---|
| 0 | `u8[4]` | Magic `"OCNN"` |
| 4 | `u32` | Major version (3) |
| 8 | `u32` | Minor version |
| 12 | `u32` | Reserved |
| 16 | `u64` | Metadata offset |
| 24 | `u64` | Metadata length |
| 32 | `u64` | Tensor payload offset (64B aligned) |
| 40 | `u64` | Tensor payload length |
| 48 | `u32` | Metadata CRC32 |
| 52 | `u32` | Tensor payload CRC32 |
| 56 | `u8[8]` | Reserved |

Loading validates section ranges, tensor dimensions and byte counts, and symbol references in dimension expressions. Models whose range or element-count calculations overflow are rejected.

### JSON metadata

Readable metadata helps investigate conversion errors. For 383 nodes, metadata occupies a few hundred KB and takes 1–7 ms to parse; inference takes tens to hundreds of milliseconds. A binary record format would require matching parsers for ten record types in Rust and Python. CBOR remains an option if loading becomes a bottleneck.

## Metadata contents

The following sketch uses ellipses for omitted entries:

```text
{
  "converter": "ocrus convert_to_ocnn 1.0",
  "created_utc": "2026-09-02T22:55:00Z",
  "source": { "file": "rec.onnx", "sha256": "26fa4f47…", "opset": 11 },
  "symbols": ["W"],
  "values":  [ { "name": "x", "shape": [{"c":1},{"c":3},{"c":48},{"sym":0}] }, … ],
  "tensors": [ { "name": "conv1.w", "dtype": "f16", "layout": "row",
                 "shape": [64,3,3,3], "offset": 0, "len": 3456, "crc32": 123 }, … ],
  "nodes":   [ { "name": "Conv.0", "op": "conv2d", "stride": [2,2], "pad": [1,1,1,1],
                 "dilation": [1,1], "groups": 1, "act": "relu",
                 "inputs": [{"v":0},{"t":0},{"t":1}], "output": 1 }, … ],
  "inputs":  [0],
  "outputs": [181],
  "golden":  [ { "seed": 12345, "width": 64, "out_shape": [1,8,18385], "argmax": [0,0,…] } ]
}
```

### 1. Named and typed parameters

The previous format packed operation parameters into an anonymous `u32[10]`, leaving their meanings implicit in the converter and executor. Each operation now has named fields.

```rust
Op::Conv2d { stride: [usize; 2], pad: [usize; 4], dilation: [usize; 2],
             groups: usize, act: Act }
```

### 2. SSA value IDs

Values and tensors use separate reference spaces: `{"v":3}` and `{"t":12}`. The former four-input limit is removed. Loading validates topological order, single assignment and reference existence.

### 3. Dynamic dimension expressions

Input width `W` is a declared symbol. Derived dimensions use `(W * mul + add) / div` expressions.

Shape-only `Shape`, `Gather`, `Slice` and `Concat` subgraphs are folded during conversion. Previously, shape values flowed through f32 tensors and `Reshape` converted them back to integers.

The converter measures intermediate shapes by running ONNX Runtime at multiple widths and fits affine expressions. Conversion fails if a dimension cannot be fitted.

### 4. Golden outputs

Conversion records argmax sequences for deterministic pseudorandom inputs, storing their seeds and widths. The executor can recreate the inputs and compare its results.

```bash
cargo test -p ocrus-nn --release --test ocnn_golden -- --nocapture
```

Invalid execution artifacts fail this check. Verification is separate from loading because it costs one inference run.

## Tensors

- `dtype`: `f32` or `f16`. f16 is the default; it halves weight size, preserves the measured f32 golden argmax and was 16% faster in the recorded benchmark.
- `layout`: Currently `row` (row-major). Kernel-specific layouts such as `conv_k8` can be added later.
- Payloads are aligned to 64-byte boundaries and validated during loading. The previous format cast `*const u8` to `*const f32` without alignment guarantees.

## Executor

The implementation lives in `crates/ocrus-nn/src/ocnn/`:

| File | Role |
|---|---|
| `format.rs` | Header, metadata, validation and tensor reading |
| `exec.rs` | Topological execution, register lifetime management and profiling |
| `verify.rs` | Reproducing golden outputs |

Errors include the node index, node name and operation name:

```text
node 106 (Reshape.3, reshape): reshape to [0, 8, 3, 8, 15] does not match the 2880 elements
```

## Conversion

```bash
uv run --with onnx --with onnxruntime --with numpy \
  python scripts/src/ocrus_scripts/convert_to_ocnn.py \
  ~/.ocrus/models/rec.onnx -o ~/.ocrus/models/rec.ocnn --dtype f16
```

The converter:

1. Measures every intermediate value with ONNX Runtime for shape fitting, constant folding and golden verification.
2. Folds nodes whose inputs are all constants into measured values.
3. Folds shape subgraphs into dimension expressions.
4. Fuses BatchNorm into Conv, activations into Conv, and decomposed LayerNorm into one operation.
5. Writes typed parameters and SSA references, embedding golden outputs.

For PP-OCRv5 recognition, conversion reduced 383 nodes to 181 and 80.5 MB to 40.2 MB with f16 weights.

## Versioning in the header

The extension remains `.ocnn`. The header's `version` field determines compatibility. Unsupported versions fail with a request to reconvert the model. The v1/v2 loaders and executors have been removed; backward compatibility is intentionally unsupported.

## Possible future changes

| Idea | Potential | Status |
|---|---|---|
| Per-channel symmetric int8 quantization | 40MB → about 21MB | Deferred: limited int8 dot products in `wide` and costly accuracy verification |
| Kernel-specific weight layouts | Convolution still accounts for about 80% | `layout` exists; only `row` is implemented |
| CBOR metadata | Loading 1.2ms → hundreds of µs | Deferred until loading matters |
| Weight deduplication | Small saving | Identical tensors could share one payload |
