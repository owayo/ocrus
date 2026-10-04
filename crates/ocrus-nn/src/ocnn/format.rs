//! The `.ocnn` container: a 64-byte header, a JSON metadata blob, and an aligned
//! tensor payload.
//!
//! ```text
//! [0, 64)                  header: magic "OCNN", format version, section table
//! [meta_offset, +meta_len) metadata (JSON, UTF-8)
//! [data_offset, +data_len) tensor payload, 64-byte aligned
//! ```
//!
//! The version lives in the header, not in the file name: a model is always `rec.ocnn`,
//! and the loader decides what it is by reading the version field.
//!
//! Why JSON for the metadata: this file is a *compiled artifact*, not an interchange
//! format, and the failure mode that actually costs time is a silently wrong conversion.
//! A human-readable graph description can be diffed against the source ONNX by eye; the
//! parse cost (a few ms for a few hundred nodes) is irrelevant next to inference, which
//! takes hundreds of milliseconds. Switching to CBOR later touches one function.
//!
//! Everything the executor needs is explicit and named. The v1/v2 format packed op
//! parameters into an anonymous `[u32; 10]` whose meaning lived only in the heads of the
//! converter and the executor — that is precisely how a model ends up loading fine and
//! answering confidently wrong.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use ocrus_core::OcrusError;
use ocrus_core::error::Result;
use serde::{Deserialize, Serialize};

/// Magic bytes at the start of every `.ocnn` file.
pub const MAGIC: &[u8; 4] = b"OCNN";
/// Fixed header length.
pub const HEADER_LEN: usize = 64;
/// Alignment of the tensor payload and of every tensor inside it.
pub const ALIGN: u64 = 64;
/// Format version this build writes and reads.
pub const VERSION_MAJOR: u32 = 3;
/// Minor version; readers accept anything with the same major.
pub const VERSION_MINOR: u32 = 0;

/// Fixed-size file header.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub version_major: u32,
    pub version_minor: u32,
    pub meta_offset: u64,
    pub meta_len: u64,
    pub data_offset: u64,
    pub data_len: u64,
    pub meta_crc32: u32,
    pub data_crc32: u32,
}

impl Header {
    /// Parse and sanity-check the header.
    ///
    /// Args:
    ///     buf: the first bytes of the file.
    ///
    /// Returns:
    ///     The parsed header, or an error naming what was wrong.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_LEN {
            return Err(bad("file is shorter than the 64-byte header"));
        }
        if &buf[0..4] != MAGIC {
            return Err(bad("not an .ocnn file: the magic bytes do not match"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());

        let header = Self {
            version_major: u32_at(4),
            version_minor: u32_at(8),
            meta_offset: u64_at(16),
            meta_len: u64_at(24),
            data_offset: u64_at(32),
            data_len: u64_at(40),
            meta_crc32: u32_at(48),
            data_crc32: u32_at(52),
        };

        if header.version_major != VERSION_MAJOR {
            return Err(bad(format!(
                "this model is format version {} but this build reads version {VERSION_MAJOR}. \n                 Re-convert it from the source ONNX.",
                header.version_major
            )));
        }
        if !header.data_offset.is_multiple_of(ALIGN) {
            return Err(bad(format!(
                "tensor payload starts at {} which is not {ALIGN}-byte aligned",
                header.data_offset
            )));
        }
        Ok(header)
    }
}

/// Where the model came from. Kept for provenance: a model that answers wrong is often
/// simply an artifact built by an older converter, and this is what makes that visible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub file: String,
    pub sha256: String,
    #[serde(default)]
    pub opset: i64,
}

/// A dimension: either a constant or an affine function of a symbol such as `W`.
///
/// `(sym * mul + add) / div`, integer division. This is what lets the converter delete the
/// Shape/Gather/Slice/Concat subgraphs that only existed to compute shapes at runtime.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    Const {
        c: i64,
    },
    Expr {
        sym: u32,
        #[serde(default = "one")]
        mul: i64,
        #[serde(default)]
        add: i64,
        #[serde(default = "one")]
        div: i64,
    },
}

fn one() -> i64 {
    1
}

impl Dim {
    /// Evaluate against concrete symbol values.
    ///
    /// Args:
    ///     symbols: current value of each symbol, indexed by symbol id.
    ///
    /// Returns:
    ///     The dimension size, or an error if a symbol is unknown or the result is invalid.
    pub fn eval(&self, symbols: &[i64]) -> Result<i64> {
        match *self {
            Dim::Const { c } => Ok(c),
            Dim::Expr { sym, mul, add, div } => {
                let v = *symbols
                    .get(sym as usize)
                    .ok_or_else(|| bad(format!("dimension refers to unknown symbol {sym}")))?;
                if div == 0 {
                    return Err(bad("dimension expression divides by zero"));
                }
                v.checked_mul(mul)
                    .and_then(|value| value.checked_add(add))
                    .and_then(|value| value.checked_div(div))
                    .ok_or_else(|| bad("dimension expression overflows i64"))
            }
        }
    }
}

/// An SSA value: the output of a node, or a graph input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueDesc {
    pub name: String,
    /// Declared shape. Used for validation and for reporting, not for execution.
    #[serde(default)]
    pub shape: Vec<Dim>,
}

/// How a tensor's bytes are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F32,
    F16,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }
}

/// Physical arrangement of a tensor's elements.
///
/// The converter is responsible for writing weights in the layout the kernel wants, so
/// that execution never rearranges anything. `Row` is plain row-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    #[default]
    Row,
}

/// A constant: weights, biases, or any other tensor baked into the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDesc {
    pub name: String,
    pub dtype: DType,
    #[serde(default)]
    pub layout: Layout,
    pub shape: Vec<i64>,
    /// Byte offset within the tensor payload.
    pub offset: u64,
    /// Byte length within the tensor payload.
    pub len: u64,
    #[serde(default)]
    pub crc32: u32,
}

impl TensorDesc {
    /// Number of elements implied by the shape.
    pub fn elem_count(&self) -> usize {
        self.shape.iter().map(|&d| d.max(0) as usize).product()
    }
}

/// Which register or constant a node input reads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputRef {
    Value { v: u32 },
    Tensor { t: u32 },
}

/// Activation fused into the producing op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Act {
    #[default]
    None,
    Relu,
    HardSwish,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolKind {
    Max,
    Avg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryKind {
    Relu,
    HardSwish,
    Sigmoid,
    Sqrt,
}

/// A typed operation. Every parameter has a name and a type — the whole point of v3.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Convolution. Weights (and optional bias) come in as tensor inputs, with batch-norm
    /// and the activation already folded in by the converter.
    Conv2d {
        stride: [usize; 2],
        pad: [usize; 4],
        #[serde(default = "dilation_default")]
        dilation: [usize; 2],
        #[serde(default = "one_usize")]
        groups: usize,
        #[serde(default)]
        act: Act,
    },
    Pool {
        kind: PoolKind,
        kernel: [usize; 2],
        stride: [usize; 2],
        pad: [usize; 4],
        #[serde(default)]
        global: bool,
    },
    MatMul {
        #[serde(default)]
        trans_b: bool,
    },
    LayerNorm {
        axis: i32,
        eps: f32,
    },
    Softmax {
        axis: i32,
    },
    Reshape {
        shape: Vec<Dim>,
    },
    Transpose {
        perm: Vec<usize>,
    },
    Concat {
        axis: i32,
    },
    Slice {
        axis: i32,
        start: Dim,
        end: Dim,
        #[serde(default = "one")]
        step: i64,
    },
    ReduceMean {
        axes: Vec<i32>,
        #[serde(default)]
        keepdims: bool,
    },
    Binary {
        kind: BinKind,
    },
    Unary {
        kind: UnaryKind,
    },
    Gather {
        axis: i32,
    },
    Squeeze {
        axes: Vec<i32>,
    },
    Unsqueeze {
        axes: Vec<i32>,
    },
    Identity,
}

fn dilation_default() -> [usize; 2] {
    [1, 1]
}

fn one_usize() -> usize {
    1
}

/// One node of the graph. Nodes are stored in topological order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDesc {
    #[serde(default)]
    pub name: String,
    pub inputs: Vec<InputRef>,
    /// Value id this node writes.
    pub output: u32,
    #[serde(flatten)]
    pub op: Op,
}

/// A recorded input/output pair, used to prove that a model file actually computes what
/// the converter measured. The input is generated from `seed` so the record stays tiny.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Golden {
    pub seed: u32,
    pub width: u32,
    pub out_shape: Vec<usize>,
    /// argmax over the class axis for each timestep.
    pub argmax: Vec<u32>,
}

/// The complete graph description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub converter: String,
    #[serde(default)]
    pub created_utc: String,
    pub source: SourceInfo,
    /// Names of the dynamic dimensions, e.g. `["W"]`.
    #[serde(default)]
    pub symbols: Vec<String>,
    pub values: Vec<ValueDesc>,
    pub tensors: Vec<TensorDesc>,
    pub nodes: Vec<NodeDesc>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    #[serde(default)]
    pub golden: Vec<Golden>,
}

/// A loaded `.ocnn` model: metadata plus an mmap of the tensor payload.
pub struct Model {
    mmap: Mmap,
    pub header: Header,
    pub meta: ModelMeta,
    /// For each value id, the node index after which it is dead (exclusive).
    pub last_use: Vec<usize>,
}

impl Model {
    /// Load and validate a model file.
    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| bad(format!("cannot open {}: {e}", path.display())))?;
        // SAFETY: the model file is read-only for the lifetime of the mapping.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| bad(format!("cannot mmap {}: {e}", path.display())))?;
        Self::from_mmap(mmap)
    }

    /// Parse a model from an existing mapping.
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let header = Header::parse(&mmap)?;

        let meta_end = header
            .meta_offset
            .checked_add(header.meta_len)
            .ok_or_else(|| bad("metadata section range overflows u64"))?;
        let data_end = header
            .data_offset
            .checked_add(header.data_len)
            .ok_or_else(|| bad("tensor payload range overflows u64"))?;
        if meta_end > mmap.len() as u64 || data_end > mmap.len() as u64 {
            return Err(bad("file is truncated: a section extends past the end"));
        }

        let meta_bytes = &mmap[header.meta_offset as usize..meta_end as usize];
        if header.meta_crc32 != 0 && crc32(meta_bytes) != header.meta_crc32 {
            return Err(bad("metadata checksum mismatch: the file is corrupt"));
        }
        let meta: ModelMeta = serde_json::from_slice(meta_bytes)
            .map_err(|e| bad(format!("cannot parse metadata: {e}")))?;

        let model = Self {
            mmap,
            header,
            meta,
            last_use: Vec::new(),
        };
        model.validate()?;
        let last_use = model.compute_last_use();
        Ok(Self { last_use, ..model })
    }

    /// The tensor payload.
    fn data(&self) -> &[u8] {
        let start = self.header.data_offset as usize;
        let end = start + self.header.data_len as usize;
        &self.mmap[start..end]
    }

    /// Structural checks that catch a malformed or mismatched artifact at load time,
    /// before it can produce confident nonsense.
    fn validate(&self) -> Result<()> {
        let nvalues = self.meta.values.len();
        let ntensors = self.meta.tensors.len();

        for (i, t) in self.meta.tensors.iter().enumerate() {
            let elements = t.shape.iter().try_fold(1usize, |count, &dim| {
                let dim = usize::try_from(dim)
                    .map_err(|_| bad(format!("tensor {i} has an invalid dimension {dim}")))?;
                count
                    .checked_mul(dim)
                    .ok_or_else(|| bad(format!("tensor {i} shape overflows usize")))
            })?;
            let expect = elements
                .checked_mul(t.dtype.size())
                .ok_or_else(|| bad(format!("tensor {i} byte length overflows usize")))?;
            if t.len != expect as u64 {
                return Err(bad(format!(
                    "tensor {i} ({}) declares shape {:?} ({expect} bytes) but stores {} bytes",
                    t.name, t.shape, t.len
                )));
            }
            if t.offset
                .checked_add(t.len)
                .is_none_or(|end| end > self.header.data_len)
            {
                return Err(bad(format!(
                    "tensor {i} ({}) runs past the end of the payload",
                    t.name
                )));
            }
            if !t.offset.is_multiple_of(ALIGN) {
                return Err(bad(format!(
                    "tensor {i} ({}) is at offset {} which is not {ALIGN}-byte aligned",
                    t.name, t.offset
                )));
            }
        }

        for (i, value) in self.meta.values.iter().enumerate() {
            for dim in &value.shape {
                if let Dim::Expr { sym, div, .. } = dim {
                    if *sym as usize >= self.meta.symbols.len() {
                        return Err(bad(format!("value {i} refers to unknown symbol {sym}")));
                    }
                    if *div == 0 {
                        return Err(bad(format!(
                            "value {i} dimension expression divides by zero"
                        )));
                    }
                }
            }
        }

        // Nodes are in topological order: every value read must already be written.
        let mut written = vec![false; nvalues];
        for &v in &self.meta.inputs {
            *written
                .get_mut(v as usize)
                .ok_or_else(|| bad(format!("graph input refers to unknown value {v}")))? = true;
        }
        for (i, node) in self.meta.nodes.iter().enumerate() {
            for input in &node.inputs {
                match *input {
                    InputRef::Value { v } => {
                        if v as usize >= nvalues {
                            return Err(bad(format!("node {i} reads unknown value {v}")));
                        }
                        if !written[v as usize] {
                            return Err(bad(format!(
                                "node {i} ({}) reads value {v} before it is produced: \
                                 the graph is not in topological order",
                                node.name
                            )));
                        }
                    }
                    InputRef::Tensor { t } => {
                        if t as usize >= ntensors {
                            return Err(bad(format!("node {i} reads unknown tensor {t}")));
                        }
                    }
                }
            }
            let out = node.output as usize;
            if out >= nvalues {
                return Err(bad(format!("node {i} writes unknown value {out}")));
            }
            if written[out] {
                return Err(bad(format!(
                    "value {out} is written twice (node {i}); values are single-assignment"
                )));
            }
            written[out] = true;
        }
        for &v in &self.meta.outputs {
            if v as usize >= nvalues {
                return Err(bad(format!("graph output refers to unknown value {v}")));
            }
            if !written[v as usize] {
                return Err(bad(format!("graph output {v} is never produced")));
            }
        }
        if self.meta.inputs.len() != 1 {
            return Err(bad(format!(
                "expected exactly one graph input, found {}",
                self.meta.inputs.len()
            )));
        }
        if self.meta.outputs.len() != 1 {
            return Err(bad(format!(
                "expected exactly one graph output, found {}",
                self.meta.outputs.len()
            )));
        }
        Ok(())
    }

    /// Index of the last node that reads each value, so the executor can drop registers
    /// as soon as they are dead instead of holding every intermediate alive.
    fn compute_last_use(&self) -> Vec<usize> {
        let mut last = vec![0usize; self.meta.values.len()];
        for (i, node) in self.meta.nodes.iter().enumerate() {
            for input in &node.inputs {
                if let InputRef::Value { v } = *input {
                    last[v as usize] = i;
                }
            }
        }
        for &v in &self.meta.outputs {
            last[v as usize] = self.meta.nodes.len();
        }
        last
    }

    /// Read a constant tensor as f32, converting from the stored dtype when needed.
    ///
    /// `f32` tensors are returned as a borrowed slice when the mapping is suitably
    /// aligned, which is the common case: the converter aligns every tensor to 64 bytes.
    pub fn tensor_f32(&self, id: u32) -> Result<std::borrow::Cow<'_, [f32]>> {
        let t = self
            .meta
            .tensors
            .get(id as usize)
            .ok_or_else(|| bad(format!("unknown tensor {id}")))?;
        let start = t.offset as usize;
        let bytes = &self.data()[start..start + t.len as usize];

        match t.dtype {
            DType::F32 => {
                let (head, floats, _) = unsafe { bytes.align_to::<f32>() };
                if head.is_empty() && floats.len() == t.elem_count() {
                    Ok(std::borrow::Cow::Borrowed(floats))
                } else {
                    // The payload is aligned by construction; this is the safety net.
                    let v = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    Ok(std::borrow::Cow::Owned(v))
                }
            }
            DType::F16 => {
                let v = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                    .collect();
                Ok(std::borrow::Cow::Owned(v))
            }
        }
    }

    /// Recompute the payload checksum and compare it with the header.
    ///
    /// Not done on load: hashing 80 MB costs more than everything else about loading.
    pub fn verify_checksums(&self) -> Result<()> {
        if self.header.data_crc32 == 0 {
            return Ok(());
        }
        if crc32(self.data()) != self.header.data_crc32 {
            return Err(bad("tensor payload checksum mismatch: the file is corrupt"));
        }
        Ok(())
    }
}

/// IEEE 754 の半精度値を f32 に変換する。
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = match exp {
        0 if frac == 0 => sign << 31,
        0 => {
            // 非正規化数を正規化し、仮数の移動量を指数に反映する。
            let mut e = 0i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + e + 1) as u32;
            (sign << 31) | (exp32 << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

/// CRC-32 (IEEE), matching Python's `zlib.crc32`.
pub fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { POLY ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }

    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn bad(msg: impl Into<String>) -> OcrusError {
    OcrusError::Model(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        // Reference values from Python's zlib.crc32.
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"hello"), 0x3610_a686);
        assert_eq!(crc32(b"OCNN"), 0x6f71_aaf2);
    }

    #[test]
    fn f16_round_trip() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert!((f16_to_f32(0x3555) - 0.333_251).abs() < 1e-5);
    }

    #[test]
    fn f16_subnormals_match_their_exact_values() {
        // 非正規化数は仮数 × 2^-24。符号を含む全パターンを検証する。
        for fraction in 1u16..=0x03ff {
            let expected = f32::from(fraction) * 2.0f32.powi(-24);
            assert_eq!(f16_to_f32(fraction), expected);
            assert_eq!(f16_to_f32(0x8000 | fraction), -expected);
        }
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn f16_all_bit_patterns_match_reference_arithmetic() {
        for bits in 0..=u16::MAX {
            let sign = if bits & 0x8000 == 0 { 1.0f64 } else { -1.0 };
            let exponent = i32::from((bits >> 10) & 0x1f);
            let fraction = f64::from(bits & 0x3ff);
            let expected = match exponent {
                0 => sign * fraction * 2.0f64.powi(-24),
                31 if fraction == 0.0 => sign * f64::INFINITY,
                31 => {
                    assert!(f16_to_f32(bits).is_nan());
                    continue;
                }
                _ => sign * (1.0 + fraction / 1024.0) * 2.0f64.powi(exponent - 15),
            };
            assert_eq!(
                f16_to_f32(bits).to_bits(),
                (expected as f32).to_bits(),
                "{bits:#06x}"
            );
        }
    }

    fn minimal_metadata() -> serde_json::Value {
        serde_json::json!({
            "converter": "test", "source": { "file": "test.onnx", "sha256": "" },
            "values": [{ "name": "input", "shape": [{ "c": 1 }] }],
            "tensors": [], "nodes": [], "inputs": [0], "outputs": [0]
        })
    }

    fn model_mapping(meta: serde_json::Value) -> memmap2::MmapMut {
        // 匿名 mmap を使い、外部モデルや一時ファイルに依存せず読み込みを検証する。
        let bytes = serde_json::to_vec(&meta).unwrap();
        let data_offset = (HEADER_LEN + bytes.len()).div_ceil(ALIGN as usize) * ALIGN as usize;
        let mut mmap = memmap2::MmapMut::map_anon(data_offset + 4).unwrap();
        mmap[..4].copy_from_slice(MAGIC);
        mmap[4..8].copy_from_slice(&VERSION_MAJOR.to_le_bytes());
        mmap[16..24].copy_from_slice(&(HEADER_LEN as u64).to_le_bytes());
        mmap[24..32].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        mmap[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());
        mmap[40..48].copy_from_slice(&4u64.to_le_bytes());
        mmap[HEADER_LEN..HEADER_LEN + bytes.len()].copy_from_slice(&bytes);
        mmap
    }

    #[test]
    fn model_rejects_section_offset_overflow() {
        for offset in [16, 32] {
            let mut mmap = model_mapping(minimal_metadata());
            mmap[offset..offset + 8].copy_from_slice(&(u64::MAX - 63).to_le_bytes());
            mmap[offset + 8..offset + 16].copy_from_slice(&128u64.to_le_bytes());
            assert!(Model::from_mmap(mmap.make_read_only().unwrap()).is_err());
        }
    }

    #[test]
    fn model_rejects_invalid_tensor_shapes_and_ranges() {
        for (shape, offset, len) in [
            (vec![-1], 0, 0),
            (vec![i64::MAX, 3], 0, 0),
            (vec![i64::MAX], 0, 0),
            (vec![1], u64::MAX - 63, 128),
        ] {
            let mut meta = minimal_metadata();
            meta["tensors"] = serde_json::json!([{
                "name": "weight", "dtype": "f32", "shape": shape,
                "offset": offset, "len": len
            }]);
            assert!(Model::from_mmap(model_mapping(meta).make_read_only().unwrap()).is_err());
        }
    }

    #[test]
    fn model_rejects_unknown_input_symbol() {
        let mut meta = minimal_metadata();
        meta["values"][0]["shape"] = serde_json::json!([{ "sym": 0 }]);
        assert!(Model::from_mmap(model_mapping(meta).make_read_only().unwrap()).is_err());
    }

    #[test]
    fn model_checks_metadata_and_payload_corruption() {
        let mut mmap = model_mapping(minimal_metadata());
        mmap[48..52].copy_from_slice(&1u32.to_le_bytes());
        assert!(Model::from_mmap(mmap.make_read_only().unwrap()).is_err());

        let mut mmap = model_mapping(minimal_metadata());
        mmap[52..56].copy_from_slice(&1u32.to_le_bytes());
        let model = Model::from_mmap(mmap.make_read_only().unwrap()).unwrap();
        assert!(model.verify_checksums().is_err());
    }

    #[test]
    fn dim_expressions_reject_arithmetic_overflow() {
        for (value, mul, add, div) in [
            (i64::MAX, 2, 0, 1),
            (i64::MAX, 1, 1, 1),
            (i64::MIN, 1, 0, -1),
            (1, 1, 0, 0),
        ] {
            assert!(
                Dim::Expr {
                    sym: 0,
                    mul,
                    add,
                    div
                }
                .eval(&[value])
                .is_err()
            );
        }
    }

    #[test]
    fn dim_expressions_evaluate() {
        let w = Dim::Expr {
            sym: 0,
            mul: 1,
            add: 0,
            div: 8,
        };
        assert_eq!(w.eval(&[104]).unwrap(), 13);
        assert_eq!(Dim::Const { c: 48 }.eval(&[104]).unwrap(), 48);
        assert!(
            Dim::Expr {
                sym: 5,
                mul: 1,
                add: 0,
                div: 1
            }
            .eval(&[104])
            .is_err()
        );
    }

    #[test]
    fn header_rejects_a_foreign_file() {
        let buf = vec![0u8; HEADER_LEN];
        let err = Header::parse(&buf).unwrap_err().to_string();
        assert!(err.contains("magic"), "{err}");
    }

    #[test]
    fn header_rejects_an_older_format_version() {
        let mut buf = vec![0u8; HEADER_LEN];
        buf[0..4].copy_from_slice(MAGIC);
        buf[4..8].copy_from_slice(&2u32.to_le_bytes());
        let err = Header::parse(&buf).unwrap_err().to_string();
        assert!(err.contains("Re-convert"), "{err}");
    }
}
