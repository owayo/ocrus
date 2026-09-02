//! Executor for `.ocnn` graphs.
//!
//! Nodes are stored in topological order, so execution is a single pass over them:
//! resolve inputs, run one typed op, store the result. Registers are dropped as soon as
//! the last node that reads them has run, which keeps peak memory close to the widest
//! point of the graph rather than the sum of every intermediate.

use std::sync::OnceLock;

use ocrus_core::OcrusError;
use ocrus_core::error::Result;

use crate::ocnn::format::{Act, BinKind, Dim, InputRef, Model, NodeDesc, Op, PoolKind, UnaryKind};
use crate::ops;
use crate::tensor::NdTensor;

/// Per-node timings from [`Executor::run_profiled`]: the op name and how long it took.
pub type NodeTimings = Vec<(String, std::time::Duration)>;

/// A model plus its materialized constants, ready to run.
///
/// Constants are turned into tensors on first use and kept, so running many images through
/// one executor pays that cost once. (Removing the copy entirely needs kernels that accept
/// a slice plus a shape; that comes with the kernel-layout work.)
pub struct Executor {
    model: Model,
    consts: Vec<OnceLock<NdTensor<f32>>>,
}

impl Executor {
    /// Wrap a loaded model.
    pub fn new(model: Model) -> Self {
        let consts = (0..model.meta.tensors.len())
            .map(|_| OnceLock::new())
            .collect();
        Self { model, consts }
    }

    /// The underlying model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Materialize a constant tensor, reusing the cached copy after the first call.
    fn constant(&self, id: u32) -> Result<&NdTensor<f32>> {
        let slot = self
            .consts
            .get(id as usize)
            .ok_or_else(|| model_err(format!("unknown tensor {id}")))?;
        if let Some(t) = slot.get() {
            return Ok(t);
        }
        let desc = &self.model.meta.tensors[id as usize];
        let data = self.model.tensor_f32(id)?.into_owned();
        let shape: Vec<usize> = desc.shape.iter().map(|&d| d.max(0) as usize).collect();
        let tensor = NdTensor::from_vec(data, &shape);
        Ok(slot.get_or_init(|| tensor))
    }

    /// Run the graph, reporting how long each node took.
    ///
    /// Optimizing without measuring is guesswork: this is what tells you that the time is
    /// in convolution rather than in, say, the transpose you were about to rewrite.
    ///
    /// Args:
    ///     input: the graph input.
    ///
    /// Returns:
    ///     The output and one `(op, duration)` entry per node, in execution order.
    pub fn run_profiled(&self, input: NdTensor<f32>) -> Result<(NdTensor<f32>, NodeTimings)> {
        let mut timings = Vec::with_capacity(self.model.meta.nodes.len());
        let out = self.run_inner(input, Some(&mut timings))?;
        Ok((out, timings))
    }

    /// Run the graph.
    ///
    /// Args:
    ///     input: the graph input, shaped `(1, 3, 48, W)`.
    ///
    /// Returns:
    ///     The graph output.
    pub fn run(&self, input: NdTensor<f32>) -> Result<NdTensor<f32>> {
        self.run_inner(input, None)
    }

    fn run_inner(
        &self,
        input: NdTensor<f32>,
        mut timings: Option<&mut NodeTimings>,
    ) -> Result<NdTensor<f32>> {
        let meta = &self.model.meta;

        // The only dynamic symbol is the input width; everything else is derived from it.
        let symbols = self.resolve_symbols(&input)?;

        let mut registers: Vec<Option<NdTensor<f32>>> = vec![None; meta.values.len()];
        registers[meta.inputs[0] as usize] = Some(input);

        for (idx, node) in meta.nodes.iter().enumerate() {
            let started = timings.as_ref().map(|_| std::time::Instant::now());
            let out = self
                .run_node(node, &registers, &symbols)
                .map_err(|e| annotate(e, idx, node))?;
            if let (Some(t), Some(start)) = (timings.as_mut(), started) {
                t.push((op_name(&node.op).to_string(), start.elapsed()));
            }
            registers[node.output as usize] = Some(out);

            // Free everything whose last reader was this node.
            for (value_id, reg) in registers.iter_mut().enumerate() {
                if reg.is_some() && self.model.last_use[value_id] == idx {
                    *reg = None;
                }
            }
        }

        registers[meta.outputs[0] as usize]
            .take()
            .ok_or_else(|| model_err("graph produced no output"))
    }

    /// Bind the dynamic symbols from the concrete input shape.
    fn resolve_symbols(&self, input: &NdTensor<f32>) -> Result<Vec<i64>> {
        let meta = &self.model.meta;
        let declared = &meta.values[meta.inputs[0] as usize].shape;
        let mut symbols = vec![0i64; meta.symbols.len()];

        for (axis, dim) in declared.iter().enumerate() {
            let actual = *input
                .shape
                .get(axis)
                .ok_or_else(|| model_err("input has fewer axes than the graph declares"))?
                as i64;
            match *dim {
                Dim::Const { c } => {
                    if c != actual {
                        return Err(model_err(format!(
                            "input axis {axis} is {actual} but the model requires {c}"
                        )));
                    }
                }
                Dim::Expr { sym, mul, add, div } => {
                    // Only the plain `sym` form can be inverted, which is all the input needs.
                    if mul != 1 || add != 0 || div != 1 {
                        return Err(model_err(
                            "graph input dimensions must be plain symbols, not expressions",
                        ));
                    }
                    symbols[sym as usize] = actual;
                }
            }
        }
        Ok(symbols)
    }

    fn input_tensor<'a>(
        &'a self,
        node: &NodeDesc,
        slot: usize,
        registers: &'a [Option<NdTensor<f32>>],
    ) -> Result<&'a NdTensor<f32>> {
        let r = node
            .inputs
            .get(slot)
            .ok_or_else(|| model_err(format!("op needs at least {} inputs", slot + 1)))?;
        match *r {
            InputRef::Value { v } => registers[v as usize]
                .as_ref()
                .ok_or_else(|| model_err(format!("value {v} was already released"))),
            InputRef::Tensor { t } => self.constant(t),
        }
    }

    fn optional_input<'a>(
        &'a self,
        node: &NodeDesc,
        slot: usize,
        registers: &'a [Option<NdTensor<f32>>],
    ) -> Option<&'a NdTensor<f32>> {
        if node.inputs.len() <= slot {
            return None;
        }
        self.input_tensor(node, slot, registers).ok()
    }

    fn run_node(
        &self,
        node: &NodeDesc,
        registers: &[Option<NdTensor<f32>>],
        symbols: &[i64],
    ) -> Result<NdTensor<f32>> {
        let a = |slot: usize| self.input_tensor(node, slot, registers);

        Ok(match &node.op {
            Op::Conv2d {
                stride,
                pad,
                dilation,
                groups,
                act,
            } => {
                if dilation != &[1, 1] {
                    return Err(model_err("dilated convolution is not implemented"));
                }
                let input = a(0)?;
                let weight = a(1)?;
                let bias = self.optional_input(node, 2, registers);
                let mut out = conv(input, weight, bias, *stride, *pad, *groups)?;
                apply_act(&mut out, *act);
                out
            }
            Op::Pool {
                kind,
                kernel,
                stride,
                pad,
                global,
            } => {
                let input = a(0)?;
                let (kh, kw) = if *global {
                    (input.shape[2], input.shape[3])
                } else {
                    (kernel[0], kernel[1])
                };
                let padded = pad_if_asymmetric(input, *pad);
                let src = padded.as_ref().unwrap_or(input);
                let (ph, pw) = if padded.is_some() {
                    (0, 0)
                } else {
                    (pad[0], pad[1])
                };
                match kind {
                    PoolKind::Max => {
                        ops::pool::max_pool2d(src, kh, kw, stride[0], stride[1], ph, pw)
                    }
                    PoolKind::Avg => {
                        ops::pool::avg_pool2d(src, kh, kw, stride[0], stride[1], ph, pw)
                    }
                }
            }
            Op::MatMul { trans_b } => {
                let lhs = a(0)?;
                let rhs = a(1)?;
                if *trans_b {
                    let n = rhs.ndim();
                    let t = rhs.transpose(n - 2, n - 1);
                    ops::matmul::matmul(lhs, &t)
                } else {
                    ops::matmul::matmul(lhs, rhs)
                }
            }
            Op::LayerNorm { axis, eps } => {
                let input = a(0)?;
                let gamma = a(1)?;
                let beta = a(2)?;
                let axis = norm_axis(*axis, input.ndim())?;
                ops::layernorm::layer_norm(input, gamma, beta, axis, *eps)
            }
            Op::Softmax { axis } => {
                let input = a(0)?;
                let axis = norm_axis(*axis, input.ndim())?;
                ops::activation::softmax(input, axis)
            }
            Op::Reshape { shape } => {
                let mut t = a(0)?.clone();
                let dims = eval_shape(shape, symbols, t.len())?;
                t.reshape(&dims);
                t
            }
            Op::Transpose { perm } => a(0)?.transpose_perm(perm),
            Op::Concat { axis } => {
                let mut parts = Vec::with_capacity(node.inputs.len());
                for slot in 0..node.inputs.len() {
                    parts.push(self.input_tensor(node, slot, registers)?);
                }
                let axis = norm_axis(*axis, parts[0].ndim())?;
                ops::tensor_ops::concat(&parts, axis)
            }
            Op::Slice {
                axis,
                start,
                end,
                step,
            } => {
                let input = a(0)?;
                let axis = norm_axis(*axis, input.ndim())?;
                ops::tensor_ops::slice_tensor(
                    input,
                    axis,
                    start.eval(symbols)?,
                    end.eval(symbols)?,
                    *step,
                )
            }
            Op::ReduceMean { axes, keepdims } => {
                let input = a(0)?;
                let axes: Result<Vec<usize>> =
                    axes.iter().map(|&x| norm_axis(x, input.ndim())).collect();
                ops::reduce::reduce_mean(input, &axes?, *keepdims)
            }
            Op::Binary { kind } => {
                let lhs = a(0)?;
                let rhs = a(1)?;
                match kind {
                    BinKind::Add => ops::binary::add(lhs, rhs),
                    BinKind::Sub => ops::binary::sub(lhs, rhs),
                    BinKind::Mul => ops::binary::mul(lhs, rhs),
                    BinKind::Div => ops::binary::div(lhs, rhs),
                    BinKind::Pow => ops::math::pow_tensor(lhs, rhs),
                }
            }
            Op::Unary { kind } => {
                let mut t = a(0)?.clone();
                match kind {
                    UnaryKind::Relu => ops::relu::relu_inplace(&mut t),
                    UnaryKind::HardSwish => ops::relu::hard_swish_inplace(&mut t),
                    UnaryKind::Sigmoid => ops::activation::sigmoid_inplace(&mut t),
                    UnaryKind::Sqrt => ops::math::sqrt_inplace(&mut t),
                }
                t
            }
            Op::Gather { axis } => {
                let input = a(0)?;
                let indices = a(1)?;
                let axis = norm_axis(*axis, input.ndim())?;
                ops::gather::gather(input, indices, axis)
            }
            Op::Squeeze { axes } => {
                let mut t = a(0)?.clone();
                let axes: Result<Vec<usize>> =
                    axes.iter().map(|&x| norm_axis(x, t.ndim())).collect();
                ops::tensor_ops::squeeze(&mut t, &axes?);
                t
            }
            Op::Unsqueeze { axes } => {
                let mut t = a(0)?.clone();
                let rank = t.ndim() + axes.len();
                let axes: Result<Vec<usize>> = axes.iter().map(|&x| norm_axis(x, rank)).collect();
                ops::tensor_ops::unsqueeze(&mut t, &axes?);
                t
            }
            Op::Identity => a(0)?.clone(),
        })
    }
}

/// Dispatch to the convolution kernel that fits, pre-padding when the padding is uneven.
fn conv(
    input: &NdTensor<f32>,
    weight: &NdTensor<f32>,
    bias: Option<&NdTensor<f32>>,
    stride: [usize; 2],
    pad: [usize; 4],
    groups: usize,
) -> Result<NdTensor<f32>> {
    let padded = pad_if_asymmetric(input, pad);
    let src = padded.as_ref().unwrap_or(input);
    let (ph, pw) = if padded.is_some() {
        (0, 0)
    } else {
        (pad[0], pad[1])
    };

    let in_ch = input.shape[1];
    let (kh, kw) = (weight.shape[2], weight.shape[3]);

    Ok(if groups > 1 {
        if groups != in_ch {
            return Err(model_err(format!(
                "grouped convolution with groups={groups} (in_channels={in_ch}) is not implemented"
            )));
        }
        ops::conv2d::conv2d_depthwise(src, weight, bias, stride[0], stride[1], ph, pw)
    } else if kh == 1 && kw == 1 && stride == [1, 1] && ph == 0 && pw == 0 {
        ops::conv2d::conv2d_pointwise(src, weight, bias)
    } else {
        ops::conv2d::conv2d_general(src, weight, bias, stride[0], stride[1], ph, pw)
    })
}

/// Pad spatially when top/bottom or left/right differ, since the kernels take one pad per
/// axis. `pad` is `[top, left, bottom, right]`.
fn pad_if_asymmetric(input: &NdTensor<f32>, pad: [usize; 4]) -> Option<NdTensor<f32>> {
    let [top, left, bottom, right] = pad;
    if top == bottom && left == right {
        return None;
    }
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (nh, nw) = (h + top + bottom, w + left + right);
    let mut out = NdTensor::zeros(&[n, c, nh, nw]);
    for b in 0..n {
        for ch in 0..c {
            let src = (b * c + ch) * h * w;
            let dst = (b * c + ch) * nh * nw;
            for row in 0..h {
                let s = src + row * w;
                let d = dst + (row + top) * nw + left;
                out.data[d..d + w].copy_from_slice(&input.data[s..s + w]);
            }
        }
    }
    Some(out)
}

fn apply_act(t: &mut NdTensor<f32>, act: Act) {
    match act {
        Act::None => {}
        Act::Relu => ops::relu::relu_inplace(t),
        Act::HardSwish => ops::relu::hard_swish_inplace(t),
        Act::Sigmoid => ops::activation::sigmoid_inplace(t),
    }
}

/// Resolve a possibly-negative axis against a rank.
fn norm_axis(axis: i32, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i32 } else { axis };
    if a < 0 || a as usize >= rank.max(1) {
        return Err(model_err(format!(
            "axis {axis} is out of range for rank {rank}"
        )));
    }
    Ok(a as usize)
}

/// Evaluate a reshape target, resolving symbols and the single optional `-1`.
fn eval_shape(shape: &[Dim], symbols: &[i64], total: usize) -> Result<Vec<usize>> {
    let raw: Result<Vec<i64>> = shape.iter().map(|d| d.eval(symbols)).collect();
    let raw = raw?;

    let unknown = raw.iter().filter(|&&d| d < 0).count();
    if unknown > 1 {
        return Err(model_err(
            "reshape has more than one inferred (-1) dimension",
        ));
    }
    let known: i64 = raw.iter().filter(|&&d| d > 0).product();
    if known == 0 {
        return Err(model_err("reshape target has a zero dimension"));
    }

    let dims: Vec<usize> = raw
        .iter()
        .map(|&d| {
            if d < 0 {
                total / known as usize
            } else {
                d as usize
            }
        })
        .collect();

    let product: usize = dims.iter().product();
    if product != total {
        return Err(model_err(format!(
            "reshape to {dims:?} does not match the {total} elements of the input"
        )));
    }
    Ok(dims)
}

fn annotate(err: OcrusError, idx: usize, node: &NodeDesc) -> OcrusError {
    OcrusError::Model(format!(
        "node {idx} ({}, {}): {err}",
        if node.name.is_empty() {
            "unnamed"
        } else {
            &node.name
        },
        op_name(&node.op),
    ))
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Conv2d { .. } => "conv2d",
        Op::Pool { .. } => "pool",
        Op::MatMul { .. } => "matmul",
        Op::LayerNorm { .. } => "layer_norm",
        Op::Softmax { .. } => "softmax",
        Op::Reshape { .. } => "reshape",
        Op::Transpose { .. } => "transpose",
        Op::Concat { .. } => "concat",
        Op::Slice { .. } => "slice",
        Op::ReduceMean { .. } => "reduce_mean",
        Op::Binary { .. } => "binary",
        Op::Unary { .. } => "unary",
        Op::Gather { .. } => "gather",
        Op::Squeeze { .. } => "squeeze",
        Op::Unsqueeze { .. } => "unsqueeze",
        Op::Identity => "identity",
    }
}

fn model_err(msg: impl Into<String>) -> OcrusError {
    OcrusError::Model(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_infers_the_minus_one_dimension() {
        let dims = eval_shape(
            &[
                Dim::Const { c: 1 },
                Dim::Const { c: -1 },
                Dim::Const { c: 4 },
            ],
            &[],
            24,
        )
        .unwrap();
        assert_eq!(dims, vec![1, 6, 4]);
    }

    #[test]
    fn reshape_rejects_a_mismatched_target() {
        let err = eval_shape(&[Dim::Const { c: 5 }, Dim::Const { c: 5 }], &[], 24).unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn axes_accept_negative_indices() {
        assert_eq!(norm_axis(-1, 3).unwrap(), 2);
        assert_eq!(norm_axis(0, 3).unwrap(), 0);
        assert!(norm_axis(3, 3).is_err());
    }
}
