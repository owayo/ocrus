"""ONNX を `.ocnn` に変換する。

旧フォーマットとの違いは 3 つ。

1. **名前と型のあるパラメータ** — 無名の `u32[10]` をやめ、
   op ごとに意味のあるフィールドを持つ
2. **形状計算をグラフから追い出す** — Shape/Gather/Slice/Concat が動的形状のためだけに
   存在していたのをやめ、幅 W の affine 式（`(W*mul + add) / div`）に畳む
3. **ゴールデン出力を埋め込む** — 変換時に onnxruntime で測った argmax 列を記録し、
   実行側が「このファイルは本当にこう計算するのか」を確かめられるようにする

形状は記号推論しない。**複数の幅で実際に onnxruntime を回して実測し、
affine 式に当てはめる**。
当てはまらない次元があれば変換を失敗させる。推測で通してしまうより、止まる方が安い。

使い方:
    uv run --with onnx --with onnxruntime --with numpy python convert_to_ocnn.py \
        ~/.ocrus/models/rec.onnx -o ~/.ocrus/models/rec.ocnn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"OCNN"
HEADER_LEN = 64
ALIGN = 64
VERSION_MAJOR = 3
VERSION_MINOR = 0
CONVERTER = "ocrus convert_to_ocnn 1.0"

#: 形状の当てはめと検証に使う幅。WIDTH_ALIGN=8 の倍数を選ぶ。
DEFAULT_WIDTHS = (64, 96, 160, 320)
TARGET_HEIGHT = 48


class ConversionError(RuntimeError):
    """変換を続行できない。黙って壊れたモデルを出すよりここで止める。"""


# --------------------------------------------------------------------------------------
# 次元の表現
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dim:
    """定数、または幅 W の affine 式 `(W * mul + add) / div`。"""

    const: int | None = None
    sym: int | None = None
    mul: int = 1
    add: int = 0
    div: int = 1

    def to_json(self) -> dict[str, int]:
        """Rust 側の `Dim` と同じ形にする。

        Returns:
            `{"c": n}` または `{"sym": .., "mul": .., "add": .., "div": ..}`。

        """
        if self.const is not None:
            return {"c": int(self.const)}
        return {
            "sym": int(self.sym or 0),
            "mul": int(self.mul),
            "add": int(self.add),
            "div": int(self.div),
        }

    def eval(self, width: int) -> int:
        """幅を与えて具体値にする。

        Args:
            width: 記号 W の値。

        Returns:
            次元の大きさ。

        """
        if self.const is not None:
            return self.const
        return (width * self.mul + self.add) // self.div


def fit_dim(values: dict[int, int]) -> Dim | None:
    """幅ごとの実測値から次元の式を当てはめる。

    Args:
        values: 幅 -> その幅で観測された次元の大きさ。

    Returns:
        当てはまった `Dim`。当てはまらなければ None。

    """
    observed = sorted(values.items())
    firsts = {v for _, v in observed}
    if len(firsts) == 1:
        return Dim(const=observed[0][1])

    # W に比例する形だけを試す。この層構成では stride による割り算しか出てこない。
    for div in (1, 2, 4, 8, 16, 32, 64):
        for mul in (1, 2, 3, 4):
            adds = {v * div - w * mul for w, v in observed}
            if len(adds) != 1:
                continue
            add = adds.pop()
            candidate = Dim(sym=0, mul=mul, add=add, div=div)
            if all(candidate.eval(w) == v for w, v in observed):
                return candidate
    return None


# --------------------------------------------------------------------------------------
# ONNX の読み取り
# --------------------------------------------------------------------------------------


@dataclass
class Graph:
    """変換途中のグラフ。"""

    nodes: list[Any]
    consts: dict[str, np.ndarray]
    input_name: str
    output_name: str
    producer: dict[str, Any] = field(default_factory=dict)
    consumers: dict[str, list[Any]] = field(default_factory=dict)

    def rebuild_index(self) -> None:
        """Rebuild the producer / consumer index."""
        self.producer = {}
        self.consumers = {}
        for node in self.nodes:
            for out in node.output:
                self.producer[out] = node
            for inp in node.input:
                self.consumers.setdefault(inp, []).append(node)


def attr(node: Any, name: str, default: Any = None) -> Any:
    """ONNX ノードの属性を取り出す。

    Args:
        node: 対象ノード。
        name: 属性名。
        default: 見つからないときの値。

    Returns:
        属性値。

    """
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == onnx.AttributeProto.INT:
            return a.i
        if a.type == onnx.AttributeProto.FLOAT:
            return a.f
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        if a.type == onnx.AttributeProto.FLOATS:
            return list(a.floats)
        if a.type == onnx.AttributeProto.STRING:
            return a.s.decode()
        if a.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(a.t)
    return default


def load_graph(path: Path) -> tuple[Any, Graph]:
    """ONNX を読み、Constant ノードを定数表に畳んだグラフを返す。

    Args:
        path: ONNX ファイル。

    Returns:
        (onnx モデル, 変換用グラフ)。

    Raises:
        ConversionError: 入力か出力が 1 つでない。

    """
    model = onnx.load(str(path))
    g = model.graph

    consts: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in g.initializer
    }
    nodes = []
    for node in g.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = np.asarray(attr(node, "value"))
        else:
            nodes.append(node)

    if len(g.input) != 1 or len(g.output) != 1:
        raise ConversionError(
            "入力 1・出力 1 のモデルだけを扱う"
            f"（実際は入力 {len(g.input)} 出力 {len(g.output)}）"
        )

    graph = Graph(
        nodes=nodes,
        consts=consts,
        input_name=g.input[0].name,
        output_name=g.output[0].name,
    )
    graph.rebuild_index()
    return model, graph


# --------------------------------------------------------------------------------------
# onnxruntime による実測
# --------------------------------------------------------------------------------------


def lcg_input(seed: int, width: int) -> np.ndarray:
    """Rust の `golden_input` と同じ擬似乱数入力を作る。

    Args:
        seed: 乱数種。
        width: 画像幅。

    Returns:
        `(1, 3, 48, W)` の float32 配列。

    """
    n = 3 * TARGET_HEIGHT * width
    out = np.empty(n, dtype=np.float32)
    state = seed & 0xFFFFFFFF
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((state >> 8) / 16777216.0 * 2.0 - 1.0)
    return out.reshape(1, 3, TARGET_HEIGHT, width)


def probe_all_values(
    model: Any, widths: tuple[int, ...], seed: int
) -> dict[int, dict[str, np.ndarray]]:
    """全中間出力を露出させた ONNX を各幅で実行し、値を集める。

    形状の当てはめにも、形状計算グラフの畳み込みにも、これ 1 つで足りる。

    Args:
        model: onnx モデル。
        widths: 実行する幅。
        seed: 入力の乱数種。

    Returns:
        幅 -> {値名: 配列}。

    """
    import onnxruntime as ort

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    for node in probe.graph.node:
        for out in node.output:
            if out and out not in existing:
                probe.graph.output.extend([onnx.ValueInfoProto(name=out)])
                existing.add(out)

    sess = ort.InferenceSession(
        probe.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]

    result: dict[int, dict[str, np.ndarray]] = {}
    for width in widths:
        outputs = sess.run(None, {input_name: lcg_input(seed, width)})
        result[width] = dict(zip(names, outputs, strict=True))
    return result


# --------------------------------------------------------------------------------------
# 形状計算グラフの畳み込み
# --------------------------------------------------------------------------------------


def fold_constants(graph: Graph, probes: dict[int, dict[str, np.ndarray]]) -> int:
    """入力が全て定数のノードを、実測値そのものに畳む。

    onnxruntime が既に全中間値を出しているので、計算し直す必要はない。幅を変えても
    同じ値であることを確かめてから定数にするので、動的な値を誤って畳むことはない。

    Args:
        graph: 対象グラフ。書き換える。
        probes: 幅ごとの実測値。

    Returns:
        畳んだノード数。

    """
    widths = sorted(probes)
    removed: set[int] = set()
    folded = 0

    for node in graph.nodes:
        inputs = [i for i in node.input if i]
        if not inputs or not all(i in graph.consts for i in inputs):
            continue
        out = node.output[0]
        if any(out not in probes[w] for w in widths):
            continue
        values = [np.asarray(probes[w][out]) for w in widths]
        first = values[0]
        if not all(
            v.shape == first.shape and np.array_equal(v, first) for v in values[1:]
        ):
            continue
        graph.consts[out] = first
        removed.add(id(node))
        folded += 1

    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        graph.rebuild_index()
    return folded


#: 形状計算に現れる op。Shape から派生した値だけを追う。
SHAPE_OPS = {
    "Shape",
    "Gather",
    "Concat",
    "Unsqueeze",
    "Squeeze",
    "Slice",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
}


def find_shape_values(
    graph: Graph, probes: dict[int, dict[str, np.ndarray]]
) -> dict[str, list[Dim]]:
    """形状計算にしか使われていない値を特定し、次元の式に変換する。

    `Shape` から始まり、整数の小さな 1 次元配列として流れていく値だけを対象にする。

    Args:
        graph: 対象グラフ。
        probes: 幅ごとの実測値。

    Returns:
        値名 -> 次元式の並び。

    """
    widths = sorted(probes)
    shape_like: dict[str, list[Dim]] = {}

    def observed(name: str) -> dict[int, np.ndarray] | None:
        vals = {}
        for w in widths:
            if name not in probes[w]:
                return None
            vals[w] = np.asarray(probes[w][name])
        return vals

    for node in graph.nodes:
        if node.op_type not in SHAPE_OPS:
            continue
        # Shape から派生した値だけを追う
        if node.op_type != "Shape" and not any(
            inp in shape_like or inp in graph.consts for inp in node.input
        ):
            continue
        if node.op_type != "Shape" and not any(inp in shape_like for inp in node.input):
            continue

        out = node.output[0]
        vals = observed(out)
        if vals is None:
            continue
        sample = vals[widths[0]]
        if sample.dtype.kind not in "iu" or sample.ndim > 1 or sample.size > 8:
            continue

        dims = []
        ok = True
        for axis in range(sample.size if sample.ndim else 1):
            per_width = {
                w: int(np.ravel(arr)[axis] if arr.ndim else arr)
                for w, arr in vals.items()
            }
            fitted = fit_dim(per_width)
            if fitted is None:
                ok = False
                break
            dims.append(fitted)
        if ok:
            shape_like[out] = dims

    return shape_like


# --------------------------------------------------------------------------------------
# 融合
# --------------------------------------------------------------------------------------


def fold_batchnorm(graph: Graph) -> int:
    """Conv の直後の BatchNormalization を Conv の重みに畳む。

    Args:
        graph: 対象グラフ。書き換える。

    Returns:
        畳んだ数。

    """
    folded = 0
    removed = set()
    for node in list(graph.nodes):
        if node.op_type != "BatchNormalization":
            continue
        src = graph.producer.get(node.input[0])
        if src is None or src.op_type != "Conv":
            continue
        if len(graph.consumers.get(node.input[0], [])) != 1:
            continue

        scale, bias, mean, var = (graph.consts.get(n) for n in node.input[1:5])
        if any(x is None for x in (scale, bias, mean, var)):
            continue
        eps = float(attr(node, "epsilon", 1e-5))

        w = graph.consts[src.input[1]].astype(np.float32)
        b = (
            graph.consts[src.input[2]].astype(np.float32)
            if len(src.input) > 2 and src.input[2] in graph.consts
            else np.zeros(w.shape[0], dtype=np.float32)
        )
        factor = scale.astype(np.float32) / np.sqrt(var.astype(np.float32) + eps)
        graph.consts[src.input[1]] = w * factor.reshape(-1, 1, 1, 1)
        new_bias = (b - mean.astype(np.float32)) * factor + bias.astype(np.float32)

        bias_name = f"{src.name or src.output[0]}__bias"
        graph.consts[bias_name] = new_bias.astype(np.float32)
        while len(src.input) < 3:
            src.input.append("")
        src.input[2] = bias_name

        # BN の出力を Conv の出力に付け替える
        old_out = node.output[0]
        for consumer in graph.consumers.get(old_out, []):
            for i, name in enumerate(consumer.input):
                if name == old_out:
                    consumer.input[i] = src.output[0]
        if old_out == graph.output_name:
            graph.output_name = src.output[0]
        removed.add(id(node))
        folded += 1

    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        graph.rebuild_index()
    return folded


ACT_OPS = {"Relu": "relu", "Sigmoid": "sigmoid"}


def fold_activation(graph: Graph) -> dict[str, str]:
    """Conv の直後の活性化を Conv に取り込む。

    Args:
        graph: 対象グラフ。書き換える。

    Returns:
        Conv の出力名 -> 活性化の種類。

    """
    acts: dict[str, str] = {}
    removed = set()
    for node in list(graph.nodes):
        if node.op_type not in ACT_OPS:
            continue
        src = graph.producer.get(node.input[0])
        if src is None or src.op_type != "Conv":
            continue
        if len(graph.consumers.get(node.input[0], [])) != 1:
            continue
        # Sigmoid は SE ブロックの門にも使われる。Conv 直後のものだけ畳む。
        acts[src.output[0]] = ACT_OPS[node.op_type]

        old_out = node.output[0]
        for consumer in graph.consumers.get(old_out, []):
            for i, name in enumerate(consumer.input):
                if name == old_out:
                    consumer.input[i] = src.output[0]
        if old_out == graph.output_name:
            graph.output_name = src.output[0]
        removed.add(id(node))

    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        graph.rebuild_index()
    return acts


def fold_layernorm(graph: Graph) -> int:
    """分解された LayerNorm を 1 つの op に畳む。

    `ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add` の並びを探す。

    Args:
        graph: 対象グラフ。書き換える。

    Returns:
        融合した数。

    """
    fused = 0
    removed: set[int] = set()

    def consumer(name: str, op_type: str) -> Any | None:
        """その値を読む、指定 op type のノードを 1 つ返す。

        中間値が複数のノードに読まれるのは普通のこと（LayerNorm では Sub の出力を
        Pow と Div が両方読む）なので、「唯一の消費者」で判定してはいけない。

        Args:
            name: 読まれる値の名前。
            op_type: 探すノードの op type。

        Returns:
            該当が 1 つならそのノード、0 個か 2 個以上なら None。

        """
        cs = [
            c
            for c in graph.consumers.get(name, [])
            if id(c) not in removed and c.op_type == op_type
        ]
        return cs[0] if len(cs) == 1 else None

    for mean1 in list(graph.nodes):
        if mean1.op_type != "ReduceMean" or id(mean1) in removed:
            continue
        x = mean1.input[0]
        sub = consumer(mean1.output[0], "Sub")
        if sub is None or sub.input[0] != x:
            continue
        pow_node = consumer(sub.output[0], "Pow")
        if pow_node is None:
            continue
        mean2 = consumer(pow_node.output[0], "ReduceMean")
        if mean2 is None:
            continue
        add_eps = consumer(mean2.output[0], "Add")
        if add_eps is None:
            continue
        sqrt = consumer(add_eps.output[0], "Sqrt")
        if sqrt is None:
            continue
        div = consumer(sqrt.output[0], "Div")
        if div is None or div.input[1] != sqrt.output[0]:
            continue
        mul = consumer(div.output[0], "Mul")
        if mul is None:
            continue
        add_beta = consumer(mul.output[0], "Add")
        if add_beta is None:
            continue

        eps_name = [n for n in add_eps.input if n != mean2.output[0]]
        gamma_name = [n for n in mul.input if n != div.output[0]]
        beta_name = [n for n in add_beta.input if n != mul.output[0]]
        if not (eps_name and gamma_name and beta_name):
            continue
        if gamma_name[0] not in graph.consts or beta_name[0] not in graph.consts:
            continue
        eps_val = graph.consts.get(eps_name[0])
        if eps_val is None:
            continue

        axes = attr(mean1, "axes", [-1])
        ln = onnx.helper.make_node(
            "OcrusLayerNorm",
            inputs=[x, gamma_name[0], beta_name[0]],
            outputs=[add_beta.output[0]],
            name=f"layernorm_{fused}",
        )
        ln.attribute.extend(
            [
                onnx.helper.make_attribute("axis", int(axes[-1])),
                onnx.helper.make_attribute("epsilon", float(np.ravel(eps_val)[0])),
            ]
        )
        idx = graph.nodes.index(mean1)
        graph.nodes.insert(idx, ln)
        for n in (mean1, sub, pow_node, mean2, add_eps, sqrt, div, mul, add_beta):
            removed.add(id(n))
        fused += 1

    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        graph.rebuild_index()
    return fused


# --------------------------------------------------------------------------------------
# .ocnn の組み立て
# --------------------------------------------------------------------------------------


class Builder:
    """値・テンソル・ノードを積み上げて `.ocnn` を書き出す。"""

    def __init__(self, dtype: str) -> None:
        """空の Builder を作る。

        Args:
            dtype: 重みの保存形式（f32 / f16）。

        """
        self.dtype = dtype
        self.values: list[dict[str, Any]] = []
        self.value_id: dict[str, int] = {}
        self.tensors: list[dict[str, Any]] = []
        self.tensor_id: dict[str, int] = {}
        self.blobs: list[bytes] = []
        self.nodes: list[dict[str, Any]] = []
        self.payload_len = 0

    def value(self, name: str, shape: list[Dim] | None = None) -> int:
        """値 ID を割り当てる（既にあれば再利用）。

        Args:
            name: ONNX の値名。
            shape: 分かっていれば次元式。

        Returns:
            値 ID。

        """
        if name in self.value_id:
            if shape is not None and not self.values[self.value_id[name]]["shape"]:
                self.values[self.value_id[name]]["shape"] = [d.to_json() for d in shape]
            return self.value_id[name]
        vid = len(self.values)
        self.value_id[name] = vid
        self.values.append(
            {"name": name, "shape": [d.to_json() for d in shape] if shape else []}
        )
        return vid

    def tensor(self, name: str, array: np.ndarray) -> int:
        """定数テンソルを登録する（同名は再利用）。

        Args:
            name: テンソル名。
            array: 中身。

        Returns:
            テンソル ID。

        """
        if name in self.tensor_id:
            return self.tensor_id[name]

        arr = np.ascontiguousarray(array)
        if arr.dtype.kind == "f":
            stored = arr.astype(np.float16 if self.dtype == "f16" else np.float32)
            dtype = "f16" if self.dtype == "f16" else "f32"
        else:
            stored = arr.astype(np.float32)
            dtype = "f32"

        pad = (-self.payload_len) % ALIGN
        if pad:
            self.blobs.append(b"\0" * pad)
            self.payload_len += pad
        blob = stored.tobytes()

        tid = len(self.tensors)
        self.tensor_id[name] = tid
        self.tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "layout": "row",
                "shape": [int(d) for d in arr.shape],
                "offset": self.payload_len,
                "len": len(blob),
                "crc32": zlib.crc32(blob),
            }
        )
        self.blobs.append(blob)
        self.payload_len += len(blob)
        return tid

    def node(
        self, name: str, op: dict[str, Any], inputs: list[dict[str, int]], output: int
    ) -> None:
        """ノードを 1 つ積む。

        Args:
            name: 名前（診断用）。
            op: op とそのパラメータ。
            inputs: 入力参照。
            output: 出力値 ID。

        """
        self.nodes.append({"name": name, "inputs": inputs, "output": output, **op})

    def write(self, path: Path, meta_extra: dict[str, Any]) -> None:
        """ファイルに書き出す。

        Args:
            path: 出力先。
            meta_extra: メタデータに足す項目。

        """
        meta = {
            "converter": CONVERTER,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbols": ["W"],
            "values": self.values,
            "tensors": self.tensors,
            "nodes": self.nodes,
            **meta_extra,
        }
        meta_bytes = json.dumps(
            meta, ensure_ascii=False, separators=(",", ":")
        ).encode()
        payload = b"".join(self.blobs)

        meta_offset = HEADER_LEN
        data_offset = meta_offset + len(meta_bytes)
        data_offset += (-data_offset) % ALIGN

        header = bytearray(HEADER_LEN)
        header[0:4] = MAGIC
        # 版番号はヘッダに持つ。ファイル名には出さない。
        struct.pack_into("<II", header, 4, VERSION_MAJOR, VERSION_MINOR)
        struct.pack_into(
            "<QQQQ", header, 16, meta_offset, len(meta_bytes), data_offset, len(payload)
        )
        struct.pack_into("<II", header, 48, zlib.crc32(meta_bytes), zlib.crc32(payload))

        with path.open("wb") as f:
            f.write(header)
            f.write(meta_bytes)
            f.write(b"\0" * (data_offset - meta_offset - len(meta_bytes)))
            f.write(payload)


def dims_of(name: str, probes: dict[int, dict[str, np.ndarray]]) -> list[Dim] | None:
    """実測から値の形状式を当てはめる。

    Args:
        name: 値名。
        probes: 幅ごとの実測値。

    Returns:
        次元式の並び。当てはまらなければ None。

    """
    widths = sorted(probes)
    if any(name not in probes[w] for w in widths):
        return None
    rank = probes[widths[0]][name].ndim
    dims = []
    for axis in range(rank):
        fitted = fit_dim({w: int(probes[w][name].shape[axis]) for w in widths})
        if fitted is None:
            return None
        dims.append(fitted)
    return dims


# --------------------------------------------------------------------------------------
# ノードの出力
# --------------------------------------------------------------------------------------

BINARY = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow"}
UNARY = {"Relu": "relu", "Sigmoid": "sigmoid", "Sqrt": "sqrt"}


def pads_of(
    node: Any, probes: dict[int, dict[str, np.ndarray]] | None = None
) -> list[int]:
    """ONNX の pads を `[top, left, bottom, right]` にする。

    `auto_pad` が指定されている場合は、実測した入出力の形から必要なパディングを逆算し、
    どの幅でも同じ値になることを確かめたうえで明示値に置き換える。幅によって変わるなら
    式で表せないので変換を失敗させる。

    Args:
        node: Conv / Pool ノード。
        probes: 幅ごとの実測値。auto_pad の解決に要る。

    Returns:
        4 要素のパディング。

    Raises:
        ConversionError: auto_pad を静的に解決できない。

    """
    auto = attr(node, "auto_pad", "NOTSET")
    if auto in ("NOTSET", "VALID"):
        pads = attr(node, "pads", [0, 0, 0, 0])
        return [int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])]
    if auto not in ("SAME_UPPER", "SAME_LOWER") or probes is None:
        raise ConversionError(f"auto_pad={auto!r} は未対応")

    name = node.name or node.output[0]
    kernel = attr(node, "kernel_shape")
    if kernel is None:
        raise ConversionError(f"{name}: auto_pad の解決に kernel_shape が要る")
    strides = attr(node, "strides", [1, 1])

    resolved: set[tuple[int, int, int, int]] = set()
    for w in sorted(probes):
        src, dst = node.input[0], node.output[0]
        if src not in probes[w] or dst not in probes[w]:
            raise ConversionError(f"{name}: auto_pad の解決に必要な実測値が無い")
        in_shape = probes[w][src].shape
        out_shape = probes[w][dst].shape
        pads = []
        for axis in (0, 1):
            total = max(
                0,
                (out_shape[2 + axis] - 1) * int(strides[axis])
                + int(kernel[axis])
                - in_shape[2 + axis],
            )
            begin = total // 2 if auto == "SAME_UPPER" else total - total // 2
            pads.append((begin, total - begin))
        resolved.add((pads[0][0], pads[1][0], pads[0][1], pads[1][1]))

    if len(resolved) != 1:
        raise ConversionError(
            f"{name}: auto_pad={auto} のパディングが幅で変わる（{sorted(resolved)}）"
        )
    return list(resolved.pop())


def build_model(
    graph: Graph,
    probes: dict[int, dict[str, np.ndarray]],
    shape_values: dict[str, list[Dim]],
    acts: dict[str, str],
    dtype: str,
) -> Builder:
    """グラフを `.ocnn` の値・テンソル・ノードに変換する。

    Args:
        graph: 融合済みグラフ。
        probes: 幅ごとの実測値。
        shape_values: 形状計算に畳んだ値。
        acts: Conv 出力名 -> 融合した活性化。
        dtype: 重みの保存形式。

    Returns:
        組み立て済み Builder。

    Raises:
        ConversionError: 未対応の op や解決できない形状があった。

    """
    b = Builder(dtype)
    b.value(
        graph.input_name,
        [Dim(const=1), Dim(const=3), Dim(const=TARGET_HEIGHT), Dim(sym=0)],
    )

    def ref(name: str) -> dict[str, int]:
        if name in graph.consts:
            return {"t": b.tensor(name, graph.consts[name])}
        if name not in b.value_id:
            raise ConversionError(
                f"値 {name} が未定義のまま参照された（トポロジカル順の乱れ）"
            )
        return {"v": b.value_id[name]}

    def out_value(node: Any) -> int:
        return b.value(node.output[0], dims_of(node.output[0], probes) or None)

    def slice_scalar(node: Any, idx: int, default: int | None = None) -> Dim:
        name = node.name or node.output[0]
        if len(node.input) <= idx or not node.input[idx]:
            if default is None:
                raise ConversionError(f"Slice {name} の引数 {idx} が無い")
            return Dim(const=default)
        src = node.input[idx]
        if src in shape_values:
            return shape_values[src][0]
        if src in graph.consts:
            return Dim(const=int(np.ravel(graph.consts[src])[0]))
        raise ConversionError(f"Slice {name} の引数 {src} を静的に解決できなかった")

    for node in graph.nodes:
        op_type = node.op_type
        name = node.name or node.output[0]

        # 形状計算に畳んだノードは実行グラフから消える
        if node.output[0] in shape_values:
            continue

        if op_type == "Conv":
            strides = attr(node, "strides", [1, 1])
            dilations = attr(node, "dilations", [1, 1])
            inputs = [ref(node.input[0]), ref(node.input[1])]
            if len(node.input) > 2 and node.input[2]:
                inputs.append(ref(node.input[2]))
            b.node(
                name,
                {
                    "op": "conv2d",
                    "stride": [int(strides[0]), int(strides[1])],
                    "pad": pads_of(node, probes),
                    "dilation": [int(dilations[0]), int(dilations[1])],
                    "groups": int(attr(node, "group", 1)),
                    "act": acts.get(node.output[0], "none"),
                },
                inputs,
                out_value(node),
            )
        elif op_type in ("MaxPool", "AveragePool"):
            kernel = attr(node, "kernel_shape", [1, 1])
            strides = attr(node, "strides", [1, 1])
            b.node(
                name,
                {
                    "op": "pool",
                    "kind": "max" if op_type == "MaxPool" else "avg",
                    "kernel": [int(kernel[0]), int(kernel[1])],
                    "stride": [int(strides[0]), int(strides[1])],
                    "pad": pads_of(node, probes),
                    "global": False,
                },
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type == "MatMul":
            b.node(
                name,
                {"op": "mat_mul", "trans_b": False},
                [ref(node.input[0]), ref(node.input[1])],
                out_value(node),
            )
        elif op_type == "OcrusLayerNorm":
            b.node(
                name,
                {
                    "op": "layer_norm",
                    "axis": int(attr(node, "axis", -1)),
                    "eps": float(attr(node, "epsilon", 1e-5)),
                },
                [ref(node.input[0]), ref(node.input[1]), ref(node.input[2])],
                out_value(node),
            )
        elif op_type == "Softmax":
            b.node(
                name,
                {"op": "softmax", "axis": int(attr(node, "axis", -1))},
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type == "Reshape":
            target = node.input[1]
            if target in shape_values:
                shape = shape_values[target]
            elif target in graph.consts:
                shape = [Dim(const=int(v)) for v in np.ravel(graph.consts[target])]
            else:
                raise ConversionError(
                    f"Reshape {name} の形状入力 {target} を静的に解決できなかった"
                )
            # ONNX では 0 は「入力の同じ軸をそのまま使う」。フォーマットにこの癖を
            # 持ち込まず、ここで具体的な次元に解決しておく。
            if any(d.const == 0 for d in shape):
                src_dims = dims_of(node.input[0], probes)
                if src_dims is None:
                    raise ConversionError(
                        f"Reshape {name}: 0 次元の解決に必要な入力形状が分からない"
                    )
                shape = [
                    src_dims[i] if d.const == 0 else d for i, d in enumerate(shape)
                ]
            b.node(
                name,
                {"op": "reshape", "shape": [d.to_json() for d in shape]},
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type == "Transpose":
            perm = attr(node, "perm")
            if perm is None:
                raise ConversionError(f"Transpose {name} に perm が無い")
            b.node(
                name,
                {"op": "transpose", "perm": [int(p) for p in perm]},
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type == "Concat":
            b.node(
                name,
                {"op": "concat", "axis": int(attr(node, "axis", 0))},
                [ref(i) for i in node.input],
                out_value(node),
            )
        elif op_type == "Slice":
            axes_src = node.input[3] if len(node.input) > 3 and node.input[3] else None
            axis = (
                int(np.ravel(graph.consts[axes_src])[0])
                if axes_src and axes_src in graph.consts
                else 0
            )
            step = slice_scalar(node, 4, 1)
            if step.const is None:
                raise ConversionError(f"Slice {name} の step が動的")
            b.node(
                name,
                {
                    "op": "slice",
                    "axis": axis,
                    "start": slice_scalar(node, 1).to_json(),
                    "end": slice_scalar(node, 2).to_json(),
                    "step": int(step.const),
                },
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type == "ReduceMean":
            b.node(
                name,
                {
                    "op": "reduce_mean",
                    "axes": [int(a) for a in attr(node, "axes", [-1])],
                    "keepdims": bool(attr(node, "keepdims", 1)),
                },
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type in BINARY:
            b.node(
                name,
                {"op": "binary", "kind": BINARY[op_type]},
                [ref(node.input[0]), ref(node.input[1])],
                out_value(node),
            )
        elif op_type in UNARY:
            b.node(
                name,
                {"op": "unary", "kind": UNARY[op_type]},
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type in ("Squeeze", "Unsqueeze"):
            axes = attr(node, "axes")
            if axes is None and len(node.input) > 1 and node.input[1] in graph.consts:
                axes = [int(v) for v in np.ravel(graph.consts[node.input[1]])]
            if axes is None:
                raise ConversionError(f"{op_type} {name} の axes を解決できなかった")
            b.node(
                name,
                {"op": op_type.lower(), "axes": [int(a) for a in axes]},
                [ref(node.input[0])],
                out_value(node),
            )
        elif op_type == "Identity":
            b.node(name, {"op": "identity"}, [ref(node.input[0])], out_value(node))
        else:
            raise ConversionError(
                f"未対応の op: {op_type} ({name})。実行系に追加するか、変換時に畳むこと"
            )

    return b


def golden_records(
    probes: dict[int, dict[str, np.ndarray]],
    output_name: str,
    widths: tuple[int, ...],
    seed: int,
) -> list[dict[str, Any]]:
    """実測した出力から、検証用の argmax 列を記録する。

    実測は既に onnxruntime で取ってあるので、ここで測り直さない。
    融合でグラフを書き換える前の出力名を使うこと
    （融合は proto を書き換えるので、後から流し直せない）。

    Args:
        probes: 幅ごとの実測値。
        output_name: 元グラフの出力名。
        widths: 記録する幅。
        seed: 入力に使った乱数種。

    Returns:
        ゴールデンレコードの並び。

    Raises:
        ConversionError: 実測に無い幅を指定した。

    """
    records = []
    for width in widths:
        if width not in probes or output_name not in probes[width]:
            raise ConversionError(f"幅 {width} の実測が無い。--widths に含めること")
        out = probes[width][output_name]
        records.append(
            {
                "seed": seed,
                "width": width,
                "out_shape": [int(d) for d in out.shape],
                "argmax": [int(i) for i in out[0].argmax(axis=-1)],
            }
        )
    return records


def main() -> int:
    """コマンドラインから変換を実行する。

    Returns:
        終了コード。

    """
    parser = argparse.ArgumentParser(description="ONNX を .ocnn に変換する")
    parser.add_argument("input", help="入力の ONNX")
    parser.add_argument("-o", "--output", required=True, help="出力の .ocnn")
    parser.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f16",
        help="重みの保存形式（既定 f16: サイズ半分、精度は同じ、キャッシュ効率で速い）",
    )
    parser.add_argument(
        "--widths",
        default=",".join(str(w) for w in DEFAULT_WIDTHS),
        help="形状の当てはめと検証に使う幅",
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="ゴールデン入力の乱数種"
    )
    parser.add_argument(
        "--golden-widths", default="64,160", help="ゴールデンを記録する幅"
    )
    args = parser.parse_args()

    widths = tuple(int(w) for w in args.widths.split(","))
    onnx_path = Path(args.input).expanduser()

    print(f"読み込み: {onnx_path}")
    model, graph = load_graph(onnx_path)
    # 融合はこの proto を書き換えるので、元の出力名をいま控える
    original_output = graph.output_name
    print(f"  ノード {len(graph.nodes)} / 定数 {len(graph.consts)}")

    print(f"onnxruntime で実測（幅 {widths}）...")
    probes = probe_all_values(model, widths, args.seed)

    folded = fold_constants(graph, probes)
    print(f"  定数に畳んだノード: {folded}")

    shape_values = find_shape_values(graph, probes)
    print(f"  形状計算に畳んだ値: {len(shape_values)}")

    bn = fold_batchnorm(graph)
    acts = fold_activation(graph)
    ln = fold_layernorm(graph)
    print(f"  融合: BatchNorm {bn} / 活性化 {len(acts)} / LayerNorm {ln}")
    print(f"  融合後のノード: {len(graph.nodes)}")

    builder = build_model(graph, probes, shape_values, acts, args.dtype)
    print(f"  出力ノード {len(builder.nodes)} / テンソル {len(builder.tensors)}")

    golden_widths = tuple(int(w) for w in args.golden_widths.split(","))
    golden = golden_records(probes, original_output, golden_widths, args.seed)
    print(f"  ゴールデン {len(golden)} 件（幅 {golden_widths}）")

    out_path = Path(args.output).expanduser()
    builder.write(
        out_path,
        {
            "source": {
                "file": onnx_path.name,
                "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                "opset": int(model.opset_import[0].version),
            },
            "inputs": [builder.value_id[graph.input_name]],
            "outputs": [builder.value_id[graph.output_name]],
            "golden": golden,
        },
    )
    size = out_path.stat().st_size
    print(f"書き出し: {out_path}  {size / 1048576:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
