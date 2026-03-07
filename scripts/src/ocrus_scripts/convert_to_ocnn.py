#!/usr/bin/env python3
"""Convert ONNX model to .ocnn v2 format for ocrus-nn inference engine."""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from typing import Any

import numpy as np

MAGIC = b"OCNN"
VERSION = 2

# Layer type enum (must match Rust LayerType)
LAYER_TYPES = {
    "Conv": 1,
    "ConvDepthwise": 2,
    "BatchNormalization": 3,
    "Relu": 4,
    "HardSwish": 5,
    "MaxPool": 6,
    "AveragePool": 7,
    "Gemm": 8,
    "Reshape": 9,
    "Flatten": 10,
    "Transpose": 11,
    "Add": 12,
    "Mul": 13,
    "Sub": 14,
    "Div": 15,
    "MatMul": 16,
    "Sigmoid": 17,
    "Softmax": 18,
    "Concat": 19,
    "Slice": 20,
    "Squeeze": 21,
    "Unsqueeze": 22,
    "ReduceMean": 24,
    "Pow": 25,
    "Sqrt": 26,
    "LayerNorm": 27,
    "Gather": 29,
}

LAYER_DESCRIPTOR_SIZE = 80
CONSTANT_ENTRY_SIZE = 32
HEADER_SIZE = 16

# Ops whose weights are packed inline (not via Constant table)
INLINE_WEIGHT_OPS = {"Conv", "ConvDepthwise", "BatchNormalization", "Gemm"}


def convert_onnx_to_ocnn(onnx_path: str, output_path: str):
    """Convert ONNX model to .ocnn v2 binary format."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("Error: onnx package required. Install with: pip install onnx")
        sys.exit(1)

    model = onnx.load(onnx_path)
    graph = model.graph

    # Build initializer lookup
    initializers: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    # Extract Constant op values
    constant_op_values: dict[str, np.ndarray] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            val = _extract_constant_value(node)
            if val is not None and len(node.output) > 0:
                constant_op_values[node.output[0]] = val

    # Build DAG: tensor_name -> producing node index (excluding Constant/Shape ops)
    # Also track Shape op outputs for dynamic shape resolution
    shape_outputs: set[str] = set()
    non_compute_nodes: list[str] = []
    compute_nodes: list[Any] = []

    for node in graph.node:
        if node.op_type == "Constant":
            non_compute_nodes.append(node.op_type)
            continue
        if node.op_type == "Shape":
            for out in node.output:
                shape_outputs.add(out)
            non_compute_nodes.append(node.op_type)
            continue
        compute_nodes.append(node)

    # Topological order is preserved from ONNX graph
    # Build output_name -> layer_index mapping
    output_to_layer: dict[str, int] = {}
    for idx, node in enumerate(compute_nodes):
        for out in node.output:
            output_to_layer[out] = idx

    # Collect all constant data (initializers + Constant ops not used as inline weights)
    # We need to identify which initializers are used as inline weights
    inline_weight_tensors: set[str] = set()
    for node in compute_nodes:
        if node.op_type in ("Conv", "ConvDepthwise"):
            # weight and bias are inline
            if len(node.input) > 1:
                inline_weight_tensors.add(node.input[1])
            if len(node.input) > 2 and node.input[2]:
                inline_weight_tensors.add(node.input[2])
        elif node.op_type == "BatchNormalization":
            for i in range(1, 5):
                if i < len(node.input):
                    inline_weight_tensors.add(node.input[i])
        elif node.op_type == "Gemm":
            if len(node.input) > 1:
                inline_weight_tensors.add(node.input[1])
            if len(node.input) > 2 and node.input[2]:
                inline_weight_tensors.add(node.input[2])

    # Build constant table: tensor_name -> constant_index
    constant_table: dict[str, int] = {}
    constant_entries: list[dict] = []  # {shape, data_bytes}
    constant_data_chunks: list[bytes] = []
    constant_data_offset = 0

    def _add_constant(name: str, arr: np.ndarray) -> int:
        if name in constant_table:
            return constant_table[name]
        data = arr.astype(np.float32).tobytes()
        shape = list(arr.shape)
        entry = {
            "data_offset": constant_data_offset,
            "data_size": len(data),
            "ndim": len(shape),
            "shape": shape,
        }
        idx = len(constant_entries)
        constant_entries.append(entry)
        constant_data_chunks.append(data)
        constant_table[name] = idx
        return idx

    # Pre-scan: register constants needed by compute nodes
    for node in compute_nodes:
        for inp_name in node.input:
            if not inp_name:
                continue
            if inp_name in inline_weight_tensors:
                continue
            if inp_name in output_to_layer:
                continue
            # Check if it's a graph input (not initializer, not constant op)
            is_graph_input = any(gi.name == inp_name for gi in graph.input)
            if is_graph_input and inp_name not in initializers:
                continue
            # It's a constant (initializer or Constant op value)
            arr = None
            if inp_name in constant_op_values:
                arr = constant_op_values[inp_name]
            elif inp_name in initializers:
                arr = initializers[inp_name]
            elif inp_name in shape_outputs:
                # Shape output - will be resolved dynamically, skip
                continue
            if arr is not None:
                _add_constant(inp_name, arr)
                constant_data_offset = sum(len(c) for c in constant_data_chunks)

    # Convert each compute node
    layers: list[dict] = []
    layer_weight_chunks: list[bytes] = []
    layer_weight_offset = 0
    op_counts: dict[str, int] = defaultdict(int)

    for node in compute_nodes:
        op = node.op_type
        result = _convert_node_v2(
            node,
            op,
            initializers,
            constant_op_values,
            constant_table,
            output_to_layer,
            shape_outputs,
            layer_weight_offset,
        )
        if result is None:
            print(f"Warning: skipping unsupported op '{op}' (node: {node.name})")
            continue

        desc, weights = result
        layers.append(desc)
        op_counts[op] += 1
        if weights is not None:
            layer_weight_chunks.append(weights)
            layer_weight_offset += len(weights)

    # Compute total constant data size
    total_constant_data = sum(len(c) for c in constant_data_chunks)

    # Adjust layer weight offsets: layer weights come after constant data
    for desc in layers:
        if desc["param_size"] > 0:
            desc["param_offset"] += total_constant_data

    # Write .ocnn v2 file
    with open(output_path, "wb") as f:
        # Header (16 bytes)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(layers)))
        f.write(struct.pack("<I", len(constant_entries)))

        # Constant table (32 bytes each)
        for entry in constant_entries:
            _write_constant_entry(f, entry)

        # Layer descriptors (80 bytes each)
        for desc in layers:
            _write_descriptor_v2(f, desc)

        # Weight data: constants first, then layer weights
        for chunk in constant_data_chunks:
            f.write(chunk)
        for chunk in layer_weight_chunks:
            f.write(chunk)

    total_weight_size = total_constant_data + layer_weight_offset
    print(f"Converted {onnx_path} -> {output_path}")
    print("  Format: .ocnn v2")
    print(f"  Layers: {len(layers)}")
    print(f"  Constants: {len(constant_entries)}")
    print(f"  Total weight size: {total_weight_size} bytes")
    print("  Op breakdown:")
    for op_name, count in sorted(op_counts.items()):
        print(f"    {op_name}: {count}")


def _extract_constant_value(node) -> np.ndarray | None:
    """Extract value from a Constant op node.

    Returns:
        The constant tensor as ndarray, or None if not found.

    """
    from onnx import numpy_helper

    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
    return None


def _resolve_input(
    inp_name: str,
    constant_table: dict[str, int],
    output_to_layer: dict[str, int],
    shape_outputs: set[str],
) -> int | None:
    """Resolve an input name to an i32 reference.

    Returns:
        >= 0: layer index
        < 0: -(constant_index + 1)
        None: graph input or unresolvable

    """
    if not inp_name:
        return None
    if inp_name in output_to_layer:
        return output_to_layer[inp_name]
    if inp_name in constant_table:
        return -(constant_table[inp_name] + 1)
    if inp_name in shape_outputs:
        # Shape outputs are dynamic; encode as special marker
        return None
    return None


def _make_descriptor(
    layer_type: int,
    num_inputs: int,
    param_offset: int,
    param_size: int,
    config: list[int],
    inputs: list[int],
) -> dict:
    """Create a v2 layer descriptor dict.

    Returns:
        Descriptor dict with padded config and inputs.

    """
    cfg = (config + [0] * 10)[:10]
    inp = (inputs + [0] * 4)[:4]
    return {
        "layer_type": layer_type,
        "num_inputs": num_inputs,
        "param_offset": param_offset,
        "param_size": param_size,
        "config": cfg,
        "inputs": inp,
    }


def _convert_node_v2(
    node,
    op: str,
    initializers: dict[str, np.ndarray],
    constant_op_values: dict[str, np.ndarray],
    constant_table: dict[str, int],
    output_to_layer: dict[str, int],
    shape_outputs: set[str],
    weight_offset: int,
) -> tuple[dict, bytes | None] | None:
    """Convert a single ONNX node to v2 layer descriptor + weights.

    Returns:
        Tuple of (descriptor dict, weight bytes) or None if unsupported.

    """

    def _resolve(name: str) -> int:
        r = _resolve_input(name, constant_table, output_to_layer, shape_outputs)
        return r if r is not None else 0

    if op == "Conv":
        return _convert_conv_v2(node, initializers, output_to_layer, weight_offset)
    elif op == "BatchNormalization":
        return _convert_batchnorm_v2(node, initializers, output_to_layer, weight_offset)
    elif op in ("Relu", "HardSwish", "Sigmoid", "Sqrt"):
        inputs = [_resolve(node.input[0])]
        return _make_descriptor(LAYER_TYPES[op], 1, 0, 0, [], inputs), None
    elif op in ("MaxPool", "AveragePool"):
        return _convert_pool_v2(node, op, output_to_layer)
    elif op == "Gemm":
        return _convert_gemm_v2(node, initializers, output_to_layer, weight_offset)
    elif op == "Reshape":
        return _convert_reshape_v2(
            node,
            initializers,
            constant_op_values,
            constant_table,
            output_to_layer,
            shape_outputs,
        )
    elif op == "Flatten":
        attrs = {a.name: a for a in node.attribute}
        axis = attrs["axis"].i if "axis" in attrs else 1
        inputs = [_resolve(node.input[0])]
        config = [axis]
        return _make_descriptor(LAYER_TYPES["Flatten"], 1, 0, 0, config, inputs), None
    elif op == "Transpose":
        return _convert_transpose_v2(node, output_to_layer)
    elif op in ("Add", "Sub", "Mul", "Div"):
        inputs_ref = [_resolve(node.input[0]), _resolve(node.input[1])]
        return (
            _make_descriptor(LAYER_TYPES[op], 2, 0, 0, [], inputs_ref),
            None,
        )
    elif op == "MatMul":
        inputs_ref = [_resolve(node.input[0]), _resolve(node.input[1])]
        return (
            _make_descriptor(LAYER_TYPES["MatMul"], 2, 0, 0, [], inputs_ref),
            None,
        )
    elif op == "Softmax":
        attrs = {a.name: a for a in node.attribute}
        axis = attrs["axis"].i if "axis" in attrs else -1
        inputs_ref = [_resolve(node.input[0])]
        config = [_i32_as_u32(axis)]
        return (
            _make_descriptor(LAYER_TYPES["Softmax"], 1, 0, 0, config, inputs_ref),
            None,
        )
    elif op == "Concat":
        return _convert_concat_v2(node, output_to_layer, constant_table, shape_outputs)
    elif op == "Slice":
        return _convert_slice_v2(
            node,
            initializers,
            constant_op_values,
            constant_table,
            output_to_layer,
            shape_outputs,
        )
    elif op == "Squeeze":
        return _convert_squeeze_v2(
            node, initializers, constant_op_values, output_to_layer
        )
    elif op == "Unsqueeze":
        return _convert_unsqueeze_v2(
            node, initializers, constant_op_values, output_to_layer
        )
    elif op == "ReduceMean":
        return _convert_reducemean_v2(node, output_to_layer)
    elif op == "Pow":
        inputs_ref = [_resolve(node.input[0]), _resolve(node.input[1])]
        return (
            _make_descriptor(LAYER_TYPES["Pow"], 2, 0, 0, [], inputs_ref),
            None,
        )
    elif op == "Gather":
        inputs_ref = [_resolve(node.input[0]), _resolve(node.input[1])]
        attrs = {a.name: a for a in node.attribute}
        axis = attrs["axis"].i if "axis" in attrs else 0
        config = [_i32_as_u32(axis)]
        return (
            _make_descriptor(LAYER_TYPES["Gather"], 2, 0, 0, config, inputs_ref),
            None,
        )
    return None


def _i32_as_u32(val: int) -> int:
    """Convert a signed i32 value to its u32 bit representation.

    Returns:
        The u32 bit-cast value.

    """
    return struct.unpack("<I", struct.pack("<i", val))[0]


def _f32_as_u32(val: float) -> int:
    """Convert f32 to u32 bit representation.

    Returns:
        The u32 bit-cast value.

    """
    return struct.unpack("<I", struct.pack("<f", val))[0]


def _get_constant_array(
    name: str,
    initializers: dict[str, np.ndarray],
    constant_op_values: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Get array from initializer or Constant op.

    Returns:
        The array if found, or None.

    """
    if name in constant_op_values:
        return constant_op_values[name]
    if name in initializers:
        return initializers[name]
    return None


# --- v2 converters ---


def _convert_conv_v2(node, initializers, output_to_layer, weight_offset):
    attrs = {a.name: a for a in node.attribute}
    weight = initializers.get(node.input[1])
    if weight is None:
        return None

    cout, cin, kh, kw = weight.shape
    group = attrs.get("group", None)
    group_val = group.i if group else 1

    strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]

    is_depthwise = group_val == cin and cout == cin
    layer_type = LAYER_TYPES["ConvDepthwise"] if is_depthwise else LAYER_TYPES["Conv"]

    has_bias = len(node.input) > 2 and node.input[2] and node.input[2] in initializers
    bias = initializers[node.input[2]] if has_bias else None

    weight_data = weight.astype(np.float32).tobytes()
    if bias is not None:
        weight_data += bias.astype(np.float32).tobytes()

    config = [
        cout,
        cin,
        kh,
        kw,
        strides[0],
        strides[1],
        pads[0],
        pads[2],
        int(has_bias),
    ]

    # Resolve data input
    inp_ref = _resolve_input(node.input[0], {}, output_to_layer, set())
    inputs = [inp_ref if inp_ref is not None else 0]

    return (
        _make_descriptor(
            layer_type, 1, weight_offset, len(weight_data), config, inputs
        ),
        weight_data,
    )


def _convert_batchnorm_v2(node, initializers, output_to_layer, weight_offset):
    gamma = initializers.get(node.input[1])
    beta = initializers.get(node.input[2])
    mean = initializers.get(node.input[3])
    var = initializers.get(node.input[4])

    if gamma is None:
        return None

    channels = len(gamma)
    attrs = {a.name: a for a in node.attribute}
    eps = attrs.get("epsilon", None)
    eps_val = eps.f if eps else 1e-5

    weight_data = (
        gamma.astype(np.float32).tobytes()
        + beta.astype(np.float32).tobytes()
        + mean.astype(np.float32).tobytes()
        + var.astype(np.float32).tobytes()
    )

    config = [channels, _f32_as_u32(eps_val)]

    inp_ref = _resolve_input(node.input[0], {}, output_to_layer, set())
    inputs = [inp_ref if inp_ref is not None else 0]

    return (
        _make_descriptor(
            LAYER_TYPES["BatchNormalization"],
            1,
            weight_offset,
            len(weight_data),
            config,
            inputs,
        ),
        weight_data,
    )


def _convert_pool_v2(node, op, output_to_layer):
    attrs = {a.name: a for a in node.attribute}
    kernel = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [2, 2]
    strides = list(attrs["strides"].ints) if "strides" in attrs else [2, 2]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]

    config = [kernel[0], kernel[1], strides[0], strides[1], pads[0], pads[2]]
    layer_type = (
        LAYER_TYPES["MaxPool"] if op == "MaxPool" else LAYER_TYPES["AveragePool"]
    )

    inp_ref = _resolve_input(node.input[0], {}, output_to_layer, set())
    inputs = [inp_ref if inp_ref is not None else 0]

    return _make_descriptor(layer_type, 1, 0, 0, config, inputs), None


def _convert_gemm_v2(node, initializers, output_to_layer, weight_offset):
    weight = initializers.get(node.input[1])
    if weight is None:
        return None

    out_f, in_f = weight.shape
    has_bias = len(node.input) > 2 and node.input[2] and node.input[2] in initializers
    bias = initializers[node.input[2]] if has_bias else None

    weight_data = weight.astype(np.float32).tobytes()
    if bias is not None:
        weight_data += bias.astype(np.float32).tobytes()

    config = [out_f, in_f, int(has_bias)]

    inp_ref = _resolve_input(node.input[0], {}, output_to_layer, set())
    inputs = [inp_ref if inp_ref is not None else 0]

    return (
        _make_descriptor(
            LAYER_TYPES["Gemm"], 1, weight_offset, len(weight_data), config, inputs
        ),
        weight_data,
    )


def _convert_reshape_v2(
    node,
    initializers,
    constant_op_values,
    constant_table,
    output_to_layer,
    shape_outputs,
):
    def _resolve(name):
        return _resolve_input(name, constant_table, output_to_layer, shape_outputs)

    data_input = _resolve(node.input[0])
    inputs = [data_input if data_input is not None else 0]
    num_inputs = 1

    # Try to get static shape
    shape_name = node.input[1] if len(node.input) > 1 else None
    shape_arr = None
    if shape_name:
        shape_arr = _get_constant_array(shape_name, initializers, constant_op_values)

    if shape_arr is not None:
        # Static shape: encode in config
        shape_list = shape_arr.flatten().astype(np.int64).tolist()
        ndim = len(shape_list)
        config = [ndim]
        for s in shape_list[:9]:
            if s == -1:
                config.append(0xFFFFFFFF)
            else:
                config.append(_i32_as_u32(int(s)))
    else:
        # Dynamic shape: reference via inputs[1]
        if shape_name:
            shape_ref = _resolve(shape_name)
            if shape_ref is not None:
                inputs.append(shape_ref)
                num_inputs = 2
        config = [0]  # ndim=0 signals dynamic

    return (
        _make_descriptor(LAYER_TYPES["Reshape"], num_inputs, 0, 0, config, inputs),
        None,
    )


def _convert_transpose_v2(node, output_to_layer):
    attrs = {a.name: a for a in node.attribute}
    perm = list(attrs["perm"].ints) if "perm" in attrs else []
    ndim = len(perm)
    config = [ndim] + [_i32_as_u32(p) for p in perm[:9]]

    inp_ref = _resolve_input(node.input[0], {}, output_to_layer, set())
    inputs = [inp_ref if inp_ref is not None else 0]

    return (
        _make_descriptor(LAYER_TYPES["Transpose"], 1, 0, 0, config, inputs),
        None,
    )


def _convert_concat_v2(node, output_to_layer, constant_table, shape_outputs):
    attrs = {a.name: a for a in node.attribute}
    axis = attrs["axis"].i if "axis" in attrs else 0

    inputs_ref = []
    for inp_name in node.input:
        r = _resolve_input(inp_name, constant_table, output_to_layer, shape_outputs)
        inputs_ref.append(r if r is not None else 0)

    num_inputs = len(inputs_ref)
    config = [_i32_as_u32(axis)]

    # Concat can have > 4 inputs; we only support up to 4 in the descriptor
    if num_inputs > 4:
        print(
            f"Warning: Concat node '{node.name}' has {num_inputs} inputs, "
            f"truncating to 4"
        )
        num_inputs = 4
        inputs_ref = inputs_ref[:4]

    return (
        _make_descriptor(LAYER_TYPES["Concat"], num_inputs, 0, 0, config, inputs_ref),
        None,
    )


def _convert_slice_v2(
    node,
    initializers,
    constant_op_values,
    constant_table,
    output_to_layer,
    shape_outputs,
):
    """Convert Slice op.

    ONNX Slice: input, starts, ends, axes (optional), steps (optional)
    For static parameters, encode in config.
    For dynamic, use input references.

    Returns:
        Tuple of (descriptor dict, None) for Slice layer.

    """

    def _resolve(name):
        return _resolve_input(name, constant_table, output_to_layer, shape_outputs)

    data_ref = _resolve(node.input[0])
    inputs = [data_ref if data_ref is not None else 0]
    num_inputs = 1

    # Try to extract static slice params
    starts_arr = (
        _get_constant_array(node.input[1], initializers, constant_op_values)
        if len(node.input) > 1 and node.input[1]
        else None
    )
    ends_arr = (
        _get_constant_array(node.input[2], initializers, constant_op_values)
        if len(node.input) > 2 and node.input[2]
        else None
    )
    axes_arr = (
        _get_constant_array(node.input[3], initializers, constant_op_values)
        if len(node.input) > 3 and node.input[3]
        else None
    )
    steps_arr = (
        _get_constant_array(node.input[4], initializers, constant_op_values)
        if len(node.input) > 4 and node.input[4]
        else None
    )

    if starts_arr is not None and ends_arr is not None:
        # Static slice: single axis for now
        starts = starts_arr.flatten().astype(np.int64).tolist()
        ends = ends_arr.flatten().astype(np.int64).tolist()
        if axes_arr is not None:
            axes = axes_arr.flatten().astype(np.int64).tolist()
        else:
            axes = list(range(len(starts)))
        if steps_arr is not None:
            steps = steps_arr.flatten().astype(np.int64).tolist()
        else:
            steps = [1] * len(starts)

        if len(starts) == 1:
            config = [
                _i32_as_u32(int(axes[0])),
                _i32_as_u32(int(starts[0])),
                _i32_as_u32(int(ends[0])),
                _i32_as_u32(int(steps[0])),
            ]
        else:
            # Multi-axis: use inputs for starts/ends/axes/steps via constant table
            config = [len(starts)]
            for i in range(1, 5):
                if i < len(node.input) and node.input[i]:
                    r = _resolve(node.input[i])
                    if r is not None:
                        inputs.append(r)
                        num_inputs += 1
    else:
        # Dynamic slice params: reference via inputs
        config = [0]  # signal dynamic
        for i in range(1, min(len(node.input), 5)):
            if node.input[i]:
                r = _resolve(node.input[i])
                if r is not None:
                    inputs.append(r)
                    num_inputs += 1

    return (
        _make_descriptor(LAYER_TYPES["Slice"], num_inputs, 0, 0, config, inputs),
        None,
    )


def _convert_squeeze_v2(node, initializers, constant_op_values, output_to_layer):
    def _resolve(name):
        return _resolve_input(name, {}, output_to_layer, set())

    inp_ref = _resolve(node.input[0])
    inputs = [inp_ref if inp_ref is not None else 0]

    # ONNX 13+: axes as second input; older: as attribute
    axes = []
    if len(node.input) > 1 and node.input[1]:
        arr = _get_constant_array(node.input[1], initializers, constant_op_values)
        if arr is not None:
            axes = arr.flatten().astype(np.int64).tolist()
    else:
        attrs = {a.name: a for a in node.attribute}
        if "axes" in attrs:
            axes = list(attrs["axes"].ints)

    config = [len(axes)] + [_i32_as_u32(int(a)) for a in axes[:9]]

    return (
        _make_descriptor(LAYER_TYPES["Squeeze"], 1, 0, 0, config, inputs),
        None,
    )


def _convert_unsqueeze_v2(node, initializers, constant_op_values, output_to_layer):
    def _resolve(name):
        return _resolve_input(name, {}, output_to_layer, set())

    inp_ref = _resolve(node.input[0])
    inputs = [inp_ref if inp_ref is not None else 0]

    # ONNX 13+: axes as second input; older: as attribute
    axes = []
    if len(node.input) > 1 and node.input[1]:
        arr = _get_constant_array(node.input[1], initializers, constant_op_values)
        if arr is not None:
            axes = arr.flatten().astype(np.int64).tolist()
    else:
        attrs = {a.name: a for a in node.attribute}
        if "axes" in attrs:
            axes = list(attrs["axes"].ints)

    config = [len(axes)] + [_i32_as_u32(int(a)) for a in axes[:9]]

    return (
        _make_descriptor(LAYER_TYPES["Unsqueeze"], 1, 0, 0, config, inputs),
        None,
    )


def _convert_reducemean_v2(node, output_to_layer):
    attrs = {a.name: a for a in node.attribute}
    axes = list(attrs["axes"].ints) if "axes" in attrs else [-1]
    keepdims = attrs["keepdims"].i if "keepdims" in attrs else 1

    config = [len(axes)]
    for a in axes[:8]:
        config.append(_i32_as_u32(int(a)))
    # Pad to fill, then keepdims at the end
    while len(config) < 9:
        config.append(0)
    config.append(keepdims)

    inp_ref = _resolve_input(node.input[0], {}, output_to_layer, set())
    inputs = [inp_ref if inp_ref is not None else 0]

    return (
        _make_descriptor(LAYER_TYPES["ReduceMean"], 1, 0, 0, config, inputs),
        None,
    )


# --- Binary writers ---


def _write_constant_entry(f, entry: dict):
    """Write a 32-byte constant table entry."""
    buf = bytearray(CONSTANT_ENTRY_SIZE)
    struct.pack_into("<Q", buf, 0, entry["data_offset"])
    struct.pack_into("<Q", buf, 8, entry["data_size"])
    struct.pack_into("<I", buf, 16, entry["ndim"])
    shape = entry["shape"]
    for i in range(3):
        if i < len(shape):
            struct.pack_into("<I", buf, 20 + i * 4, shape[i])
    f.write(buf)


def _write_descriptor_v2(f, desc: dict):
    """Write an 80-byte v2 layer descriptor."""
    buf = bytearray(LAYER_DESCRIPTOR_SIZE)
    buf[0] = desc["layer_type"]
    buf[1] = desc["num_inputs"]
    # bytes 2-7: reserved
    struct.pack_into("<Q", buf, 8, desc["param_offset"])
    struct.pack_into("<Q", buf, 16, desc["param_size"])
    for i, val in enumerate(desc["config"][:10]):
        struct.pack_into("<I", buf, 24 + i * 4, val & 0xFFFFFFFF)
    for i, val in enumerate(desc["inputs"][:4]):
        struct.pack_into("<i", buf, 64 + i * 4, val)
    f.write(buf)


def main():
    """CLI entry point for ONNX to .ocnn conversion."""
    parser = argparse.ArgumentParser(description="Convert ONNX to .ocnn v2")
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument("-o", "--output", required=True, help="Output .ocnn path")
    args = parser.parse_args()
    convert_onnx_to_ocnn(args.input, args.output)


if __name__ == "__main__":
    main()
