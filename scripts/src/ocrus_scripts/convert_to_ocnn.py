#!/usr/bin/env python3
"""Convert ONNX model to .ocnn format for ocrus-nn inference engine."""
import argparse
import struct

import numpy as np

MAGIC = b"OCNN"
VERSION = 1

# Layer type enum (must match Rust LayerType)
LAYER_TYPES = {
    "Conv": 1,
    "ConvDepthwise": 2,
    "BatchNormalization": 3,
    "Relu": 4,
    "HardSwish": 5,
    "MaxPool": 6,
    "AveragePool": 7,
    "Gemm": 8,  # Linear
    "Reshape": 9,
    "Flatten": 10,
    "Transpose": 11,
}


def convert_onnx_to_ocnn(
    onnx_path: str, output_path: str, fuse_bn: bool = True
):
    """Convert ONNX model to .ocnn binary format."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("Error: onnx package required. Install with: pip install onnx")
        return

    model = onnx.load(onnx_path)
    graph = model.graph

    # Build initializer lookup
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    layers = []
    weight_chunks = []
    weight_offset = 0

    for node in graph.node:
        op = node.op_type
        layer_info = convert_node(node, op, initializers, weight_offset)
        if layer_info is None:
            continue

        desc, weights = layer_info
        layers.append(desc)
        if weights is not None:
            weight_chunks.append(weights)
            weight_offset += len(weights)

    # Write .ocnn file
    with open(output_path, "wb") as f:
        # Header: magic(4) + version(4) + num_layers(4) = 12 bytes
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(layers)))

        # Layer descriptors (64 bytes each)
        for desc in layers:
            write_descriptor(f, desc)

        # Weight data
        for chunk in weight_chunks:
            f.write(chunk)

    print(f"Converted {onnx_path} -> {output_path}")
    print(f"  Layers: {len(layers)}")
    print(f"  Weights: {weight_offset} bytes")


def convert_node(node, op, initializers, weight_offset):
    """Convert a single ONNX node to layer descriptor + weights.

    Returns:
        Tuple of (descriptor dict, weight bytes) or None if unsupported.
    """
    if op == "Conv":
        return convert_conv(node, initializers, weight_offset)
    elif op == "BatchNormalization":
        return convert_batchnorm(node, initializers, weight_offset)
    elif op in ("Relu", "HardSwish"):
        config = [0] * 10
        return (
            {"type": LAYER_TYPES[op], "offset": 0, "size": 0, "config": config},
            None,
        )
    elif op in ("MaxPool", "AveragePool"):
        return convert_pool(node, op)
    elif op == "Gemm":
        return convert_gemm(node, initializers, weight_offset)
    elif op == "Reshape":
        return convert_reshape(node, initializers)
    elif op == "Flatten":
        attrs = {a.name: a for a in node.attribute}
        axis = attrs.get("axis", None)
        start = axis.i if axis else 1
        config = [start, len(node.input) - 1] + [0] * 8
        return (
            {
                "type": LAYER_TYPES["Flatten"],
                "offset": 0,
                "size": 0,
                "config": config,
            },
            None,
        )
    elif op == "Transpose":
        attrs = {a.name: a for a in node.attribute}
        perm = list(attrs["perm"].ints) if "perm" in attrs else [0, 1]
        config = list(perm[:2]) + [0] * 8
        return (
            {
                "type": LAYER_TYPES["Transpose"],
                "offset": 0,
                "size": 0,
                "config": config,
            },
            None,
        )
    return None


def convert_conv(node, initializers, weight_offset):
    """Convert Conv node to layer descriptor and weight data."""
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
    layer_type = (
        LAYER_TYPES["ConvDepthwise"] if is_depthwise else LAYER_TYPES["Conv"]
    )

    has_bias = len(node.input) > 2 and node.input[2] in initializers
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
        0,
    ]

    return (
        {
            "type": layer_type,
            "offset": weight_offset,
            "size": len(weight_data),
            "config": config,
        },
        weight_data,
    )


def convert_batchnorm(node, initializers, weight_offset):
    """Convert BatchNormalization node to layer descriptor and weight data."""
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

    config = [
        channels,
        struct.unpack("<I", struct.pack("<f", eps_val))[0],
    ] + [0] * 8

    return (
        {
            "type": LAYER_TYPES["BatchNormalization"],
            "offset": weight_offset,
            "size": len(weight_data),
            "config": config,
        },
        weight_data,
    )


def convert_pool(node, op):
    """Convert MaxPool or AveragePool node to layer descriptor."""
    attrs = {a.name: a for a in node.attribute}
    kernel = (
        list(attrs["kernel_shape"].ints)
        if "kernel_shape" in attrs
        else [2, 2]
    )
    strides = list(attrs["strides"].ints) if "strides" in attrs else [2, 2]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]

    config = [
        kernel[0],
        kernel[1],
        strides[0],
        strides[1],
        pads[0],
        pads[2],
    ] + [0] * 4
    layer_type = (
        LAYER_TYPES["MaxPool"] if op == "MaxPool" else LAYER_TYPES["AveragePool"]
    )

    return (
        {"type": layer_type, "offset": 0, "size": 0, "config": config},
        None,
    )


def convert_gemm(node, initializers, weight_offset):
    weight = initializers.get(node.input[1])
    if weight is None:
        return None

    out_f, in_f = weight.shape
    has_bias = len(node.input) > 2 and node.input[2] in initializers
    bias = initializers[node.input[2]] if has_bias else None

    weight_data = weight.astype(np.float32).tobytes()
    if bias is not None:
        weight_data += bias.astype(np.float32).tobytes()

    config = [out_f, in_f, int(has_bias)] + [0] * 7

    return (
        {
            "type": LAYER_TYPES["Gemm"],
            "offset": weight_offset,
            "size": len(weight_data),
            "config": config,
        },
        weight_data,
    )


def convert_reshape(node, initializers):
    shape_input = node.input[1] if len(node.input) > 1 else None
    shape = initializers.get(shape_input) if shape_input else None

    if shape is not None:
        shape_list = shape.astype(np.int32).tolist()
        ndim = len(shape_list)
        config = (
            [ndim]
            + shape_list[:9]
            + [0] * (9 - len(shape_list[:9]))
        )
    else:
        config = [0] * 10

    return (
        {"type": LAYER_TYPES["Reshape"], "offset": 0, "size": 0, "config": config},
        None,
    )


def write_descriptor(f, desc):
    """Write a 64-byte layer descriptor."""
    buf = bytearray(64)
    buf[0] = desc["type"]
    # bytes 1-7: reserved
    struct.pack_into("<Q", buf, 8, desc["offset"])
    struct.pack_into("<Q", buf, 16, desc["size"])
    for i, val in enumerate(desc["config"][:10]):
        struct.pack_into("<I", buf, 24 + i * 4, val)
    f.write(buf)


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to .ocnn")
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument(
        "-o", "--output", required=True, help="Output .ocnn path"
    )
    parser.add_argument(
        "--fuse-bn",
        action="store_true",
        default=True,
        help="Fuse BatchNorm into Conv",
    )
    args = parser.parse_args()
    convert_onnx_to_ocnn(args.input, args.output, args.fuse_bn)


if __name__ == "__main__":
    main()
