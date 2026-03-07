"""Quantize ONNX model for INT8 inference."""

import argparse
from pathlib import Path


def main():
    """Quantize an ONNX model to INT8 using dynamic quantization.

    Raises:
        FileNotFoundError: If the input model file does not exist.

    """
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8",
    )
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output quantized ONNX model path",
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode (default: dynamic)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
