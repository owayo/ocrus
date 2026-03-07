#!/usr/bin/env python3
"""Export fine-tuned PaddlePaddle model to ONNX format for ocrus."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _check_dependencies() -> None:
    """Check that paddle2onnx and onnx are available."""
    missing = []
    try:
        import paddle2onnx  # noqa: F401
    except ImportError:
        missing.append("paddle2onnx")
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")

    if missing:
        print(
            f"Error: Missing dependencies: {', '.join(missing)}\n"
            "Install training dependencies with:\n"
            "  uv pip install -e '.[train]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _find_paddle_model(model_dir: Path) -> tuple[Path, Path]:
    """Find inference model and params files in model directory.

    Args:
        model_dir: Directory containing PaddlePaddle model files.

    Returns:
        Tuple of (model_file, params_file).

    """
    # Check for inference model format first
    inference_dir = model_dir / "inference"
    candidates = [inference_dir, model_dir]

    for d in candidates:
        model_file = d / "inference.pdmodel"
        params_file = d / "inference.pdiparams"
        if model_file.exists() and params_file.exists():
            return model_file, params_file

    # Try alternative naming
    for d in candidates:
        for mf in d.glob("*.pdmodel"):
            pf = mf.with_suffix(".pdiparams")
            if pf.exists():
                return mf, pf

    print(
        f"Error: Could not find PaddlePaddle model files in {model_dir}\n"
        "Expected: inference.pdmodel and inference.pdiparams",
        file=sys.stderr,
    )
    sys.exit(1)


def _export_to_onnx(model_file: Path, params_file: Path, output_path: Path) -> None:
    """Convert PaddlePaddle model to ONNX using paddle2onnx.

    Args:
        model_file: Path to .pdmodel file.
        params_file: Path to .pdiparams file.
        output_path: Path for output .onnx file.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "paddle2onnx",
        "--model_dir",
        str(model_file.parent),
        "--model_filename",
        model_file.name,
        "--params_filename",
        params_file.name,
        "--save_file",
        str(output_path),
        "--opset_version",
        "11",
        "--enable_onnx_checker",
        "True",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _validate_onnx(onnx_path: Path) -> None:
    """Validate the exported ONNX model and print shape info.

    Args:
        onnx_path: Path to the ONNX model file.

    """
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("ONNX model validation: OK")

    print("\nModel inputs:")
    for inp in model.graph.input:
        shape = [
            d.dim_value if d.dim_value else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  {inp.name}: {shape}")

    print("\nModel outputs:")
    for out in model.graph.output:
        shape = [
            d.dim_value if d.dim_value else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  {out.name}: {shape}")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\nModel size: {size_mb:.1f} MB")


def main() -> None:
    """Export fine-tuned PaddlePaddle model to ONNX."""
    import argparse

    parser = argparse.ArgumentParser(description="Export fine-tuned model to ONNX")
    parser.add_argument("--model", required=True, help="PaddlePaddle model directory")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Copy to ~/.ocrus/models/rec.onnx",
    )
    args = parser.parse_args()

    _check_dependencies()

    model_dir = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # Find model files
    model_file, params_file = _find_paddle_model(model_dir)
    print(f"Model: {model_file}")
    print(f"Params: {params_file}")

    # Export
    _export_to_onnx(model_file, params_file, output_path)
    print(f"\nExported to: {output_path}")

    # Validate
    _validate_onnx(output_path)

    # Install to default location
    if args.install:
        install_dir = Path.home() / ".ocrus" / "models"
        install_dir.mkdir(parents=True, exist_ok=True)
        install_path = install_dir / "rec.onnx"
        shutil.copy2(output_path, install_path)
        print(f"\nInstalled to: {install_path}")


if __name__ == "__main__":
    main()
