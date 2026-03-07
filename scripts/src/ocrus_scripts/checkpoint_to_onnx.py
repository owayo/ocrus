#!/usr/bin/env python3
"""Export a PaddleOCR checkpoint (.pdparams) to ONNX for ocrus.

Handles the full pipeline:
  1. Load checkpoint into PaddleOCR model architecture
  2. Export to Paddle inference format (.pdmodel + .pdiparams)
  3. Convert to ONNX via paddle2onnx
  4. (Optional) Install to ~/.ocrus/models/rec.onnx

Works on macOS, Linux, and Windows (no shell dependencies).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path


def _check_dependencies() -> None:
    missing = []
    for mod in ("paddle", "paddle2onnx", "onnx"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(
            f"Error: Missing dependencies: {', '.join(missing)}\n"
            "Install with:\n  uv pip install -e '.[train]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _find_paddleocr_dir() -> Path:
    """Find PaddleOCR source directory.

    Returns:
        Path to the PaddleOCR root directory.

    """
    candidates = [
        Path("/tmp/PaddleOCR"),
        Path.home() / "PaddleOCR",
        Path.cwd() / "PaddleOCR",
    ]
    for d in candidates:
        if (d / "tools" / "export_model.py").exists():
            return d

    # Try to find via pip package
    try:
        import ppocr

        return Path(ppocr.__file__).parent.parent
    except ImportError:
        pass

    print(
        "Error: PaddleOCR source directory not found.\n"
        "Expected at: /tmp/PaddleOCR or ~/PaddleOCR",
        file=sys.stderr,
    )
    sys.exit(1)


def _export_checkpoint_to_inference(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    """Export checkpoint to Paddle inference format.

    Args:
        config_path: Path to training config.yml.
        checkpoint_path: Path to .pdparams file.
        output_dir: Directory for inference model output.

    Returns:
        Path to the inference directory.

    """
    paddleocr_dir = _find_paddleocr_dir()

    # Add PaddleOCR to sys.path for imports
    paddleocr_str = str(paddleocr_dir)
    tools_str = str(paddleocr_dir / "tools")
    for p in (paddleocr_str, tools_str):
        if p not in sys.path:
            sys.path.insert(0, p)

    from ppocr.utils.export_model import export
    from tools.program import load_config, merge_config

    # Build config with overrides
    config = load_config(str(config_path))

    # Remove .pdparams extension if present
    ckpt_stem = str(checkpoint_path)
    for ext in (".pdparams", ".pdopt", ".states"):
        if ckpt_stem.endswith(ext):
            ckpt_stem = ckpt_stem[: -len(ext)]
            break

    inference_dir = output_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "Global.pretrained_model": ckpt_stem,
        "Global.save_inference_dir": str(inference_dir),
    }
    config = merge_config(config, overrides)

    print("Exporting checkpoint to inference format...")
    print(f"  Checkpoint: {ckpt_stem}")
    print(f"  Output:     {inference_dir}")

    export(config)

    # Verify output — PaddleOCR may export either:
    #   .pdmodel (legacy) or .json (PIR format, Paddle 3.x+)
    pdiparams = inference_dir / "inference.pdiparams"
    pdmodel = inference_dir / "inference.pdmodel"
    pir_json = inference_dir / "inference.json"

    if not pdiparams.exists():
        print(
            f"Error: Export failed. inference.pdiparams not found in {inference_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not pdmodel.exists() and not pir_json.exists():
        print(
            f"Error: Export failed. No model file (.pdmodel/.json) in {inference_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    model_file = pdmodel if pdmodel.exists() else pir_json
    print(f"Inference model exported: {model_file.name}")
    return inference_dir


def _convert_to_onnx(inference_dir: Path, onnx_path: Path) -> None:
    """Convert Paddle inference model to ONNX.

    Args:
        inference_dir: Directory with .pdmodel and .pdiparams.
        onnx_path: Output ONNX file path.

    """
    import paddle2onnx

    # Support both legacy (.pdmodel) and PIR (.json) formats
    pdmodel = inference_dir / "inference.pdmodel"
    pir_json = inference_dir / "inference.json"
    model_file = str(pdmodel if pdmodel.exists() else pir_json)
    params_file = str(inference_dir / "inference.pdiparams")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nConverting to ONNX...")
    print(f"  Output: {onnx_path}")

    # Use paddle2onnx Python API directly (no subprocess)
    paddle2onnx.export(
        model_file,
        params_file,
        save_file=str(onnx_path),
        opset_version=11,
        enable_onnx_checker=True,
    )

    print("ONNX conversion complete.")


def _validate_onnx(onnx_path: Path) -> None:
    """Validate and print ONNX model info."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("\nONNX validation: OK")

    print("Inputs:")
    for inp in model.graph.input:
        shape = [
            d.dim_value if d.dim_value else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  {inp.name}: {shape}")

    print("Outputs:")
    for out in model.graph.output:
        shape = [
            d.dim_value if d.dim_value else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  {out.name}: {shape}")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")


def _install_model(onnx_path: Path) -> Path:
    """Copy ONNX model to ~/.ocrus/models/rec.onnx.

    Returns:
        Path to the installed model.

    """
    install_dir = Path.home() / ".ocrus" / "models"
    install_dir.mkdir(parents=True, exist_ok=True)
    install_path = install_dir / "rec.onnx"

    # Backup existing model
    if install_path.exists():
        backup = install_dir / "rec.onnx.bak"
        shutil.copy2(install_path, backup)
        print(f"\nBacked up existing model to: {backup}")

    shutil.copy2(onnx_path, install_path)
    print(f"Installed to: {install_path}")
    return install_path


def main() -> None:
    """Export PaddleOCR checkpoint to ONNX and optionally install."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export PaddleOCR checkpoint to ONNX for ocrus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  checkpoint-to-onnx \\\n"
            "    --config /tmp/ocrus_finetune_output/"
            "config.yml \\\n"
            "    --checkpoint /tmp/ocrus_finetune_output/"
            "best_accuracy.pdparams\n"
            "\n"
            "  checkpoint-to-onnx \\\n"
            "    --config /tmp/ocrus_finetune_output/"
            "config.yml \\\n"
            "    --checkpoint /tmp/ocrus_finetune_output/"
            "best_accuracy.pdparams \\\n"
            "    --install\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config.yml",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pdparams checkpoint file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path (default: <checkpoint_dir>/rec_finetuned.onnx)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install to ~/.ocrus/models/rec.onnx (backs up existing)",
    )
    parser.add_argument(
        "--keep-inference",
        action="store_true",
        help="Keep intermediate Paddle inference model files",
    )
    args = parser.parse_args()

    _check_dependencies()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not config_path.exists():
        print(
            f"Error: Config not found: {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not checkpoint_path.exists():
        print(
            f"Error: Checkpoint not found: {checkpoint_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine output path
    if args.output:
        onnx_path = Path(args.output).expanduser().resolve()
    else:
        onnx_path = checkpoint_path.parent / "rec_finetuned.onnx"

    # Use temp dir for intermediate inference model
    if args.keep_inference:
        work_dir = checkpoint_path.parent
        inference_dir = _export_checkpoint_to_inference(
            config_path, checkpoint_path, work_dir
        )
    else:
        tmp_dir = tempfile.mkdtemp(prefix="ocrus_export_")
        try:
            inference_dir = _export_checkpoint_to_inference(
                config_path,
                checkpoint_path,
                Path(tmp_dir),
            )
            _convert_to_onnx(inference_dir, onnx_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _validate_onnx(onnx_path)
        if args.install:
            _install_model(onnx_path)
        return

    _convert_to_onnx(inference_dir, onnx_path)
    _validate_onnx(onnx_path)

    if args.install:
        _install_model(onnx_path)


if __name__ == "__main__":
    main()
