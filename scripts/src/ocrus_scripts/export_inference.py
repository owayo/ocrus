#!/usr/bin/env python3
"""Export PaddleOCR checkpoint to Paddle inference format (PIR).

Step 1 of the ONNX export pipeline. Produces inference.json + inference.pdiparams
which can then be converted to ONNX via paddle2onnx (Step 2).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_paddleocr_root() -> Path:
    """Find PaddleOCR repo root installed by PaddleX.

    Returns:
        Path to the PaddleOCR repository root directory.

    """
    try:
        import paddlex

        repo = Path(paddlex.__file__).parent / "repo_manager" / "repos" / "PaddleOCR"
        if repo.exists():
            return repo
    except ImportError:
        pass

    print(
        "Error: PaddleOCR repo not found.\n"
        "Install PaddleOCR plugin with:\n"
        "  uv run python -m paddlex --install PaddleOCR -y --no_deps",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    """Export checkpoint to Paddle inference format."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Export PaddleOCR checkpoint to inference format"
    )
    parser.add_argument(
        "--output",
        default="./finetune_output",
        help="Fine-tune output directory (containing train_config.yml and rec_model/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    config_path = output_dir / "train_config.yml"
    best_model = output_dir / "rec_model" / "best_accuracy"
    inference_dir = output_dir / "inference"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if not best_model.with_suffix(".pdparams").exists():
        print(
            f"Error: Best model not found: {best_model}.pdparams",
            file=sys.stderr,
        )
        sys.exit(1)

    ocr_root = _find_paddleocr_root()
    export_py = ocr_root / "tools" / "export_model.py"

    # Load config and set export parameters
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["Global"]["pretrained_model"] = str(best_model)
    cfg["Global"]["save_inference_dir"] = str(inference_dir)

    export_config = output_dir / "export_config.yml"
    with export_config.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    export_cmd = [sys.executable, str(export_py), "-c", str(export_config)]

    print(f"Checkpoint: {best_model}")
    print(f"Output:     {inference_dir}")

    result = subprocess.run(export_cmd, cwd=str(ocr_root))
    if result.returncode != 0:
        print(
            f"Export failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    # Verify output
    json_file = inference_dir / "inference.json"
    pdiparams = inference_dir / "inference.pdiparams"
    if json_file.exists() and pdiparams.exists():
        size_mb = pdiparams.stat().st_size / (1024 * 1024)
        print(f"\nInference model exported ({size_mb:.1f} MB)")
        print(f"  {json_file}")
        print(f"  {pdiparams}")
    else:
        print("Error: Expected output files not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
