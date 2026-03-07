#!/usr/bin/env python3
"""Fine-tune PP-OCRv5 recognition model using PaddleX API."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _check_paddle_installed() -> None:
    """Check that PaddlePaddle is available."""
    try:
        import paddle  # noqa: F401
    except ImportError:
        print(
            "Error: PaddlePaddle is not installed.\n"
            "Install training dependencies with:\n"
            "  uv pip install -e '.[train]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _convert_ocrus_to_paddlex(data_dir: Path, output_dir: Path) -> Path:
    r"""Convert ocrus dataset format to PaddleX format.

    PaddleX expects:
      dataset_dir/
        images/
          img1.png
          img2.png
        train.txt   (image_path\tlabel per line)
        val.txt

    Args:
        data_dir: ocrus dataset directory (containing labels.tsv + samples/).
        output_dir: Directory to write PaddleX-format dataset.

    Returns:
        Path to the PaddleX dataset directory.

    """
    import csv
    import random

    labels_file = data_dir / "labels.tsv"
    samples_dir = data_dir / "samples"

    if not labels_file.exists():
        print(f"Error: {labels_file} not found.", file=sys.stderr)
        sys.exit(1)

    entries: list[tuple[str, str]] = []
    with labels_file.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            filename, label = row[0], row[1]
            src = samples_dir / filename
            if src.exists():
                entries.append((filename, label))

    if not entries:
        print("Error: No valid entries found in labels.tsv.", file=sys.stderr)
        sys.exit(1)

    # Shuffle and split
    random.shuffle(entries)
    val_count = max(1, int(len(entries) * 0.1))
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    # Create PaddleX dataset structure
    pdx_dir = output_dir / "paddlex_dataset"
    images_dir = pdx_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images
    for filename, _ in entries:
        src = samples_dir / filename
        dst = images_dir / filename
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # Write label files
    def _write_labels(items: list[tuple[str, str]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for filename, label in items:
                f.write(f"images/{filename}\t{label}\n")

    _write_labels(train_entries, pdx_dir / "train.txt")
    _write_labels(val_entries, pdx_dir / "val.txt")

    print(f"PaddleX dataset: {len(train_entries)} train, {len(val_entries)} val")
    return pdx_dir


def _find_paddlex_main() -> Path:
    """Find PaddleX main.py entry point.

    Returns:
        Path to the PaddleX main.py script.

    """
    try:
        import paddlex

        pkg_dir = Path(paddlex.__file__).parent
        main_py = pkg_dir / "main.py"
        if main_py.exists():
            return main_py
        # Some versions use run_pipeline.py
        run_py = pkg_dir / "run_pipeline.py"
        if run_py.exists():
            return run_py
    except ImportError:
        pass

    # PaddleX may also be available as a CLI module
    return Path("paddlex.main")


def main() -> None:
    """Fine-tune PP-OCRv5 recognition model with ocrus dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv5 recognition model")
    parser.add_argument("--data", required=True, help="ocrus training data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to pretrained .pdparams (auto-downloaded if not specified)",
    )
    parser.add_argument(
        "--output",
        default="./finetune_output",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Device (gpu:0, cpu, etc.)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export to ONNX after training",
    )
    args = parser.parse_args()

    _check_paddle_installed()

    data_dir = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert ocrus dataset to PaddleX format
    pdx_dataset = _convert_ocrus_to_paddlex(data_dir, output_dir)

    # Find PP-OCRv5_server_rec config from PaddleX
    try:
        import paddlex

        pkg_dir = Path(paddlex.__file__).parent
        config_path = (
            pkg_dir
            / "configs"
            / "modules"
            / "text_recognition"
            / "PP-OCRv5_server_rec.yaml"
        )
        if not config_path.exists():
            print(f"Error: PaddleX config not found at {config_path}", file=sys.stderr)
            sys.exit(1)
    except ImportError:
        print(
            "Error: PaddleX not found. Install with:\n  uv pip install -e '.[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build PaddleX training command
    main_py = _find_paddlex_main()

    overrides = [
        "Global.mode=train",
        f"Global.dataset_dir={pdx_dataset}",
        f"Global.device={args.device}",
        f"Global.output={output_dir}",
        f"Train.epochs_iters={args.epochs}",
        f"Train.batch_size={args.batch_size}",
        f"Train.learning_rate={args.lr}",
    ]

    if args.pretrained:
        pretrained_path = Path(args.pretrained).expanduser().resolve()
        if not pretrained_path.exists():
            print(
                f"Error: Pretrained model not found: {pretrained_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        overrides.append(f"Train.pretrain_weight_path={pretrained_path}")

    cmd = [sys.executable, str(main_py), "-c", str(config_path)]
    for ov in overrides:
        cmd.extend(["-o", ov])

    print(f"\n{'=' * 60}")
    print("Starting PP-OCRv5 fine-tuning via PaddleX")
    print(f"{'=' * 60}")
    print(f"Config:     {config_path}")
    print(f"Dataset:    {pdx_dataset}")
    print(f"Output:     {output_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Device:     {args.device}")
    if args.pretrained:
        print(f"Pretrained: {args.pretrained}")
    print(f"{'=' * 60}\n")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nTraining complete. Model saved to: {output_dir}")

    # Export to ONNX if requested
    if args.export_onnx:
        inference_dir = output_dir / "best_accuracy" / "inference"
        if inference_dir.exists():
            onnx_output = output_dir / "rec_finetuned.onnx"
            print(f"\nExporting to ONNX: {onnx_output}")
            export_cmd = [
                sys.executable,
                "-m",
                "paddle2onnx",
                "--model_dir",
                str(inference_dir),
                "--model_filename",
                "inference.pdmodel",
                "--params_filename",
                "inference.pdiparams",
                "--save_file",
                str(onnx_output),
                "--opset_version",
                "11",
                "--enable_onnx_checker",
                "True",
            ]
            subprocess.run(export_cmd, check=True)
            print(f"ONNX model exported to: {onnx_output}")

            # Optionally install
            install_dir = Path.home() / ".ocrus" / "models"
            install_path = install_dir / "rec.onnx"
            print("\nTo install the fine-tuned model:")
            print(f"  cp {onnx_output} {install_path}")
        else:
            print(
                f"Warning: Inference dir not found at {inference_dir}. "
                "Run export-onnx manually.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
