#!/usr/bin/env python3
"""Fine-tune PP-OCRv5 recognition model using PaddleOCR tools directly."""

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
            "  uv sync --extra train",
            file=sys.stderr,
        )
        sys.exit(1)


def _find_paddleocr_root() -> Path:
    """Find PaddleOCR repo root installed by PaddleX."""
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


def _convert_ocrus_to_paddleocr(data_dir: Path, output_dir: Path) -> Path:
    r"""Convert ocrus dataset format to PaddleOCR format.

    PaddleOCR expects:
      dataset_dir/
        images/
          img1.png
        train.txt   (image_path\tlabel per line)
        val.txt

    Args:
        data_dir: ocrus dataset directory (containing labels.tsv + samples/).
        output_dir: Directory to write PaddleOCR-format dataset.

    Returns:
        Path to the dataset directory.

    """
    import csv
    import random
    import shutil

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

    # Create dataset structure
    ds_dir = output_dir / "dataset"
    images_dir = ds_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy images (symlinks require admin privileges on Windows)
    for filename, _ in entries:
        src = samples_dir / filename
        dst = images_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)

    # Write label files (PaddleOCR format: relative_path\tlabel)
    def _write_labels(items: list[tuple[str, str]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for filename, label in items:
                f.write(f"images/{filename}\t{label}\n")

    _write_labels(train_entries, ds_dir / "train.txt")
    _write_labels(val_entries, ds_dir / "val.txt")

    print(f"Dataset: {len(train_entries)} train, {len(val_entries)} val")
    return ds_dir


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

    # Find PaddleOCR repo
    ocr_root = _find_paddleocr_root()
    train_py = ocr_root / "tools" / "train.py"
    config_yaml = ocr_root / "configs" / "rec" / "PP-OCRv5" / "PP-OCRv5_server_rec.yml"
    dict_path = ocr_root / "ppocr" / "utils" / "dict" / "ppocrv5_dict.txt"

    if not config_yaml.exists():
        print(f"Error: Config not found at {config_yaml}", file=sys.stderr)
        sys.exit(1)

    # Convert dataset
    ds_dir = _convert_ocrus_to_paddleocr(data_dir, output_dir)

    # Resolve pretrained path
    pretrained = args.pretrained
    if pretrained:
        pretrained = str(Path(pretrained).expanduser().resolve())
        if not Path(pretrained).exists():
            print(f"Error: Pretrained model not found: {pretrained}", file=sys.stderr)
            sys.exit(1)

    # Determine device flags
    device = args.device
    use_gpu = "gpu" in device

    # Build PaddleOCR training command
    save_dir = str(output_dir / "rec_model")
    cmd = [
        sys.executable,
        str(train_py),
        "-c",
        str(config_yaml),
        "-o",
        f"Global.use_gpu={use_gpu}",
        "-o",
        f"Global.epoch_num={args.epochs}",
        "-o",
        f"Global.save_model_dir={save_dir}",
        "-o",
        f"Global.character_dict_path={dict_path}",
        "-o",
        f"Train.dataset.data_dir={ds_dir}",
        "-o",
        f"Train.dataset.label_file_list=['{ds_dir / 'train.txt'}']",
        "-o",
        f"Train.loader.batch_size_per_card={args.batch_size}",
        "-o",
        f"Train.sampler.first_bs={args.batch_size}",
        "-o",
        f"Eval.dataset.data_dir={ds_dir}",
        "-o",
        f"Eval.dataset.label_file_list=['{ds_dir / 'val.txt'}']",
        "-o",
        f"Eval.loader.batch_size_per_card={args.batch_size}",
        "-o",
        f"Optimizer.lr.learning_rate={args.lr}",
    ]

    if pretrained:
        cmd.extend(["-o", f"Global.pretrained_model={pretrained}"])

    print(f"\n{'=' * 60}")
    print("Starting PP-OCRv5 fine-tuning via PaddleOCR")
    print(f"{'=' * 60}")
    print(f"Config:     {config_yaml}")
    print(f"Dataset:    {ds_dir}")
    print(f"Output:     {save_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Device:     {device}")
    if pretrained:
        print(f"Pretrained: {pretrained}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=str(ocr_root))
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nTraining complete. Model saved to: {save_dir}")

    # Export to ONNX if requested
    if args.export_onnx:
        # First export to inference format
        export_py = ocr_root / "tools" / "export_model.py"
        best_model = Path(save_dir) / "best_accuracy"
        if best_model.exists():
            inference_dir = output_dir / "inference"
            export_cmd = [
                sys.executable,
                str(export_py),
                "-c",
                str(config_yaml),
                "-o",
                f"Global.pretrained_model={best_model}",
                "-o",
                f"Global.save_inference_dir={inference_dir}",
                "-o",
                f"Global.character_dict_path={dict_path}",
            ]
            print(f"\nExporting to inference format: {inference_dir}")
            subprocess.run(export_cmd, cwd=str(ocr_root), check=True)

            # Convert to ONNX
            onnx_output = output_dir / "rec_finetuned.onnx"
            print(f"Converting to ONNX: {onnx_output}")
            onnx_cmd = [
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
            subprocess.run(onnx_cmd, check=True)
            print(f"ONNX model exported to: {onnx_output}")

            install_dir = Path.home() / ".ocrus" / "models"
            install_path = install_dir / "rec.onnx"
            print("\nTo install the fine-tuned model:")
            print(f"  cp {onnx_output} {install_path}")
        else:
            print(
                f"Warning: Best model not found at {best_model}. "
                "Training may not have completed successfully.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
