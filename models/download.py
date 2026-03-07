#!/usr/bin/env python3
"""Download OCR models for ocrus.

Downloads PP-OCRv5 recognition model (ONNX) and dictionary to
the default model directory (~/.ocrus/models/).

Works on macOS, Linux, and Windows (no shell dependencies).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# Default URLs
DEFAULT_REC_URL = (
    "https://huggingface.co/monkt/paddleocr-onnx/"
    "resolve/main/languages/chinese/rec.onnx"
)
DEFAULT_DICT_URL = (
    "https://huggingface.co/monkt/paddleocr-onnx/"
    "resolve/main/languages/chinese/dict.txt"
)


def _download(url: str, dest: Path) -> None:
    """Download a file with progress indication.

    Args:
        url: URL to download from.
        dest: Destination file path.

    """
    print(f"  {url}")
    print(f"  -> {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "ocrus"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total else None

        downloaded = 0
        chunk_size = 8192
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = downloaded * 100 // total_bytes
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_bytes / (1024 * 1024)
                    print(
                        f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)",
                        end="",
                        flush=True,
                    )

    print()


def main() -> None:
    """Download OCR models to the model directory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download OCR models for ocrus",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model directory (default: ~/.ocrus/models/ or $OCRUS_MODEL_DIR)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
    else:
        env_dir = os.environ.get("OCRUS_MODEL_DIR")
        if env_dir:
            model_dir = Path(env_dir).expanduser().resolve()
        else:
            model_dir = Path.home() / ".ocrus" / "models"

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {model_dir}\n")

    # URLs (overridable via environment variables)
    rec_url = os.environ.get("OCRUS_REC_MODEL_URL", DEFAULT_REC_URL)
    dict_url = os.environ.get("OCRUS_DICT_URL", DEFAULT_DICT_URL)
    lite_url = os.environ.get("OCRUS_LITE_MODEL_URL", "")

    # Download recognition model
    rec_path = model_dir / "rec.onnx"
    if args.force or not rec_path.exists():
        print("Downloading recognition model (~80MB)...")
        _download(rec_url, rec_path)
    else:
        print("Recognition model already exists, skipping.")

    # Download dictionary
    dict_path = model_dir / "dict.txt"
    if args.force or not dict_path.exists():
        print("Downloading dictionary...")
        _download(dict_url, dict_path)
    else:
        print("Dictionary already exists, skipping.")

    # Download lightweight model if URL is set
    if lite_url:
        lite_path = model_dir / "rec_lite.onnx"
        if args.force or not lite_path.exists():
            print("Downloading lightweight recognition model...")
            _download(lite_url, lite_path)
        else:
            print("Lightweight model already exists, skipping.")

    # List installed files
    print(f"\nDone. Models installed to: {model_dir}")
    print("Files:")
    for f in sorted(model_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:30s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
