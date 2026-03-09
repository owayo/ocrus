#!/usr/bin/env python3
"""Convert Paddle inference model (PIR) to ONNX via WSL.

Step 2 of the ONNX export pipeline. Uses WSL to run paddle2onnx because
the Windows build of paddle2onnx 2.1.0 has DLL loading issues.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _wsl_path(win_path: Path) -> str:
    """Convert Windows path to WSL path.

    Args:
        win_path: Windows path to convert.

    Returns:
        WSL-compatible path string.

    """
    resolved = win_path.resolve()
    drive = resolved.drive[0].lower()
    rest = str(resolved)[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def main() -> None:
    """Convert PIR inference model to ONNX via WSL."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Paddle inference model to ONNX via WSL"
    )
    parser.add_argument(
        "--output",
        default="./finetune_output",
        help="Fine-tune output directory (containing inference/)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install to ~/.ocrus/models/rec.onnx",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    inference_dir = output_dir / "inference"
    json_file = inference_dir / "inference.json"
    pdiparams = inference_dir / "inference.pdiparams"
    onnx_output = output_dir / "rec_finetuned.onnx"

    if not json_file.exists() or not pdiparams.exists():
        print(
            f"Error: Inference model not found in {inference_dir}\n"
            "Run export_inference first (or finetune with --export-onnx).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check WSL availability
    try:
        result = subprocess.run(
            ["wsl", "--status"],
            capture_output=True,
            timeout=10,
        )
    except FileNotFoundError:
        print("Error: WSL is not installed.", file=sys.stderr)
        sys.exit(1)

    wsl_model = _wsl_path(json_file)
    wsl_params = _wsl_path(pdiparams)
    wsl_output = _wsl_path(onnx_output)

    # Build the conversion script to run inside WSL
    wsl_script = f"""\
set -e
if ! python3 -c "import paddle2onnx" 2>/dev/null; then
    echo "Installing paddle2onnx in WSL..."
    pip install -q paddle2onnx==2.1.0 paddlepaddle packaging 2>&1 | tail -3
    sudo apt-get install -y -qq libgomp1 > /dev/null 2>&1 || true
fi
PADDLE_DIR=$(python3 -c "
import os, paddle
print(os.path.join(os.path.dirname(paddle.__file__), 'libs'))
" 2>/dev/null || echo "")
export LD_LIBRARY_PATH="${{PADDLE_DIR}}:${{LD_LIBRARY_PATH}}"
python3 -c "
import paddle2onnx
model = open('{wsl_model}', 'rb').read()
params = open('{wsl_params}', 'rb').read()
print('Converting PIR to ONNX...')
result = paddle2onnx.export(
    model, params,
    opset_version=11,
    auto_upgrade_opset=True,
    verbose=True,
    enable_onnx_checker=True,
    enable_experimental_op=True,
    enable_optimize=True,
    deploy_backend='onnxruntime',
)
with open('{wsl_output}', 'wb') as f:
    f.write(result)
import os
size = os.path.getsize('{wsl_output}') / (1024*1024)
print(f'ONNX exported: {{size:.1f}} MB')
"
"""

    print(f"Input:  {json_file}")
    print(f"Output: {onnx_output}")
    print("Running paddle2onnx in WSL...\n")

    result = subprocess.run(
        ["wsl", "bash", "-c", wsl_script],
    )
    if result.returncode != 0:
        print(
            f"\nONNX conversion failed (exit code {result.returncode})",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    if not onnx_output.exists() or onnx_output.stat().st_size == 0:
        print("Error: ONNX output file is empty or missing.", file=sys.stderr)
        sys.exit(1)

    size_mb = onnx_output.stat().st_size / (1024 * 1024)
    print(f"\nONNX model: {onnx_output} ({size_mb:.1f} MB)")

    if args.install:
        install_dir = Path.home() / ".ocrus" / "models"
        install_dir.mkdir(parents=True, exist_ok=True)
        install_path = install_dir / "rec.onnx"
        if install_path.exists():
            backup = install_dir / "rec.onnx.bak"
            shutil.copy2(install_path, backup)
            print(f"Backed up existing model to: {backup}")
        shutil.copy2(onnx_output, install_path)
        print(f"Installed to: {install_path}")


if __name__ == "__main__":
    main()
