@echo off
REM Export fine-tuned PP-OCRv5 model to ONNX
REM Usage: export_onnx.bat [--output DIR] [--install]
REM
REM Step 1: Export checkpoint to inference format (GPU, Windows)
REM Step 2: Convert inference model to ONNX via WSL (paddle2onnx)

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64"
set "PATH=%CUDNN_PATH%;%CUDA_PATH%\bin;%PATH%"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"

cd /d "%~dp0"

echo === Step 1: Export checkpoint to inference format ===
uv run python -m ocrus_scripts.export_inference %*
if errorlevel 1 (
    echo Step 1 failed.
    exit /b 1
)

echo.
echo === Step 2: Convert to ONNX via WSL ===
uv run python -m ocrus_scripts.convert_onnx_wsl %*
