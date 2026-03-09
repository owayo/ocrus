@echo off
REM Export fine-tuned PP-OCRv5 model to ONNX via Docker
REM Usage: export_onnx.bat [--output DIR] [--install]

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64"
set "PATH=%CUDNN_PATH%;%CUDA_PATH%\bin;%PATH%"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"

cd /d "%~dp0"

REM Step 1: Export checkpoint to PIR inference format (on Windows with GPU)
echo === Step 1: Exporting checkpoint to inference format ===
uv run python -m ocrus_scripts.export_inference %*
if errorlevel 1 (
    echo Export to inference format failed.
    exit /b 1
)

REM Step 2: Convert PIR to ONNX via Docker (paddle2onnx works on Linux)
echo.
echo === Step 2: Converting to ONNX via Docker ===
uv run python -m ocrus_scripts.convert_onnx_docker %*
