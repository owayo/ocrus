@echo off
REM Fine-tune PP-OCRv5 recognition model
REM Usage: finetune.bat [--data DIR] [--epochs N] [--batch-size N] [--lr F] [--device DEV] [--export-onnx] [--output DIR]

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64"
set "PATH=%CUDNN_PATH%;%CUDA_PATH%\bin;%PATH%"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"

cd /d "%~dp0"
uv run python -m ocrus_scripts.finetune %*
