#!/usr/bin/env bash
# ClearParty: Linux Launcher Script for WebApp (CPU / GPU)
set -e

echo "=========================================================="
echo "🎙️ ClearParty AI: Audio Separation & Speech Enhancement"
echo "=========================================================="

# Check Python environment
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo "❌ Error: Python not found. Please activate your conda/venv environment."
    exit 1
fi

echo "🔍 Using Python: $($PY_CMD --version)"

# Check CUDA GPU status
$PY_CMD -c "
import torch
if torch.cuda.is_available():
    print(f'✅ NVIDIA GPU Detected: {torch.cuda.get_device_name(0)} ({round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB VRAM)')
else:
    print('ℹ️ Running in CPU Mode (AVX2 Accelerated)')
"

# Check Streamlit
if ! $PY_CMD -c "import streamlit" &> /dev/null; then
    echo "⚠️ Streamlit not installed. Installing requirements..."
    pip install streamlit soundfile librosa scipy numpy pandas matplotlib
fi

echo "🚀 Starting ClearParty WebApp on http://localhost:8501..."
$PY_CMD -m streamlit run webapp.py --server.port 8501 --server.address 0.0.0.0
