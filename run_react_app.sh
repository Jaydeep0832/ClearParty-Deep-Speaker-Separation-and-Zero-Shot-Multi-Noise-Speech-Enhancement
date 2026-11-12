#!/usr/bin/env bash
# ClearParty: Launch Low-Latency React UI + FastAPI Backend
set -e

echo "=========================================================="
echo "⚡ ClearParty: Modern React UI + FastAPI High-Speed Backend"
echo "=========================================================="

# Check Python command
if command -v python3 &> /dev/null; then
    PY="python3"
else
    PY="python"
fi

# Optional port argument (defaults to 7860 - Gradio standard port)
PORT="${1:-7860}"

# Ensure FastAPI and Uvicorn are installed
if ! $PY -c "import fastapi, uvicorn" &> /dev/null; then
    echo "📦 Installing FastAPI and Uvicorn..."
    pip install fastapi uvicorn python-multipart
fi

echo "🚀 Starting High-Performance API & React UI on port $PORT..."
$PY api_server.py --port "$PORT"
