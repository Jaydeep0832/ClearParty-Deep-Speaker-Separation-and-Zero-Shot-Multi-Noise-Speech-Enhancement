# api_server.py
"""
ClearParty AI: High-Performance, Ultra-Low-Latency FastAPI Backend
Serves the Modern Gradio-Style React Frontend with GPU Acceleration.
"""

import os
import sys
import time
import io
import json
import shutil
import tempfile
import numpy as np
import scipy.signal
import soundfile as sf
import torch

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_dir(candidates):
    for c in candidates:
        full = os.path.join(BASE_DIR, c)
        if os.path.exists(full):
            return full
    return os.path.join(BASE_DIR, candidates[0])

SEP_DIR = find_dir(["speaker_separation", "speaker separation"])
ENH_DIR = find_dir(["speech_enhancement", "speech enhancement"])

sys.path.insert(0, os.path.join(SEP_DIR, "src"))
sys.path.insert(0, ENH_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "dist")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# GLOBAL MODEL CACHE (PERSISTENT IN GPU VRAM)
# -----------------------------------------------------------------------------
MODELS = {
    "sep": None,
    "enh": None,
    "whisper": None
}

def get_separation_model():
    if MODELS["sep"] is None:
        print(f"⚡ Loading Conv-TasNet Separation Model onto {DEVICE}...")
        from model import ConvTasNet
        m = ConvTasNet().to(DEVICE)
        ckpt_path = os.path.join(SEP_DIR, "checkpoints", "tasnet_v2_epoch50.pth")
        if os.path.exists(ckpt_path):
            st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            m.load_state_dict(st.get("model_state_dict", st), strict=False)
        m.eval()
        MODELS["sep"] = m
    return MODELS["sep"]

def get_enhancement_model():
    if MODELS["enh"] is None:
        print(f"⚡ Loading Conv-TasNet Denoiser onto {DEVICE}...")
        from src.model import ConvTasNetDenoiser
        m = ConvTasNetDenoiser().to(DEVICE)
        ckpt_path = os.path.join(ENH_DIR, "checkpoints", "denoiser_best.pth")
        if os.path.exists(ckpt_path):
            st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            m.load_state_dict(st.get("model_state_dict", st), strict=False)
        m.eval()
        MODELS["enh"] = m
    return MODELS["enh"]

def get_whisper_model():
    if MODELS["whisper"] is None:
        try:
            import whisper
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            # Use medium.en if VRAM >= 10GB (like TITAN Xp), else small.en
            w_name = "medium.en" if vram_gb >= 10.0 else "small.en"
            print(f"⚡ Pre-loading Whisper Model ('{w_name}') onto {DEVICE}...")
            MODELS["whisper"] = whisper.load_model(w_name, device=DEVICE)
        except Exception as e:
            print(f"Whisper initialization deferred: {e}")
            MODELS["whisper"] = False
    return MODELS["whisper"]

# -----------------------------------------------------------------------------
# FASTAPI APP INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(title="ClearParty AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# AUDIO PROCESSING PIPELINE UTILITIES
# -----------------------------------------------------------------------------
def resample_to_8k(audio, orig_sr):
    if orig_sr == 8000:
        return audio
    gcd = np.gcd(orig_sr, 8000)
    return scipy.signal.resample_poly(audio, 8000 // gcd, orig_sr // gcd).astype(np.float32)

def run_gpu_separation_core(sig_8k, model):
    chunk_samples = 32000 # 4 sec
    hop_samples = 16000   # 50% overlap
    win = np.sin((np.arange(chunk_samples) + 0.5) / chunk_samples * np.pi).astype(np.float32)
    win_sq = win ** 2
    
    padded = np.pad(sig_8k, (hop_samples, hop_samples), mode='reflect')
    pad_len = len(padded)
    total_len = len(sig_8k)
    
    out_s1 = np.zeros(pad_len, dtype=np.float32)
    out_s2 = np.zeros(pad_len, dtype=np.float32)
    weight_buf = np.zeros(pad_len, dtype=np.float32)
    
    prev_s1, prev_s2 = None, None
    with torch.no_grad():
        for start in range(0, pad_len - chunk_samples + 1, hop_samples):
            chunk = padded[start:start + chunk_samples]
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE).float()
            est = model(chunk_t)
            c_s1 = est[0, 0].cpu().numpy()
            c_s2 = est[0, 1].cpu().numpy()
            
            # Cross-chunk correlation alignment
            if prev_s1 is not None:
                r1, r2 = prev_s1[hop_samples:], prev_s2[hop_samples:]
                u1, u2 = c_s1[:hop_samples], c_s2[:hop_samples]
                corr_keep = np.dot(u1, r1) + np.dot(u2, r2)
                corr_swap = np.dot(u1, r2) + np.dot(u2, r1)
                if corr_swap > corr_keep:
                    c_s1, c_s2 = c_s2, c_s1
                    
            prev_s1, prev_s2 = c_s1, c_s2
            out_s1[start:start + chunk_samples] += c_s1 * win_sq
            out_s2[start:start + chunk_samples] += c_s2 * win_sq
            weight_buf[start:start + chunk_samples] += win_sq
            
    norm = np.maximum(weight_buf[hop_samples:hop_samples + total_len], 1e-6)
    s1 = out_s1[hop_samples:hop_samples + total_len] / norm
    s2 = out_s2[hop_samples:hop_samples + total_len] / norm
    
    # Wiener Post-Filter
    n_fft, hop = 512, 128
    _, _, X = scipy.signal.stft(sig_8k, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, S1 = scipy.signal.stft(s1, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, S2 = scipy.signal.stft(s2, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    p1 = np.abs(S1)**2
    p2 = np.abs(S2)**2
    tot = p1 + p2 + 1e-8
    _, s1_clean = scipy.signal.istft((p1 / tot) * X, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    _, s2_clean = scipy.signal.istft((p2 / tot) * X, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    
    return s1_clean[:total_len], s2_clean[:total_len]

def run_gpu_enhancement_core(sig_8k, model):
    chunk_samples = 24000 # 3 sec
    hop_samples = 12000
    win = np.sin((np.arange(chunk_samples) + 0.5) / chunk_samples * np.pi).astype(np.float32)
    win_sq = win ** 2
    
    padded = np.pad(sig_8k, (hop_samples, hop_samples), mode='reflect')
    pad_len = len(padded)
    total_len = len(sig_8k)
    out_clean = np.zeros(pad_len, dtype=np.float32)
    weight_buf = np.zeros(pad_len, dtype=np.float32)
    
    with torch.no_grad():
        for start in range(0, pad_len - chunk_samples + 1, hop_samples):
            chunk = padded[start:start + chunk_samples]
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE).float()
            clean_chunk = model(chunk_t)[0].cpu().numpy()
            out_clean[start:start + chunk_samples] += clean_chunk * win_sq
            weight_buf[start:start + chunk_samples] += win_sq
            
    norm = np.maximum(weight_buf[hop_samples:hop_samples + total_len], 1e-6)
    enhanced = out_clean[hop_samples:hop_samples + total_len] / norm
    
    # Wiener Filter
    n_fft, hop = 512, 128
    _, _, X = scipy.signal.stft(sig_8k, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, S = scipy.signal.stft(enhanced, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    N = X - S
    W = np.abs(S)**2 / (np.abs(S)**2 + np.abs(N)**2 + 1e-8)
    _, s_post = scipy.signal.istft(W * X, fs=8000, nperseg=n_fft, noverlap=n_fft - hop)
    return s_post[:total_len]

# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/api/system-status")
def get_system_status():
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (AVX2 Mode)"
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda_avail else 0.0
    vram_used = round(torch.cuda.memory_allocated() / 1e9, 2) if cuda_avail else 0.0
    
    return {
        "cuda_available": cuda_avail,
        "device_name": dev_name,
        "vram_total_gb": vram_total,
        "vram_used_gb": vram_used,
        "models_cached": {
            "separation": MODELS["sep"] is not None,
            "enhancement": MODELS["enh"] is not None,
            "whisper": MODELS["whisper"] is not None and MODELS["whisper"] is not False
        }
    }

@app.get("/api/samples")
def list_demo_samples():
    sample_items = []
    
    # Separation samples
    s_dir = os.path.join(SEP_DIR, "samples")
    if os.path.exists(s_dir):
        for f in sorted(os.listdir(s_dir)):
            if f.endswith(".wav") and "Mixture" in f:
                sample_items.append({
                    "id": f"sep_{f}",
                    "name": f"👥 Separation: {f}",
                    "path": os.path.join(s_dir, f),
                    "type": "SEPARATION"
                })
                
    # Enhancement samples
    e_dir = os.path.join(ENH_DIR, "benchmark_25_tests", "audio")
    if not os.path.exists(e_dir):
        e_dir = os.path.join(BASE_DIR, "audio", "tests")
    if os.path.exists(e_dir):
        for f in sorted(os.listdir(e_dir))[:5]:
            if "mixture" in f and f.endswith(".wav"):
                sample_items.append({
                    "id": f"enh_{f}",
                    "name": f"🎧 Denoising: {f}",
                    "path": os.path.join(e_dir, f),
                    "type": "ENHANCEMENT"
                })
                
    return sample_items

@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    mode: str = Form("AUTO"), # AUTO, SEPARATION, ENHANCEMENT
    enable_asr: bool = Form(True)
):
    t_start = time.time()
    
    # 1. Resolve Audio Input
    raw_audio, sr = None, 8000
    file_name = "audio.wav"
    
    if file is not None:
        file_name = file.filename
        content = await file.read()
        buf = io.BytesIO(content)
        raw_audio, sr = sf.read(buf)
    elif sample_id:
        samples = list_demo_samples()
        match = next((s for s in samples if s["id"] == sample_id), None)
        if match and os.path.exists(match["path"]):
            raw_audio, sr = sf.read(match["path"])
            file_name = match["name"]
        else:
            raise HTTPException(status_code=400, detail="Invalid sample_id specified")
    else:
        raise HTTPException(status_code=400, detail="No audio file or sample provided")
        
    if len(raw_audio.shape) > 1:
        raw_audio = np.mean(raw_audio, axis=1)
        
    sig_8k = resample_to_8k(raw_audio, sr)
    dur_sec = len(sig_8k) / 8000.0
    
    if dur_sec > 310.0:
        raise HTTPException(status_code=400, detail="Audio duration exceeds 5.0 minutes limit.")
        
    # 2. Determine Task (Auto or Manual)
    selected_task = mode
    classification_meta = None
    
    if mode == "AUTO":
        try:
            from task_classifier import classify_audio_task
            temp_in = os.path.join(OUTPUT_DIR, "temp_in.wav")
            sf.write(temp_in, sig_8k, 8000)
            clf = classify_audio_task(temp_in)
            selected_task = clf["task"]
            classification_meta = clf
        except Exception:
            selected_task = "SEPARATION"
            
    # 3. Execute Neural Pipeline
    t_infer_start = time.time()
    response_data = {
        "input_filename": file_name,
        "duration_seconds": round(dur_sec, 2),
        "task_executed": selected_task,
        "classification": classification_meta,
        "outputs": {},
        "dialogue": [],
        "srt_content": ""
    }
    
    if selected_task == "SEPARATION":
        model = get_separation_model()
        s1, s2 = run_gpu_separation_core(sig_8k, model)
        t_infer_end = time.time()
        
        # Save output WAVs
        s1_name = f"sep_s1_{int(time.time()*1000)}.wav"
        s2_name = f"sep_s2_{int(time.time()*1000)}.wav"
        s1_path = os.path.join(OUTPUT_DIR, s1_name)
        s2_path = os.path.join(OUTPUT_DIR, s2_name)
        sf.write(s1_path, s1, 8000)
        sf.write(s2_path, s2, 8000)
        
        response_data["outputs"]["speaker1_url"] = f"/api/audio/{s1_name}"
        response_data["outputs"]["speaker2_url"] = f"/api/audio/{s2_name}"
        
        # Diarized Whisper ASR
        if enable_asr:
            try:
                from transcribe_engine import transcribe_separated_audio
                asr = transcribe_separated_audio(s1_path, s2_path)
                response_data["dialogue"] = asr["events"]
                response_data["srt_content"] = asr["srt"]
            except Exception as e:
                response_data["asr_error"] = str(e)
                
    else: # ENHANCEMENT
        model = get_enhancement_model()
        clean = run_gpu_enhancement_core(sig_8k, model)
        t_infer_end = time.time()
        
        enh_name = f"enh_clean_{int(time.time()*1000)}.wav"
        enh_path = os.path.join(OUTPUT_DIR, enh_name)
        sf.write(enh_path, clean, 8000)
        
        response_data["outputs"]["enhanced_url"] = f"/api/audio/{enh_name}"
        
        if enable_asr:
            try:
                from transcribe_engine import transcribe_enhanced_audio
                asr = transcribe_enhanced_audio(enh_path)
                response_data["dialogue"] = asr["events"]
                response_data["srt_content"] = asr["srt"]
            except Exception as e:
                response_data["asr_error"] = str(e)
                
    t_end = time.time()
    total_elapsed = t_end - t_start
    infer_elapsed = t_infer_end - t_infer_start
    
    response_data["latency"] = {
        "total_seconds": round(total_elapsed, 3),
        "neural_gpu_seconds": round(infer_elapsed, 3),
        "real_time_factor": round(total_elapsed / dur_sec, 3),
        "gpu_speedup": f"{round(dur_sec / max(infer_elapsed, 0.001), 1)}x Real-Time"
    }
    
    return response_data

@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

# -----------------------------------------------------------------------------
# SERVE REACT FRONTEND
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_react_app():
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ClearParty AI Backend is Running. React UI bundle not yet built.</h1>"

if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="ClearParty Fast API Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)), help="Port to run on (default 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    # Function to check if a port is in use and find the next open port
    def get_open_port(start_port):
        p = start_port
        for _ in range(20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
            p += 1
        return start_port

    target_port = get_open_port(args.port)

    # Warm up models on startup for instant zero-latency responses
    get_separation_model()
    get_enhancement_model()

    print(f"\n🚀 ClearParty Ultra-Low-Latency API & React UI starting on: http://localhost:{target_port}")
    if target_port != args.port:
        print(f"ℹ️ (Port {args.port} was busy, automatically switched to open port {target_port})")

    uvicorn.run(app, host=args.host, port=target_port)
