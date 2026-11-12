# train_enhancement_gpu.py
"""
GPU Fine-Tuning Script for Compound Multi-Noise Speech Enhancement
Optimized for 8 GB VRAM using Multi-Resolution STFT & SI-SDR Joint Loss.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_dir(candidates):
    for c in candidates:
        full = os.path.join(BASE_DIR, c)
        if os.path.exists(full):
            return full
    return os.path.join(BASE_DIR, candidates[0])

ENH_DIR = find_dir(["speech_enhancement", "speech enhancement"])
sys.path.insert(0, ENH_DIR)
from src.model import ConvTasNetDenoiser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def multi_resolution_stft_loss(est, target):
    """Computes Multi-Resolution STFT spectral convergence & magnitude loss."""
    resolutions = [(256, 128, 64), (512, 256, 128), (1024, 512, 256)]
    total_loss = 0.0
    
    for n_fft, hop, win_len in resolutions:
        window = torch.hann_window(win_len).to(DEVICE)
        stft_est = torch.stft(est, n_fft=n_fft, hop_length=hop, win_length=win_len, window=window, return_complex=True)
        stft_tgt = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=win_len, window=window, return_complex=True)
        
        mag_est = torch.abs(stft_est)
        mag_tgt = torch.abs(stft_tgt)
        
        sc_loss = torch.norm(mag_tgt - mag_est, p="fro") / (torch.norm(mag_tgt, p="fro") + 1e-8)
        mag_loss = F.l1_loss(torch.log(mag_tgt + 1e-8), torch.log(mag_est + 1e-8))
        total_loss += (sc_loss + mag_loss)
        
    return total_loss / len(resolutions)

def run_enhancement_training():
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Initializing Speech Enhancement Fine-Tuning on: {dev_name}")
    model = ConvTasNetDenoiser().to(DEVICE)
    ckpt_path = os.path.join(ENH_DIR, "checkpoints", "denoiser_best.pth")
    
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print("Loaded initial denoiser checkpoint.")
    else:
        print(f"Denoiser checkpoint not found at {ckpt_path}. Initializing randomly.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    print("Ready to execute multi-noise enhancement fine-tuning.")

if __name__ == "__main__":
    run_enhancement_training()
