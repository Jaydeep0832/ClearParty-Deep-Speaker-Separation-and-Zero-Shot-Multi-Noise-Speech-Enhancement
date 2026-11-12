# train_separation_gpu.py
"""
ClearParty: GPU Fine-Tuning Script for Monaural Two-Speaker Separation
Configured for NVIDIA 8 GB VRAM with Automatic Mixed Precision (AMP).
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_dir(candidates):
    for c in candidates:
        full = os.path.join(BASE_DIR, c)
        if os.path.exists(full):
            return full
    return os.path.join(BASE_DIR, candidates[0])

SEP_DIR = find_dir(["speaker_separation", "speaker separation"])
sys.path.insert(0, os.path.join(SEP_DIR, "src"))
from model import ConvTasNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8 GB VRAM Configuration
BATCH_SIZE = 6             # Physical batch size fitted for 8 GB VRAM
ACCUMULATION_STEPS = 4     # Effective batch size = 6 * 4 = 24
LEARNING_RATE = 3e-4
MAX_EPOCHS = 50
SAMPLE_RATE = 8000
SEGMENT_LEN = 32000        # 4.0 seconds @ 8 kHz

def compute_sisdr(est, tgt, eps=1e-8):
    """Calculates Scale-Invariant Signal-to-Distortion Ratio."""
    tgt_energy = torch.sum(tgt ** 2, dim=-1, keepdim=True) + eps
    proj = (torch.sum(est * tgt, dim=-1, keepdim=True) / tgt_energy) * tgt
    noise = est - proj
    ratio = (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)

def upit_loss(est_sources, tgt_sources):
    """
    Computes permutation-invariant negative SI-SDR.
    est_sources: (Batch, 2, Samples)
    tgt_sources: (Batch, 2, Samples)
    """
    sdr1_p1 = compute_sisdr(est_sources[:, 0, :], tgt_sources[:, 0, :])
    sdr2_p1 = compute_sisdr(est_sources[:, 1, :], tgt_sources[:, 1, :])
    loss_p1 = -(sdr1_p1 + sdr2_p1) / 2.0
    
    sdr1_p2 = compute_sisdr(est_sources[:, 0, :], tgt_sources[:, 1, :])
    sdr2_p2 = compute_sisdr(est_sources[:, 1, :], tgt_sources[:, 0, :])
    loss_p2 = -(sdr1_p2 + sdr2_p2) / 2.0
    
    best_loss = torch.min(loss_p1, loss_p2)
    return torch.mean(best_loss)

def run_fine_tuning():
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Initializing Speaker Separation Fine-Tuning on: {dev_name}")
    
    model = ConvTasNet().to(DEVICE)
    ckpt_path = os.path.join(SEP_DIR, "checkpoints", "tasnet_v2_epoch50.pth")
    
    if os.path.exists(ckpt_path):
        print(f"Loading base checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    else:
        print(f"Checkpoint not found at {ckpt_path}. Initializing randomly.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    
    os.makedirs("checkpoints_gpu_v2", exist_ok=True)
    print("Fine-tuning pipeline configured successfully. Ready for dataset iteration.")

if __name__ == "__main__":
    run_fine_tuning()
