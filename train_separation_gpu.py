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

class SimpleAudioMixtureDataset(Dataset):
    """
    Dataset that loads mixtures and targets from a Libri2Mix directory,
    or falls back to dynamic sample pairing.
    """
    def __init__(self, root_dir=None, segment_len=SEGMENT_LEN):
        self.segment_len = segment_len
        self.pairs = []
        
        # Check standard Libri2Mix structure: <root>/mix_clean, <root>/s1, <root>/s2
        if root_dir and os.path.exists(os.path.join(root_dir, "mix_clean")):
            mix_dir = os.path.join(root_dir, "mix_clean")
            s1_dir = os.path.join(root_dir, "s1")
            s2_dir = os.path.join(root_dir, "s2")
            files = [f for f in os.listdir(mix_dir) if f.endswith(".wav")]
            for f in files:
                p_mix = os.path.join(mix_dir, f)
                p_s1 = os.path.join(s1_dir, f)
                p_s2 = os.path.join(s2_dir, f)
                if os.path.exists(p_s1) and os.path.exists(p_s2):
                    self.pairs.append((p_mix, p_s1, p_s2))
                    
        # Fallback to local evaluation samples for demonstration/smoke-testing
        if len(self.pairs) == 0:
            sample_dir = os.path.join(SEP_DIR, "samples")
            candidates = [
                os.path.join(sample_dir, "Input Mixture (Both speakers talking).wav"),
                os.path.join(sample_dir, "final_pairs", "pair_01_mixture.wav")
            ]
            for c in candidates:
                if os.path.exists(c):
                    # Use sample as both mix and targets for pipeline validation
                    self.pairs.append((c, c, c))
                    
        print(f"Loaded {len(self.pairs)} training instances.")

    def __len__(self):
        return max(len(self.pairs), 24) # Ensure at least 1 full batch

    def __getitem__(self, idx):
        p_mix, p_s1, p_s2 = self.pairs[idx % len(self.pairs)]
        mix, _ = sf.read(p_mix)
        s1, _ = sf.read(p_s1)
        s2, _ = sf.read(p_s2)
        
        # Flatten multi-channel
        if len(mix.shape) > 1: mix = np.mean(mix, axis=1)
        if len(s1.shape) > 1: s1 = np.mean(s1, axis=1)
        if len(s2.shape) > 1: s2 = np.mean(s2, axis=1)
        
        # Pad or chunk to exact segment length
        def pad_chunk(sig):
            if len(sig) < self.segment_len:
                return np.pad(sig, (0, self.segment_len - len(sig)))
            return sig[:self.segment_len]
            
        mix = pad_chunk(mix).astype(np.float32)
        s1 = pad_chunk(s1).astype(np.float32)
        s2 = pad_chunk(s2).astype(np.float32)
        
        target = np.stack([s1, s2], axis=0) # (2, segment_len)
        return torch.from_numpy(mix), torch.from_numpy(target)

def run_fine_tuning(data_dir=None, epochs=5):
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Initializing Speaker Separation Fine-Tuning on: {dev_name}")
    print(f"⚙️ Hyperparameters: Batch Size = {BATCH_SIZE} | Accumulation = {ACCUMULATION_STEPS} (Eff. Batch: {BATCH_SIZE*ACCUMULATION_STEPS})")
    
    model = ConvTasNet().to(DEVICE)
    ckpt_path = os.path.join(SEP_DIR, "checkpoints", "tasnet_v2_epoch50.pth")
    
    if os.path.exists(ckpt_path):
        print(f"Loading base checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    else:
        print(f"Checkpoint not found at {ckpt_path}. Initializing randomly.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))
    
    dataset = SimpleAudioMixtureDataset(root_dir=data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    
    os.makedirs("checkpoints_gpu_v2", exist_ok=True)
    print("\nStarting Training Loop...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()
        
        for step, (mix, tgt) in enumerate(loader):
            mix = mix.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                est = model(mix) # (B, 2, Samples)
                loss = upit_loss(est, tgt) / ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            total_loss += loss.item() * ACCUMULATION_STEPS
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        elapsed = time.time() - t0
        avg_loss = total_loss / len(loader)
        vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Mean uPIT SI-SDR Loss: {avg_loss:+.2f} dB | Time: {elapsed:.2f}s | VRAM: {vram_mb:.1f} MB")
        
        # Save checkpoint
        save_path = f"checkpoints_gpu_v2/tasnet_epoch_{epoch:02d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, save_path)
        print(f"  💾 Saved Checkpoint -> {save_path}")
        
    print("\n🎉 Fine-Tuning Execution Finished Successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Conv-TasNet on GPU")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to Libri2Mix dataset split")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    args = parser.parse_args()
    run_fine_tuning(data_dir=args.data_dir, epochs=args.epochs)
