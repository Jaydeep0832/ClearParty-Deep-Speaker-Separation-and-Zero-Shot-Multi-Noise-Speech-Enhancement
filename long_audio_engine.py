# long_audio_engine.py
"""
5-Minute Long Audio Engine with 50% Overlap-Add (OLA)
and Cross-Chunk Permutation Alignment on 8 GB GPU.
Supports both Monaural Speaker Separation and Speech Enhancement.
"""

import os
import sys
import time
import torch
import numpy as np
import soundfile as sf
import scipy.signal

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

def process_5min_separation(input_path, output_dir="separated_outputs", chunk_sec=4.0, sr=8000):
    """
    Executes 50% Overlap-Add separation with Cross-Chunk Permutation Alignment on GPU.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_sig, in_sr = sf.read(input_path)
    if len(raw_sig.shape) > 1:
        raw_sig = np.mean(raw_sig, axis=1)
        
    if in_sr != sr:
        gcd = np.gcd(in_sr, sr)
        raw_sig = scipy.signal.resample_poly(raw_sig, sr // gcd, in_sr // gcd).astype(np.float32)
        
    total_len = len(raw_sig)
    dur_sec = total_len / sr
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n📂 Processing Long Audio: {input_path}")
    print(f"⏱️ Duration: {dur_sec:.2f} s ({dur_sec/60:.2f} mins) | Device: {dev_name}")
    
    from model import ConvTasNet
    model = ConvTasNet().to(DEVICE)
    ckpt_path = os.path.join(SEP_DIR, "checkpoints", "tasnet_v2_epoch50.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    
    chunk_samples = int(chunk_sec * sr)     # 32,000 samples (4 sec)
    hop_samples = chunk_samples // 2        # 16,000 samples (50% overlap)
    
    win = np.sin((np.arange(chunk_samples) + 0.5) / chunk_samples * np.pi).astype(np.float32)
    win_sq = win ** 2
    
    padded = np.pad(raw_sig, (hop_samples, hop_samples), mode='reflect')
    pad_len = len(padded)
    
    out_s1 = np.zeros(pad_len, dtype=np.float32)
    out_s2 = np.zeros(pad_len, dtype=np.float32)
    weight_buf = np.zeros(pad_len, dtype=np.float32)
    
    prev_s1, prev_s2 = None, None
    t0 = time.time()
    
    with torch.no_grad():
        for start in range(0, pad_len - chunk_samples + 1, hop_samples):
            chunk = padded[start:start + chunk_samples]
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE).float()
            
            est = model(chunk_t)
            c_s1 = est[0, 0].cpu().numpy()
            c_s2 = est[0, 1].cpu().numpy()
            
            if prev_s1 is not None:
                overlap_ref1 = prev_s1[hop_samples:]
                overlap_ref2 = prev_s2[hop_samples:]
                overlap_cur1 = c_s1[:hop_samples]
                overlap_cur2 = c_s2[:hop_samples]
                
                corr_keep = np.dot(overlap_cur1, overlap_ref1) + np.dot(overlap_cur2, overlap_ref2)
                corr_swap = np.dot(overlap_cur1, overlap_ref2) + np.dot(overlap_cur2, overlap_ref1)
                
                if corr_swap > corr_keep:
                    c_s1, c_s2 = c_s2, c_s1
                    
            prev_s1, prev_s2 = c_s1, c_s2
            
            out_s1[start:start + chunk_samples] += c_s1 * win_sq
            out_s2[start:start + chunk_samples] += c_s2 * win_sq
            weight_buf[start:start + chunk_samples] += win_sq
            
    norm = np.maximum(weight_buf[hop_samples:hop_samples + total_len], 1e-6)
    s1_sep = out_s1[hop_samples:hop_samples + total_len] / norm
    s2_sep = out_s2[hop_samples:hop_samples + total_len] / norm
    t1 = time.time()
    
    rtf = (t1 - t0) / dur_sec
    print(f"✅ Neural Separation complete in {t1-t0:.2f}s (RTF: {rtf:.3f}x)")
    
    # Apply Mixture-Consistent Wiener Post-Filter
    n_fft, hop = 512, 128
    _, _, X = scipy.signal.stft(raw_sig, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, S1 = scipy.signal.stft(s1_sep, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, S2 = scipy.signal.stft(s2_sep, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    pow1 = np.abs(S1)**2
    pow2 = np.abs(S2)**2
    tot = pow1 + pow2 + 1e-8
    
    _, s1_clean = scipy.signal.istft((pow1 / tot) * X, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, s2_clean = scipy.signal.istft((pow2 / tot) * X, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    s1_path = os.path.join(output_dir, "speaker1_separated.wav")
    s2_path = os.path.join(output_dir, "speaker2_separated.wav")
    sf.write(s1_path, s1_clean[:total_len], sr)
    sf.write(s2_path, s2_clean[:total_len], sr)
    
    return {
        's1_path': s1_path,
        's2_path': s2_path,
        'duration_sec': dur_sec,
        'elapsed_sec': t1 - t0,
        'rtf': rtf
    }

def process_5min_enhancement(input_path, output_dir="enhanced_outputs", chunk_sec=3.0, sr=8000):
    """
    Executes 50% Overlap-Add speech enhancement for long audio up to 5 minutes.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_sig, in_sr = sf.read(input_path)
    if len(raw_sig.shape) > 1:
        raw_sig = np.mean(raw_sig, axis=1)
        
    if in_sr != sr:
        gcd = np.gcd(in_sr, sr)
        raw_sig = scipy.signal.resample_poly(raw_sig, sr // gcd, in_sr // gcd).astype(np.float32)
        
    total_len = len(raw_sig)
    dur_sec = total_len / sr
    
    from src.model import ConvTasNetDenoiser
    model = ConvTasNetDenoiser().to(DEVICE)
    ckpt_path = os.path.join(ENH_DIR, "checkpoints", "denoiser_best.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    
    chunk_samples = int(chunk_sec * sr)
    hop_samples = chunk_samples // 2
    win = np.sin((np.arange(chunk_samples) + 0.5) / chunk_samples * np.pi).astype(np.float32)
    win_sq = win ** 2
    
    padded = np.pad(raw_sig, (hop_samples, hop_samples), mode='reflect')
    out_clean = np.zeros(len(padded), dtype=np.float32)
    weight_buf = np.zeros(len(padded), dtype=np.float32)
    
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(padded) - chunk_samples + 1, hop_samples):
            chunk = padded[start:start + chunk_samples]
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE).float()
            clean_chunk = model(chunk_t)[0].cpu().numpy()
            
            out_clean[start:start + chunk_samples] += clean_chunk * win_sq
            weight_buf[start:start + chunk_samples] += win_sq
            
    norm = np.maximum(weight_buf[hop_samples:hop_samples + total_len], 1e-6)
    enhanced = out_clean[hop_samples:hop_samples + total_len] / norm
    t1 = time.time()
    
    n_fft, hop = 512, 128
    _, _, X = scipy.signal.stft(raw_sig, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, S = scipy.signal.stft(enhanced, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    N = X - S
    W = np.abs(S)**2 / (np.abs(S)**2 + np.abs(N)**2 + 1e-8)
    _, clean_post = scipy.signal.istft(W * X, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    out_path = os.path.join(output_dir, "clean_enhanced.wav")
    sf.write(out_path, clean_post[:total_len], sr)
    
    return {
        'enhanced_path': out_path,
        'duration_sec': dur_sec,
        'elapsed_sec': t1 - t0,
        'rtf': (t1 - t0) / dur_sec
    }
