# evaluate_gpu.py
"""Computes SI-SDR, PESQ, and STOI metrics."""
import soundfile as sf
import numpy as np
from pystoi import stoi
from pesq import pesq

def score_file(clean_path, est_path, sr=8000):
    clean, _ = sf.read(clean_path)
    est, _ = sf.read(est_path)
    min_len = min(len(clean), len(est))
    clean, est = clean[:min_len], est[:min_len]
    
    target_energy = np.sum(clean ** 2) + 1e-8
    scaling = np.sum(est * clean) / target_energy
    target_proj = scaling * clean
    noise = est - target_proj
    sisdr = 10.0 * np.log10(np.sum(target_proj ** 2) / (np.sum(noise ** 2) + 1e-8))
    
    stoi_val = stoi(clean, est, sr, extended=False)
    pesq_val = pesq(sr, clean, est, 'nb')
    
    print(f"\n📊 Evaluation Results for: {est_path}")
    print(f"  • SI-SDR Gain : {sisdr:+.2f} dB")
    print(f"  • PESQ (MOS)  : {pesq_val:.3f}")
    print(f"  • STOI Score  : {stoi_val:.3f}")

if __name__ == "__main__":
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    candidates_ref = [
        os.path.join(BASE, "speaker_separation", "samples", "True Speaker 1 (Clean Reference).wav"),
        os.path.join(BASE, "speaker separation", "samples", "True Speaker 1 (Clean Reference).wav"),
        os.path.join(BASE, "audio", "separation", "True Speaker 1 (Clean Reference).wav")
    ]
    candidates_out = [
        os.path.join(BASE, "speaker_separation", "samples", "Natural Wiener-Reconstructed Speaker 1 (Natural human voice).wav"),
        os.path.join(BASE, "speaker separation", "samples", "Natural Wiener-Reconstructed Speaker 1 (Natural human voice).wav"),
        os.path.join(BASE, "audio", "separation", "Natural Wiener-Reconstructed Speaker 1 (Natural human voice).wav")
    ]
    ref = next((c for c in candidates_ref if os.path.exists(c)), None)
    out = next((c for c in candidates_out if os.path.exists(c)), None)
    if ref and out:
        score_file(ref, out)
    else:
        print("Evaluation files not found in standard sample paths.")
