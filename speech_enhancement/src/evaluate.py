"""
evaluate.py — Multi-Noise Stress Testing & Benchmark Suite
Benchmarks:
  1. Single-Noise Condition (Standard Denoising)
  2. Dual-Noise Compound Condition (2 Simultaneous Noises)
  3. Triple-Noise Compound Condition (3 Simultaneous Noises)
Metrics: SI-SDR Improvement (ΔdB), PESQ (ITU-T P.862), STOI
"""
import os
import argparse
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import NOIZEUSEnhancementDataset
from src.model import ConvTasNetDenoiser
from src.utils import si_sdr, wiener_denoise_reconstruction, to_clean_numpy

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False


def compute_pesq_safe(ref, deg, sr=8000):
    if not HAS_PESQ:
        return None
    try:
        return pesq(sr, ref.astype(np.float32), deg.astype(np.float32), 'nb')
    except Exception:
        return None


def compute_stoi_safe(ref, deg, sr=8000):
    if not HAS_STOI:
        return None
    try:
        return stoi(ref.astype(np.float32), deg.astype(np.float32), sr, extended=False)
    except Exception:
        return None


def evaluate_condition(model, dataset, desc="Evaluation"):
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    sr = Config.SAMPLE_RATE

    in_sdrs, out_sdrs, gains = [], [], []
    in_pesqs, out_pesqs = [], []
    in_stois, out_stois = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                mix, clean, _ = batch
            else:
                mix, clean = batch

            mix = mix.to(Config.DEVICE)
            clean = clean.to(Config.DEVICE)

            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                raw_est = model(mix)

            B = mix.shape[0]
            for b in range(B):
                m_t = mix[b].float()
                c_t = clean[b].float()
                r_t = raw_est[b].float()

                # Clean natural Wiener reconstruction (Zero robotic artifacts)
                enhanced = wiener_denoise_reconstruction(m_t, r_t)

                # SI-SDR
                base_sdr = si_sdr(m_t, c_t).item()
                est_sdr  = si_sdr(enhanced, c_t).item()
                in_sdrs.append(base_sdr)
                out_sdrs.append(est_sdr)
                gains.append(est_sdr - base_sdr)

                # Numpy conversion
                m_np = to_clean_numpy(m_t)
                c_np = to_clean_numpy(c_t)
                e_np = to_clean_numpy(enhanced)

                # PESQ & STOI
                q_in = compute_pesq_safe(c_np, m_np, sr)
                q_out = compute_pesq_safe(c_np, e_np, sr)
                if q_in and q_out:
                    in_pesqs.append(q_in)
                    out_pesqs.append(q_out)

                st_in = compute_stoi_safe(c_np, m_np, sr)
                st_out = compute_stoi_safe(c_np, e_np, sr)
                if st_in and st_out:
                    in_stois.append(st_in)
                    out_stois.append(st_out)

    res = {
        'desc': desc,
        'count': len(gains),
        'in_sdr': np.mean(in_sdrs),
        'out_sdr': np.mean(out_sdrs),
        'gain_sdr': np.mean(gains),
        'in_pesq': np.mean(in_pesqs) if in_pesqs else None,
        'out_pesq': np.mean(out_pesqs) if out_pesqs else None,
        'gain_pesq': (np.mean(out_pesqs) - np.mean(in_pesqs)) if in_pesqs else None,
        'in_stoi': np.mean(in_stois) if in_stois else None,
        'out_stoi': np.mean(out_stois) if out_stois else None,
        'gain_stoi': (np.mean(out_stois) - np.mean(in_stois)) if in_stois else None,
    }
    return res


def run_comprehensive_evaluation(args):
    print("=" * 70)
    print("🧪 SPEECH ENHANCEMENT MULTI-NOISE GENERALIZATION BENCHMARK")
    print(f"   Checkpoint: {args.checkpoint}")
    print("=" * 70)

    model = ConvTasNetDenoiser().to(Config.DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=Config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 1. Single Noise Condition (Standard Training Domain)
    ds_single = NOIZEUSEnhancementDataset(data_root=args.data_dir, split='test_single')
    res_single = evaluate_condition(model, ds_single, desc="1. Single-Noise (Baseline Test)")

    # 2. Dual Noise Compound Condition (2 Simultaneous Noises Stress Test)
    ds_dual = NOIZEUSEnhancementDataset(data_root=args.data_dir, split='test_multi', multi_noise_count=2)
    res_dual = evaluate_condition(model, ds_dual, desc="2. Dual-Noise Compound (2 Simultaneous)")

    # 3. Triple Noise Compound Condition (3 Simultaneous Noises Extreme Stress Test)
    ds_triple = NOIZEUSEnhancementDataset(data_root=args.data_dir, split='test_multi', multi_noise_count=3)
    res_triple = evaluate_condition(model, ds_triple, desc="3. Triple-Noise Compound (3 Simultaneous)")

    # ─── FINAL COMPARATIVE SCORECARD ───
    print("\n" + "=" * 70)
    print("📊 MULTI-NOISE STRESS GENERALIZATION RESULTS TABLE")
    print("=" * 70)
    print(f"{'Condition':<35} | {'Input SI-SDR':<12} | {'Enhanced SI-SDR':<15} | {'Δ Gain':<10} | {'PESQ (In -> Out)'}")
    print("-" * 70)

    for r in [res_single, res_dual, res_triple]:
        p_str = f"{r['in_pesq']:.2f} -> {r['out_pesq']:.2f}" if r['in_pesq'] else "N/A"
        print(f"{r['desc']:<35} | {r['in_sdr']:>+8.2f} dB   | {r['out_sdr']:>+10.2f} dB    | {r['gain_sdr']:>+6.2f} dB | {p_str}")

    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Denoising on Multi-Noise Conditions")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained denoiser checkpoint")
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR, help="Path to extracted NOIZEUS dataset")
    args = parser.parse_args()
    run_comprehensive_evaluation(args)
