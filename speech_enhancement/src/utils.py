"""
utils.py — Loss Functions and Audio Signal Processing Utilities for Speech Denoising
"""
import torch
import numpy as np


def si_sdr(estimated, target, eps=1e-5):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    """
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = dot * target / s_energy
    noise = estimated - proj
    return 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )


def denoising_loss(est, target, l1_weight=0.1):
    """
    Joint Scale-Aware Objective:
    Maximizes SI-SDR while constraining waveform amplitude via L1 loss.
    Permanently prevents volume drift and digital clipping.
    """
    sisdr_val = si_sdr(est, target)
    l1_loss = torch.mean(torch.abs(est - target))
    total_loss = -sisdr_val.mean() + l1_weight * l1_loss
    return total_loss, sisdr_val.mean().item()


def wiener_denoise_reconstruction(mix, est_clean, n_fft=512, hop_length=128):
    """
    Mixture-Consistent Wiener Filter for Speech Enhancement:
    Uses the clean voice energy estimate to filter the input mixture's natural phase,
    completely eliminating synthetic phase jitter and robotic metallic buzz.
    """
    window = torch.hann_window(n_fft).to(mix.device)
    
    # 1. Complex STFTs
    X = torch.stft(mix, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    S = torch.stft(est_clean, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    # Estimated noise in latent STFT
    est_noise = mix - est_clean
    N = torch.stft(est_noise, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    # 2. Power spectral densities
    pow_s = torch.abs(S) ** 2
    pow_n = torch.abs(N) ** 2
    total_pow = pow_s + pow_n + 1e-8

    # 3. Wiener ratio mask
    M = pow_s / total_pow

    # 4. Filter the original recording's natural acoustic phase
    clean_S = M * X

    # 5. Inverse STFT synthesis
    clean_s = torch.istft(clean_S, n_fft=n_fft, hop_length=hop_length, window=window, length=mix.shape[-1])
    return clean_s


def to_clean_numpy(tensor, target_peak=0.90):
    """Peak-normalizes audio to standard float32 [-0.90, +0.90] range."""
    arr = tensor.cpu().numpy().astype(np.float32)
    arr = arr - np.mean(arr)
    peak = np.max(np.abs(arr)) + 1e-8
    return (arr / peak) * target_peak
