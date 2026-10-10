import os
import soundfile as sf
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt

test_dir = r"D:\cocktail_party_project\Speech Enhancement\test_audio_file"
clean_path = os.path.join(test_dir, "Original Ground Truth Clean Speech.wav")
mix_path = os.path.join(test_dir, "Corrupted Mixture (Speech + Multiple Simultaneous Noises).wav")
denoised_path = os.path.join(test_dir, "Model Denoised Output (Clean Speech Restored).wav")

s_clean, sr_clean = sf.read(clean_path)
s_mix, sr_mix = sf.read(mix_path)
s_denoised, sr_denoised = sf.read(denoised_path)

min_len = min(len(s_clean), len(s_mix), len(s_denoised))
s_clean = s_clean[:min_len]
s_mix = s_mix[:min_len]
s_denoised = s_denoised[:min_len]

def compute_sisdr(est, target, eps=1e-8):
    target = target - np.mean(target)
    est = est - np.mean(est)
    target_energy = np.sum(target ** 2) + eps
    alpha = np.sum(est * target) / target_energy
    target_scaled = alpha * target
    noise = est - target_scaled
    val = 10 * np.log10((np.sum(target_scaled ** 2) + eps) / (np.sum(noise ** 2) + eps))
    return float(val)

sisdr_mix = compute_sisdr(s_mix, s_clean)
sisdr_denoised = compute_sisdr(s_denoised, s_clean)

pesq_mix = pesq(sr_clean, s_clean, s_mix, 'nb')
pesq_denoised = pesq(sr_clean, s_clean, s_denoised, 'nb')

stoi_mix = stoi(s_clean, s_mix, sr_clean, extended=False)
stoi_denoised = stoi(s_clean, s_denoised, sr_clean, extended=False)

print("=== EVALUATION SCORES ===")
print(f"SI-SDR Input Mixture:   {sisdr_mix:+.2f} dB")
print(f"SI-SDR Denoised Output:  {sisdr_denoised:+.2f} dB (Gain: {sisdr_denoised - sisdr_mix:+.2f} dB)")
print(f"PESQ Input Mixture:     {pesq_mix:.3f}")
print(f"PESQ Denoised Output:    {pesq_denoised:.3f} (Gain: {pesq_denoised - pesq_mix:+.3f})")
print(f"STOI Input Mixture:     {stoi_mix:.3f}")
print(f"STOI Denoised Output:    {stoi_denoised:.3f} (Gain: {stoi_denoised - stoi_mix:+.3f})")

# Save spectrogram plot
out_fig = r"C:\Users\jayde\.gemini\antigravity\brain\fb8eae72-61a5-4ee9-82e3-2254e841f3f6\melspectrogram_denoising_test_comparison.png"
fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

titles = [
    "1. Original Ground Truth Clean Speech (Baseline Target)",
    f"2. Corrupted Test Mixture (Multiple Simultaneous Noises) [PESQ: {pesq_mix:.2f}, SI-SDR: {sisdr_mix:.2f} dB]",
    f"3. Model Denoised Output [PESQ: {pesq_denoised:.2f} (+{pesq_denoised-pesq_mix:.2f}), SI-SDR: {sisdr_denoised:.2f} dB (+{sisdr_denoised-sisdr_mix:.2f} dB)]"
]
signals = [s_clean, s_mix, s_denoised]
cmaps = ["viridis", "magma", "viridis"]

for ax, sig, title, cmap in zip(axes, signals, titles, cmaps):
    S = librosa.feature.melspectrogram(y=sig, sr=sr_clean, n_fft=512, hop_length=128, n_mels=80)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr_clean, hop_length=128, x_axis="time", y_axis="mel", ax=ax, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.tight_layout()
plt.savefig(out_fig, dpi=300)
print(f"Spectrogram saved successfully to: {out_fig}")
