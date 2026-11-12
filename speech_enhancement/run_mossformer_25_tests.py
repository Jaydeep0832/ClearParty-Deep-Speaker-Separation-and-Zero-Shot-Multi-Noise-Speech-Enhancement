#!/usr/bin/env python3
"""
Comprehensive 25-Test Compound Multi-Noise Benchmark for MossFormerGAN_SE_16K
ClearerVoice-Studio State-of-the-Art Speech Enhancement Architecture

Evaluates 25 realistic cocktail-party speech mixtures (3-4 simultaneous noises)
across 8 distinct NOIZEUS environments and SNRs (0.0 dB to 10.0 dB).
Measures:
- SI-SDR (dB) Input, Output, and Gain
- PESQ (ITU-T P.862 Narrowband) Input, Output, and Gain
- STOI Input, Output, and Gain
- Background Noise Power Energy Attenuation (%)
- Waveform Pearson Correlation (r)
- Pass / Accuracy Rate
"""

import sys, types, os, json, time, shutil
import numpy as np

# Mock unused dependencies for ClearVoice loading
mock_pydub = types.ModuleType('pydub')
mock_pydub.AudioSegment = object
sys.modules['pydub'] = mock_pydub

mock_yap = types.ModuleType('yamlargparse')
mock_yap.ArgumentParser = object
mock_yap.ActionConfigFile = object
sys.modules['yamlargparse'] = mock_yap

sys.path.append(r'C:\Python\Python311\Lib\site-packages')

import torch
import soundfile as sf
import librosa
from pesq import pesq
import pystoi
import argparse
import pandas as pd
from clearvoice.models.mossformer_gan_se.generator import MossFormerGAN_SE_16K
from clearvoice.utils.decode import decode_one_audio_mossformergan_se_16k

def calc_sisdr(ref, est):
    eps = 1e-8
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    s_target = alpha * ref
    e_noise = est - s_target
    return float(10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + eps)))

def calc_attenuation(clean, mix, enh):
    eps = 1e-8
    in_noise_energy = np.sum((mix - clean) ** 2)
    res_noise_energy = np.sum((enh - clean) ** 2)
    atten = (1.0 - (res_noise_energy / (in_noise_energy + eps))) * 100.0
    return float(max(0.0, min(100.0, atten)))

def calc_pearson(ref, est):
    eps = 1e-8
    ref_z = ref - np.mean(ref)
    est_z = est - np.mean(est)
    r = np.dot(ref_z, est_z) / (np.sqrt(np.dot(ref_z, ref_z) * np.dot(est_z, est_z)) + eps)
    return float(r)

def main():
    print("=" * 80)
    print("STARTING SOTA MOSSFORMER SPEECH ENHANCEMENT 25-TEST BENCHMARK")
    print("=" * 80)

    # 1. Paths
    results_dir = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_mossformer_enhancement"
    audio_out_dir = os.path.join(results_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    prior_manifest_path = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_speech_enhancement\test_manifest.json"
    prior_audio_dir = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_speech_enhancement\audio"
    checkpoint_path = r"D:\cocktail_party_project\Speech_Enhancement\results\MossFormer_GAN_SE_16K\last_best_checkpoint (1).pt"

    with open(prior_manifest_path, 'r') as f:
        manifest = json.load(f)

    # 2. Setup MossFormerGAN Model
    print(f"Loading MossFormerGAN_SE_16K from {checkpoint_path}...")
    t0 = time.time()
    args = argparse.Namespace()
    args.fft_len = 400
    args.win_len = 400
    args.win_inc = 100
    args.sampling_rate = 16000
    args.win_type = 'hamming'
    args.decode_window = 10
    args.one_time_decode_length = 10
    args.mode = 'inference'
    args.network = 'MossFormerGAN_SE_16K'

    model = MossFormerGAN_SE_16K(args)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    print(f"MossFormerGAN model successfully loaded in {time.time() - t0:.2f} seconds!")

    # 3. Process All 25 Tests
    records = []
    updated_manifest = []

    print("\n" + "-" * 80)
    print(f"{'Test':<5} | {'Clean':<8} | {'SNR':<7} | {'Noises':<28} | {'SI-SDR Gain':<11} | {'PESQ Gain':<10} | {'STOI Gain':<10} | {'Status'}")
    print("-" * 80)

    for item in manifest:
        tid = item['test_id']
        c_file = item['clean_file']
        snr = item['target_snr_db']
        k_noises = item['noise_count']
        noises_desc = item['noise_description']
        dur = item['duration_sec']

        src_clean = os.path.join(prior_audio_dir, f"test_{tid:02d}_clean.wav")
        src_mix = os.path.join(prior_audio_dir, f"test_{tid:02d}_mixture_{k_noises}noises.wav")

        dest_clean = os.path.join(audio_out_dir, f"test_{tid:02d}_clean.wav")
        dest_mix = os.path.join(audio_out_dir, f"test_{tid:02d}_mixture_{k_noises}noises.wav")
        dest_enh = os.path.join(audio_out_dir, f"test_{tid:02d}_mossformer_enhanced.wav")

        # Copy clean and mix to keep folder standalone
        shutil.copy2(src_clean, dest_clean)
        shutil.copy2(src_mix, dest_mix)

        # Load audio (8kHz)
        clean_audio, sr_c = sf.read(dest_clean)
        mix_audio, sr_m = sf.read(dest_mix)

        # Resample mix to 16kHz for MossFormer
        mix_16k = librosa.resample(mix_audio, orig_sr=8000, target_sr=16000)
        inputs_16k = np.expand_dims(mix_16k, axis=0)

        # Run MossFormerGAN Inference
        t_infer_start = time.time()
        with torch.no_grad():
            enh_16k = decode_one_audio_mossformergan_se_16k(model.model, torch.device('cpu'), inputs_16k, args)
        t_infer = time.time() - t_infer_start

        # Resample enhanced audio back to 8kHz
        enh_8k = librosa.resample(enh_16k, orig_sr=16000, target_sr=8000)

        # Align length
        min_len = min(len(clean_audio), len(enh_8k), len(mix_audio))
        clean_eval = clean_audio[:min_len]
        mix_eval = mix_audio[:min_len]
        enh_eval = enh_8k[:min_len]

        # Save enhanced audio
        sf.write(dest_enh, enh_eval, 8000)

        # Compute Metrics
        in_pesq = pesq(8000, clean_eval, mix_eval, 'nb')
        out_pesq = pesq(8000, clean_eval, enh_eval, 'nb')
        gain_pesq = out_pesq - in_pesq

        in_stoi = pystoi.stoi(clean_eval, mix_eval, 8000, extended=False)
        out_stoi = pystoi.stoi(clean_eval, enh_eval, 8000, extended=False)
        gain_stoi = out_stoi - in_stoi

        in_sisdr = calc_sisdr(clean_eval, mix_eval)
        out_sisdr = calc_sisdr(clean_eval, enh_eval)
        gain_sisdr = out_sisdr - in_sisdr

        atten_pct = calc_attenuation(clean_eval, mix_eval, enh_eval)
        pearson_r = calc_pearson(clean_eval, enh_eval)

        status = "PASS" if (gain_sisdr > 0 and gain_pesq > 0) else "REVIEW"

        short_desc = noises_desc if len(noises_desc) <= 28 else noises_desc[:25] + "..."
        print(f"#{tid:02d}   | {c_file:<8} | {snr:>4.1f} dB | {short_desc:<28} | {gain_sisdr:>+7.2f} dB  | {gain_pesq:>+6.3f}    | {gain_stoi:>+6.3f}    | {status}")

        rec = {
            "test_id": tid,
            "clean_file": c_file,
            "noise_count": k_noises,
            "noise_types": item['noise_types'],
            "noise_description": noises_desc,
            "target_snr_db": round(float(snr), 1),
            "duration_sec": round(float(dur), 2),
            "input_pesq": round(float(in_pesq), 3),
            "mossformer_pesq": round(float(out_pesq), 3),
            "pesq_gain": round(float(gain_pesq), 3),
            "input_stoi": round(float(in_stoi), 3),
            "mossformer_stoi": round(float(out_stoi), 3),
            "stoi_gain": round(float(gain_stoi), 3),
            "input_sisdr_db": round(float(in_sisdr), 2),
            "mossformer_sisdr_db": round(float(out_sisdr), 2),
            "sisdr_gain_db": round(float(gain_sisdr), 2),
            "noise_attenuation_pct": round(float(atten_pct), 2),
            "pearson_r": round(float(pearson_r), 4),
            "inference_time_sec": round(float(t_infer), 3),
            "status": status
        }
        records.append(rec)

        updated_item = dict(item)
        updated_item['clean_wav'] = dest_clean
        updated_item['mixture_wav'] = dest_mix
        updated_item['enhanced_wav'] = dest_enh
        updated_manifest.append(updated_item)

    print("-" * 80)

    # 4. Save Master CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, "master_benchmark_25_tests_mossformer.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved master benchmark CSV: {csv_path}")

    # 5. Save updated manifest
    manifest_path = os.path.join(results_dir, "test_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(updated_manifest, f, indent=2)
    print(f"Saved test manifest JSON: {manifest_path}")

    # 6. Read prior Conv-TasNet results for direct comparison
    prior_csv_path = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_speech_enhancement\master_benchmark_25_tests.csv"
    convtasnet_metrics = None
    if os.path.exists(prior_csv_path):
        df_conv = pd.read_csv(prior_csv_path)
        convtasnet_metrics = {
            "mean_sisdr_gain_db": round(float(df_conv['sisdr_gain_db'].mean()), 2),
            "mean_pesq_gain": round(float(df_conv['pesq_gain'].mean()), 3),
            "mean_stoi_gain": round(float(df_conv['stoi_gain'].mean()), 3),
            "mean_attenuation_pct": round(float(df_conv['noise_attenuation_pct'].mean()), 2),
            "mean_pearson_r": round(float(df_conv['pearson_r'].mean()), 4),
            "pass_rate_pct": round(float((df_conv['status'] == 'PASS').mean() * 100), 1)
        }

    # 7. Summary Statistics
    total_tests = len(df)
    pass_count = int((df['status'] == 'PASS').sum())
    pass_rate = round(float(pass_count / total_tests * 100), 1)

    summary = {
        "model_name": "MossFormerGAN_SE_16K (ClearerVoice-Studio)",
        "total_tests": total_tests,
        "pass_count": pass_count,
        "pass_rate_pct": pass_rate,
        "accuracy_rating": f"{pass_count}/{total_tests} ({pass_rate}%)",
        "metrics_summary": {
            "pesq": {
                "input_mean": round(float(df['input_pesq'].mean()), 3),
                "output_mean": round(float(df['mossformer_pesq'].mean()), 3),
                "mean_gain": round(float(df['pesq_gain'].mean()), 3),
                "std_gain": round(float(df['pesq_gain'].std()), 3),
                "min_gain": round(float(df['pesq_gain'].min()), 3),
                "max_gain": round(float(df['pesq_gain'].max()), 3)
            },
            "stoi": {
                "input_mean": round(float(df['input_stoi'].mean()), 3),
                "output_mean": round(float(df['mossformer_stoi'].mean()), 3),
                "mean_gain": round(float(df['stoi_gain'].mean()), 3),
                "std_gain": round(float(df['stoi_gain'].std()), 3),
                "min_gain": round(float(df['stoi_gain'].min()), 3),
                "max_gain": round(float(df['stoi_gain'].max()), 3)
            },
            "sisdr": {
                "input_mean_db": round(float(df['input_sisdr_db'].mean()), 2),
                "output_mean_db": round(float(df['mossformer_sisdr_db'].mean()), 2),
                "mean_gain_db": round(float(df['sisdr_gain_db'].mean()), 2),
                "std_gain_db": round(float(df['sisdr_gain_db'].std()), 2),
                "min_gain_db": round(float(df['sisdr_gain_db'].min()), 2),
                "max_gain_db": round(float(df['sisdr_gain_db'].max()), 2)
            },
            "noise_attenuation_pct": {
                "mean": round(float(df['noise_attenuation_pct'].mean()), 2),
                "std": round(float(df['noise_attenuation_pct'].std()), 2),
                "min": round(float(df['noise_attenuation_pct'].min()), 2),
                "max": round(float(df['noise_attenuation_pct'].max()), 2)
            },
            "pearson_r": {
                "mean": round(float(df['pearson_r'].mean()), 4),
                "std": round(float(df['pearson_r'].std()), 4),
                "min": round(float(df['pearson_r'].min()), 4),
                "max": round(float(df['pearson_r'].max()), 4)
            }
        },
        "convtasnet_baseline_comparison": convtasnet_metrics
    }

    summary_path = os.path.join(results_dir, "summary_statistics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics JSON: {summary_path}")

    # Copy to ClearParty_repo if directory exists
    repo_results_dir = r"C:\Users\jayde\ClearParty_repo\speech_enhancement\results\25_tests_mossformer_enhancement"
    try:
        if os.path.exists(r"C:\Users\jayde\ClearParty_repo"):
            os.makedirs(repo_results_dir, exist_ok=True)
            shutil.copy2(csv_path, os.path.join(repo_results_dir, "master_benchmark_25_tests_mossformer.csv"))
            shutil.copy2(summary_path, os.path.join(repo_results_dir, "summary_statistics.json"))
            shutil.copy2(manifest_path, os.path.join(repo_results_dir, "test_manifest.json"))
            print(f"Mirrored benchmark files to GitHub repository folder: {repo_results_dir}")
    except Exception as e:
        print(f"Notice: repo mirror skipped ({e})")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total Tests Evaluated: {total_tests}")
    print(f"Passed Tests:          {pass_count} ({pass_rate}%)")
    print(f"Mean SI-SDR Gain:      {summary['metrics_summary']['sisdr']['mean_gain_db']:+.2f} dB")
    print(f"Mean PESQ Gain:        {summary['metrics_summary']['pesq']['mean_gain']:+.3f}")
    print(f"Mean STOI Gain:        {summary['metrics_summary']['stoi']['mean_gain']:+.3f}")
    print(f"Mean Noise Attenuation:{summary['metrics_summary']['noise_attenuation_pct']['mean']:.2f}%")
    print(f"Mean Pearson r:        {summary['metrics_summary']['pearson_r']['mean']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
