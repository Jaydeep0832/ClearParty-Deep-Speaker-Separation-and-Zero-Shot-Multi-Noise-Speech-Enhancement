import pandas as pd, json, os

csv_path = r'C:\Users\jayde\Speech_Enhancement\results\25_tests_mossformer_enhancement\master_benchmark_25_tests_mossformer.csv'
df = pd.read_csv(csv_path)

records = []
for _, r in df.iterrows():
    tid = int(r['test_id'])
    k = int(r['noise_count'])
    snr = float(r['target_snr_db'])
    
    tier = 'Severe (0–2.5 dB)' if snr <= 2.5 else ('Medium (3–5.5 dB)' if snr <= 5.5 else 'Mild (6–10 dB)')
    
    pad_id = f'{tid:02d}'
    records.append({
        'id': tid,
        'padId': pad_id,
        'cleanFile': str(r['clean_file']),
        'snr': snr,
        'snrTier': tier,
        'noiseCount': k,
        'noiseTypes': [n.strip() for n in str(r['noise_description']).split('+')],
        'noiseDescription': str(r['noise_description']),
        'duration': float(r['duration_sec']),
        'inSisdr': float(r['input_sisdr_db']),
        'outSisdr': float(r['mossformer_sisdr_db']),
        'sisdrGain': float(r['sisdr_gain_db']),
        'inPesq': float(r['input_pesq']),
        'outPesq': float(r['mossformer_pesq']),
        'pesqGain': float(r['pesq_gain']),
        'inStoi': float(r['input_stoi']),
        'outStoi': float(r['mossformer_stoi']),
        'stoiGain': float(r['stoi_gain']),
        'attenuationPct': float(r['noise_attenuation_pct']),
        'pearsonR': float(r['pearson_r']),
        'inferenceTime': float(r['inference_time_sec']),
        'status': str(r['status']),
        'cleanAudio': f'/audio/tests/test_{pad_id}_clean.wav',
        'mixAudio': f'/audio/tests/test_{pad_id}_mixture_{k}noises.wav',
        'enhAudio': f'/audio/tests/test_{pad_id}_mossformer_enhanced.wav'
    })

data_dir = r'C:\Users\jayde\ClearParty_repo\frontend\src\data'
os.makedirs(data_dir, exist_ok=True)

out_file = os.path.join(data_dir, 'benchmarkData.js')
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('export const BENCHMARK_TESTS = ' + json.dumps(records, indent=2) + ';\n\n')
    
    summary = {
        'totalTests': 25,
        'passRate': 100.0,
        'meanSisdrGain': 8.99,
        'meanPesqGain': 1.184,
        'meanPesqOutput': 2.790,
        'peakPesqOutput': 3.468,
        'meanStoiGain': 0.141,
        'meanStoiOutput': 0.920,
        'meanAttenuationPct': 87.18,
        'meanPearsonR': 0.9782,
        'baselineComparison': {
            'convTasNetSisdrGain': 4.64,
            'convTasNetPesqGain': 0.268,
            'convTasNetStoiGain': 0.034,
            'convTasNetAttenuation': 65.01,
            'convTasNetPearsonR': 0.9442
        }
    }
    f.write('export const SUMMARY_STATS = ' + json.dumps(summary, indent=2) + ';\n\n')

    phases = [
        {
            'phase': 'Phase 1',
            'tag': 'BASELINE ABLATION',
            'title': 'Spectrogram U-Net & The Phase Catastrophe',
            'domain': 'Time-Frequency STFT (2D Magnitude)',
            'dataset': 'Libri2Mix dev (2,700 pairs)',
            'params': '1.8 M',
            'result': '+4.95 dB Test SI-SDRi, 1.72 PESQ, 73.5% STOI',
            'bottleneck': 'Discarding spectral phase and inverting with the corrupted mixture phase creates destructive comb-filtering and watery artifacts (< +6 dB ceiling).',
            'lesson': 'Proved that spectral magnitude masking cannot overcome phase cancellation in monaural cocktail party mixtures.'
        },
        {
            'phase': 'Phase 2A',
            'tag': 'WAVEFORM BREAKTHROUGH',
            'title': 'Time-Domain 1D Conv-TasNet Baseline',
            'domain': 'Raw Discrete Waveform (1D Time-Domain)',
            'dataset': 'Libri2Mix dev (2,700 pairs)',
            'params': '1.4 M',
            'result': '+6.98 dB Test SI-SDRi, 1.94 PESQ, 79.2% STOI',
            'bottleneck': 'Limited parameter capacity (1.4M) and small speaker count caused slight underfitting on challenging phoneme transitions.',
            'lesson': 'Learned 1D convolutional feature encoders completely bypassed the STFT phase estimation bottleneck.'
        },
        {
            'phase': 'Phase 2B',
            'tag': 'HARDWARE ACCELERATION',
            'title': 'Scaled 14k Dataset & Mixed Precision (AMP FP16)',
            'domain': 'Time-Domain TCN (R=3, B=256, Sc=256)',
            'dataset': 'LibriSpeech train-clean-100 (14,269 pairs, 251 speakers)',
            'params': '3.4 M (3,396,144 params)',
            'result': '+12.99 dB Train SI-SDR, +11.76 dB Val SI-SDR (+4.08 dB generalization boost)',
            'bottleneck': 'Throughput bottleneck solved: AMP FP16 and NVMe SSD staging reduced epoch duration from 14.0 min to 4.5 min (3.1x speedup).',
            'lesson': 'Demonstrated that scaling speaker diversity from 40 to 251 talkers prevents voice-memorization and enforces true acoustic disentanglement.'
        },
        {
            'phase': 'Phase 3',
            'tag': 'ACOUSTIC POLISH',
            'title': '200-Test Benchmark & Natural Spectral Wiener Post-Filter',
            'domain': 'Hybrid Time-Domain + Spectral Wiener Module',
            'dataset': '200 Held-Out Diverse Test Mixtures',
            'params': '3.4 M + Analytical Wiener Formulation',
            'result': '+11.06 dB Mean SI-SDRi (Peak +13.13 dB), 2.30 PESQ, 88.1% STOI',
            'bottleneck': 'Raw time-domain transposed convolution occasionally emitted high-frequency phase jitter (metallic synthetic buzz).',
            'lesson': 'Constructed analytical soft spectral Wiener filter W_c(f,t) = |S_c|^2 / (|S_1|^2 + |S_2|^2 + eps), restoring natural warm vocal timbre.'
        },
        {
            'phase': 'Phase 5',
            'tag': 'SOTA MULTI-NOISE CHAMPION',
            'title': 'MossFormerGAN SyncANet with Gated Triple-Attention',
            'domain': 'Complex-Compressed STFT (Magnitude + Phase Residual)',
            'dataset': 'NOIZEUS 8-Environment Stress Corpus (25 Compound Tests)',
            'params': 'ClearerVoice-Studio Generator + Multi-Scale Discriminator',
            'result': '+8.99 dB SI-SDR Gain, +1.184 PESQ Boost (2.790 mean, peak 3.468), 87.18% Noise Attenuation, 100% Pass Rate',
            'bottleneck': 'Addressed severe non-stationary compound acoustic interference (3–4 simultaneous noises at 0–10 dB SNR).',
            'lesson': 'Gated Triple-Attention avoids O(T^2) quadratic memory explosion; dual decoders solve phase mismatch; GAN discriminator eliminates musical noise.'
        }
    ]
    f.write('export const EVOLUTION_PHASES = ' + json.dumps(phases, indent=2) + ';\n\n')

    viva = [
        {
            'q': 'Why is the output PESQ score 2.79–3.468 instead of hitting the theoretical maximum of 4.5?',
            'category': 'Perceptual Evaluation',
            'a': 'Two physical and experimental constraints govern this: (1) The 8 kHz Nyquist Bandwidth Limit: NOIZEUS audio is natively sampled at 8 kHz (narrowband), which has a hard brick-wall frequency cutoff at 4,000 Hz. Upsampling to 16 kHz for MossFormer leaves empty upper spectral bins. ITU-T P.862 compares against full-spectrum expectations; on narrowband telephony audio, 2.7–3.2 represents commercial cellular VoLTE call quality. (2) Extreme Multi-Noise Protocol: Standard papers report >3.2 PESQ testing only 1 single stationary noise at +15 dB SNR. ClearParty evaluates 3 to 4 simultaneous non-stationary noises starting at 0.0 dB SNR, where input PESQ is a severely degraded 1.606. Gaining +1.184 MOS points under compound babble/airport/street chatter is a massive achievement.'
        },
        {
            'q': 'What was the root cause of the Phase Catastrophe in Phase 1 Spectrogram U-Net?',
            'category': 'Architecture Evolution',
            'a': 'The spectrogram U-Net predicted real-valued magnitude masks applied to the Short-Time Fourier Transform (STFT) magnitude, discarding the phase. In monaural mixtures, when two voices speak simultaneously, their acoustic waves interfere constructively and destructively, altering the true instantaneous phase. Inverting the masked spectrogram using the mixture noisy phase caused severe phase mismatch, producing audible comb filtering and watery synthetic distortions, capping SI-SDR at < +6 dB.'
        },
        {
            'q': 'How does Utterance-Level Permutation Invariant Training (uPIT) solve label ambiguity?',
            'category': 'Loss Functions & Training',
            'a': 'In monaural blind speaker separation, the network outputs Speaker A on Channel 1 and Speaker B on Channel 2, but the assignment order is completely arbitrary. Fixed-target MSE or L1 loss penalizes the network if it separates Speaker A onto Channel 2, causing gradient conflict. uPIT computes the negative SI-SDR loss across all C! possible permutations (2 permutations for dual speakers) and backpropagates solely through the permutation yielding the lowest loss.'
        },
        {
            'q': 'How does MossFormerGAN achieve linear memory complexity over long audio sequences?',
            'category': 'Attention Mechanisms',
            'a': 'Standard Transformer self-attention scales quadratically O(T^2) with time, creating memory exhaustion on long continuous speech files. MossFormer replaces quadratic global attention with Gated Triple-Attention combined with Finite Short Memory Networks (FSMN). It decomposes attention into full-band temporal self-attention, local sub-band attention, and feed-forward memory blocks, reducing computational complexity to linear O(T) while capturing both long-range phonetic cadence and local acoustic phone transitions.'
        },
        {
            'q': 'Why does raw Conv-TasNet have a metallic buzz, and how does the Wiener Filter eliminate it?',
            'category': 'Signal Processing & Timbre',
            'a': 'Conv-TasNet uses learned 1D transposed convolutions for its waveform decoder. While time-domain loss optimizes SI-SDR, it lacks explicit harmonic frequency constraints, causing micro-discontinuities across adjacent synthesis frames that human ears perceive as a faint metallic or robotic timbre. The soft spectral Wiener post-filter computes W_c(f,t) = |S_hat_c(f,t)|^2 / (|S_hat_1(f,t)|^2 + |S_hat_2(f,t)|^2 + eps) on the original mixture STFT, which naturally preserves the continuous harmonic vocal tract envelope of clean human speech.'
        },
        {
            'q': 'Why is SI-SDR preferred over standard SDR or SNR for speech separation?',
            'category': 'Metrics & Methodology',
            'a': 'Standard Signal-to-Noise Ratio (SNR) and classical BSS-EVAL SDR are highly sensitive to arbitrary amplitude scaling. If a separation model outputs alpha * s(t) where alpha != 1, standard SNR reports poor scores even though the audio sounds identical to human listeners. Scale-Invariant SDR (SI-SDR) mathematically orthogonalizes the estimated signal into an exact scaled target projection e_target = (s_hat . s / ||s||^2) * s and an orthogonal distortion error e_art = s_hat - e_target, ensuring scale-invariance and precise distortion measurement.'
        },
        {
            'q': 'What is the operational distinction between Speech Enhancement and Blind Speaker Separation?',
            'category': 'Problem Formulation',
            'a': 'Blind Speaker Separation solves x(t) = s_1(t) + s_2(t), where both signals are structured human speech with identical harmonic and formant statistics, requiring speaker disentanglement and permutation invariance. Speech Enhancement solves x(t) = s(t) + n_compound(t), where the interference consists of ambient acoustic environmental noises (mechanical hum, diffuse babble, traffic), requiring acoustic noise suppression while preserving speech formants with zero permutation ambiguity.'
        },
        {
            'q': 'What was the practical impact of Mixed Precision (AMP FP16) in Phase 2B?',
            'category': 'Systems & Optimization',
            'a': 'When scaling Conv-TasNet from 1.4M to 3.4M parameters across 14,269 LibriSpeech pairs, FP32 training took ~14 minutes per epoch. Migrating to Automatic Mixed Precision (AMP FP16) leveraged NVIDIA Tensor Cores for half-precision matrix multiplications while maintaining FP32 master weights for numerical stability. Combined with NVMe SSD dataset staging and batch size scaling to 12, epoch training time was slashed by ~70% down to 4.5 minutes (3.1x speedup) with zero loss in SI-SDR precision.'
        }
    ]
    f.write('export const VIVA_QUESTIONS = ' + json.dumps(viva, indent=2) + ';\n\n')

    sep_samples = [
        {
            'title': 'Overlapping Dual-Speaker Mixture',
            'subtitle': '0 dB SNR | Single Microphone Input',
            'tag': 'Corrupted Mixture',
            'tagColor': 'amber',
            'desc': 'Two clean speakers talking simultaneously with total vocal overlap.',
            'audio': '/audio/separation/Input Mixture (Both speakers talking).wav'
        },
        {
            'title': 'True Speaker 1 (Clean Reference)',
            'subtitle': 'Clean Target Ground Truth',
            'tag': 'Target Ground Truth',
            'tagColor': 'blue',
            'desc': 'Original uncorrupted speech utterance of Speaker 1.',
            'audio': '/audio/separation/True Speaker 1 (Clean Reference).wav'
        },
        {
            'title': 'Separated Speaker 1 (Raw Conv-TasNet)',
            'subtitle': '+11.76 dB SI-SDR | Waveform Output',
            'tag': 'Raw Neural Model',
            'tagColor': 'slate',
            'desc': 'Clean vocal separation, but note the subtle high-frequency synthetic metallic timbre.',
            'audio': '/audio/separation/Separated Speaker 1 (Model Output).wav'
        },
        {
            'title': 'Wiener-Restored Speaker 1',
            'subtitle': 'Post-Filtered Natural Voice',
            'tag': 'Wiener Restored',
            'tagColor': 'green',
            'desc': 'Analytical Wiener post-filter applied: metallic buzz eliminated, warm natural timbre restored.',
            'audio': '/audio/separation/Natural Wiener-Reconstructed Speaker 1 (Natural human voice).wav'
        },
        {
            'title': 'True Speaker 2 (Clean Reference)',
            'subtitle': 'Clean Target Ground Truth',
            'tag': 'Target Ground Truth',
            'tagColor': 'blue',
            'desc': 'Original uncorrupted speech utterance of Speaker 2.',
            'audio': '/audio/separation/True Speaker 2 (Clean Reference).wav'
        },
        {
            'title': 'Separated Speaker 2 (Raw Conv-TasNet)',
            'subtitle': '+11.76 dB SI-SDR | Waveform Output',
            'tag': 'Raw Neural Model',
            'tagColor': 'slate',
            'desc': 'Isolated secondary speaker output from raw Conv-TasNet 1D decoder.',
            'audio': '/audio/separation/Separated Speaker 2 (Model Output).wav'
        },
        {
            'title': 'Wiener-Restored Speaker 2',
            'subtitle': 'Post-Filtered Natural Voice',
            'tag': 'Wiener Restored',
            'tagColor': 'green',
            'desc': 'Secondary speaker vocal stream re-synthesized with natural organic overtones.',
            'audio': '/audio/separation/Natural Wiener-Reconstructed Speaker 2.wav'
        }
    ]
    f.write('export const SEPARATION_SAMPLES = ' + json.dumps(sep_samples, indent=2) + ';\n')

print('Generated benchmarkData.js successfully!')
