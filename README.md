# 🎙️ ClearParty: Deep Speaker Separation & Zero-Shot Multi-Noise Speech Enhancement
> **An End-to-End Deep Learning Framework for the Cocktail Party Problem: Time-Domain Disentanglement and Complex-Domain Spectral Recovery under Compound Acoustic Interference**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![ClearerVoice-Studio](https://img.shields.io/badge/ClearerVoice--Studio-ICASSP%20Winner-green.svg)](https://github.com/modelscope/ClearerVoice-Studio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Interactive Dashboard](https://img.shields.io/badge/Web%20Console-Interactive%20HTML-blue?style=for-the-badge&logo=html5&logoColor=white)](index.html)

> 🚀 **Instant Interactive Console:** Double-click **[`index.html`](index.html)** in any web browser to explore the interactive audio lab, real-time waveform visualizer, 25-test benchmark matrix, and viva defense guide with zero installation!

---

## 🎯 Problems This Project Solves

In everyday acoustic environments, human speech is rarely captured in isolation. ClearParty addresses **three fundamental acoustic challenges** using a single monaural microphone ($1 \text{ sensor} \ll N \text{ sources}$):

1. **The Classic Cocktail Party Problem (Overlapping Talkers):**
   * *Problem:* When two or more people speak at the same time, their vocal harmonics and formants overlap across the exact same frequency bands. Traditional linear filters cannot separate them.
   * *Solution:* **Track 1 (Conv-TasNet + uPIT)** performs end-to-end time-domain blind speaker separation, isolating distinct talkers without spatial microphone arrays.
2. **Severe Compound Environmental Noise (3–4 Simultaneous Sources):**
   * *Problem:* Real-world noise is rarely simple white noise. Speech is often simultaneously corrupted by diffuse speech babble, vehicle traffic, engine hum, and transient clatter at low SNRs (0–10 dB).
   * *Solution:* **Track 2 (MossFormerGAN SOTA)** achieves zero-shot multi-noise suppression using complex-domain attention, attenuating 87.18% of unwanted noise energy while preserving speech intelligibility.
3. **The Phase Catastrophe & Synthetic Metallic Buzz:**
   * *Problem:* Traditional magnitude spectrogram masking discards phase, causing destructive distortion upon inversion. Raw time-domain convolutions can also produce an unnatural "robotic/metallic" frame buzz.
   * *Solution:* Our **Natural Spectral Wiener Post-Filter** smooths micro-phase discontinuities and re-synthesizes authentic vocal tract harmonics.

### 🌍 Real-World Applications & Impact
* 🦻 **Next-Gen Hearing Aids:** Real-time conversational tracking in noisy restaurants and crowded social settings.
* 🎙️ **Teleconferencing & Virtual Meetings:** Clean speaker isolation and background noise elimination for Zoom, Teams, and Google Meet.
* 🤖 **Voice Assistants & ASR Preprocessing:** Slashes Word Error Rates (WER) for voice-driven smart devices and automotive cabins.
* 🚨 **Emergency, Cockpit & Air-Traffic Radios:** Restores life-critical speech comprehension in high-stress, noisy environments.

---

## 🧠 Conceptual System Architecture

```mermaid
graph LR
    subgraph Input ["Acoustic Interference"]
        A["Single Monaural Mic x(t)"] --> B["Overlapping Human Talkers<br>(Cocktail Party Problem)"]
        A --> C["3 to 4 Compound Noises<br>(Babble, Traffic, Engine, Hum)"]
    end

    subgraph Engine ["ClearParty Dual-Track Solution"]
        B --> D["Track 1: Conv-TasNet<br>• 1D Dilated TCN Separator<br>• uPIT Permutation Loss<br>• Natural Spectral Wiener Filter"]
        C --> E["Track 2: MossFormerGAN SOTA<br>• SyncANet Complex STFT Backbone<br>• Joint Sub-Band & Full-Band Attention<br>• Multi-Metric Discriminator"]
    end

    subgraph Output ["Isolated Pristine Speech"]
        D --> F["Speaker 1 Clean Voice"]
        D --> G["Speaker 2 Clean Voice"]
        E --> H["Enhanced Speech Output<br>+8.99 dB SI-SDR | 87.18% Attenuation"]
    end

    style Input fill:#1e293b,stroke:#ef4444,stroke-width:1.5px,color:#fff
    style Engine fill:#0f172a,stroke:#3b82f6,stroke-width:1.5px,color:#fff
    style Output fill:#14532d,stroke:#22c55e,stroke-width:1.5px,color:#fff
```

---

## 📊 Visual Acoustic Evidence & Spectrogram Verification

The Mel-spectrogram comparisons below demonstrate the model's ability to cleanly reconstruct vocal tract formants while eradicating compound interference:

| Track 1: Speaker Separation (Wiener Restoration) | Track 2: 6.79s Continuous Multi-Noise Enhancement |
|:---:|:---:|
| ![Wiener vs Raw Comparison](speaker_separation/docs/melspectrogram_wiener_vs_raw_comparison.png) | ![6-sec LibriSpeech Mel-Spectrogram](speech_enhancement/results/mel_spectrogram_libri2mix_6sec.png) |
| *Eliminates synthetic metallic buzz & restores authentic human vocal timbre* | *Suppresses compound noise across continuous speech (+7.22 dB SI-SDR, 2.54 PESQ)* |

---

## 🏆 Master Experimental Results Matrix

Evaluated across **14,269 LibriSpeech pairs** (Speaker Separation) and **25 Multi-Noise Stress Tests** combining 3 to 4 simultaneous non-stationary noises:

| Model Architecture | Objective / Task | Input Domain | SI-SDR Gain | PESQ Score | STOI Score | Noise Attenuation | Pass Rate |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **STFT U-Net Baseline** | Speaker Separation | Magnitude STFT | +3.12 dB | 1.84 | 0.72 | 48.2% | Failed (Phase Smearing) |
| **Conv-TasNet (Phase 2B)** | Speaker Separation | 1D Waveform | **+13.13 dB** | 2.18 | 0.88 | 88.1% | 100% (200/200 Pairs) |
| **Conv-TasNet + Wiener Filter** | Speaker Separation | Time + Spectral | **+13.13 dB** | **2.32** | **0.90** | **88.1%** | **100% (Natural Timbre)** |
| **Conv-TasNet Baseline** | Multi-Noise Enhancement | 1D Waveform | +4.64 dB | 1.87 | 0.81 | 65.0% | Baseline |
| **MossFormerGAN SOTA** | Multi-Noise Enhancement | Complex STFT | **+8.99 dB** | **2.79** | **0.92** | **87.18%** | **100% (25/25 Tests)** |

> **Key Takeaway:** MossFormerGAN nearly doubled the noise attenuation and SI-SDR performance of Conv-TasNet on compound noise, achieving **87.18% noise reduction** and **+1.184 PESQ gain** across all 25 stress test mixtures.

---

## ⚡ Quickstart & Interactive Console Launch

The entire interactive research suite is packaged into a single, self-contained HTML console that runs natively in any browser with **zero dependencies**:

| Mode | Command / Action | Description |
|---|---|---|
| **💻 Standalone Offline Run** | Double-click `index.html` | Opens instantly in Chrome, Edge, Firefox, or Safari with audio playback, live oscilloscope, and viva drill. |
| **🌐 Local HTTP Server** | `python -m http.server 8000` | Serves the console at `http://localhost:8000` with streaming playback for all 75 benchmark audio files. |

```bash
# Optional: Clone and run locally
git clone https://github.com/Jaydeep0832/ClearParty-Deep-Speaker-Separation-and-Zero-Shot-Multi-Noise-Speech-Enhancement.git
cd ClearParty-Deep-Speaker-Separation-and-Zero-Shot-Multi-Noise-Speech-Enhancement
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## 📂 Repository Structure

```text
ClearParty-Deep-Speaker-Separation-and-Zero-Shot-Multi-Noise-Speech-Enhancement/
├── index.html                                   # Master standalone interactive web console
├── cocktail_party_problem_walkthrough.html      # Self-contained presentation walkthrough
├── README.md                                    # Project documentation & visual summary
├── audio/                                       # Ingested benchmark audio assets
│   ├── tests/                                   # 75 benchmark audio files (25 clean, 25 mixtures, 25 SOTA)
│   └── separation/                              # 9 separation samples (mixtures, clean, raw, Wiener restored)
├── speaker_separation/                          # Track 1: Blind Speaker Separation
│   ├── docs/                                    # Spectrogram comparisons & evaluation plots
│   ├── models/                                  # Conv-TasNet PyTorch architecture
│   └── samples/                                 # 2-speaker audio samples
└── speech_enhancement/                          # Track 2: Multi-Noise Speech Enhancement
    ├── results/                                 # Mel-spectrograms & 25-test benchmark data
    └── samples/                                 # Multi-noise audio samples
```
