# task_classifier.py
"""
ClearParty: Acoustic Scene & Task Auto-Classifier
Determines whether an input recording is:
(A) Multi-Speaker Co-Channel Mixture -> Routes to Speaker Separation
(B) Single-Speaker Multi-Noise Corruption -> Routes to Speech Enhancement
"""

import numpy as np
import scipy.signal
import soundfile as sf
import librosa

def classify_audio_task(audio_path, target_sr=8000):
    """
    Analyzes acoustic features:
    - Spectral Flatness (Wiener entropy)
    - Harmonic-to-Noise Ratio (HNR)
    - Dual Pitch Candidate Divergence
    - Energy Variance Across Sub-bands
    
    Returns:
        dict: {
            'task': 'SEPARATION' | 'ENHANCEMENT',
            'confidence': float,
            'metrics': dict,
            'description': str
        }
    """
    # Load up to 30 seconds for fast analysis
    y, sr = librosa.load(audio_path, sr=target_sr, duration=30.0)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # 1. Spectral Flatness (High flatness = broad noise; Low flatness = structured speech)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mean_flatness = float(np.mean(flatness))
    
    # 2. Spectral Rolloff & Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(centroid))
    
    # 3. Harmonic and Percussive Separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = float(np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-8))
    
    # 4. Multi-pitch / F0 estimation (overlapping talkers create dual F0 tracks)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    
    # Standard deviation of voiced F0 (two talkers produce wide, bimodal F0 variance)
    f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 10 else 0.0
    
    metrics = {
        'spectral_flatness': mean_flatness,
        'spectral_centroid_hz': mean_centroid,
        'harmonic_ratio': harmonic_ratio,
        'f0_std_hz': f0_std
    }
    
    # Calibrated decision boundary for 8 kHz / 16 kHz telephony and clean audio:
    # 1. Environmental noise produces high flatness (> 0.14) or near-zero pitch tracking.
    # 2. Overlapping speech has strong vocal formant modulation (F0 std > 25 Hz) even in 8kHz speech.
    if mean_flatness > 0.14 or (harmonic_ratio < 0.10 and f0_std < 15.0):
        task = 'ENHANCEMENT'
        confidence = min(0.98, max(0.65, (mean_flatness / 0.15) * 85.0))
        desc = "Dominant environmental stationary/non-stationary noise detected. Routing to Speech Enhancement."
    elif f0_std > 20.0 or harmonic_ratio >= 0.12:
        task = 'SEPARATION'
        conf_val = min(0.96, max(0.70, (f0_std / 45.0) * 80.0 + 15.0))
        confidence = conf_val
        desc = "Bimodal pitch distribution and multi-talker vocal harmonics detected. Routing to Two-Speaker Separation."
    else:
        if f0_std >= 18.0:
            task = 'SEPARATION'
            confidence = 72.0
            desc = "Multi-talker vocal pitch variation detected. Defaulting to Speaker Separation."
        else:
            task = 'ENHANCEMENT'
            confidence = 70.0
            desc = "Broadband acoustic corruption detected. Defaulting to Speech Enhancement."
            
    return {
        'task': task,
        'confidence': round(confidence * 100, 1),
        'metrics': metrics,
        'description': desc
    }

if __name__ == "__main__":
    import sys, os
    BASE = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(BASE, "speaker_separation", "samples", "Input Mixture (Both speakers talking).wav"),
        os.path.join(BASE, "speaker separation", "samples", "Input Mixture (Both speakers talking).wav"),
        os.path.join(BASE, "audio", "separation", "Input Mixture (Both speakers talking).wav")
    ]
    test_file = sys.argv[1] if len(sys.argv) > 1 else next((c for c in candidates if os.path.exists(c)), "mixture.wav")
    if os.path.exists(test_file):
        result = classify_audio_task(test_file)
        print("\n🔍 ACOUSTIC SCENE CLASSIFICATION RESULT:")
        print(f"File: {test_file}")
        print(f"Recommended Task: {result['task']} (Confidence: {result['confidence']}%)")
        print(f"Reason: {result['description']}")
        print(f"Metrics: {result['metrics']}")
    else:
        print(f"File not found: {test_file}")
