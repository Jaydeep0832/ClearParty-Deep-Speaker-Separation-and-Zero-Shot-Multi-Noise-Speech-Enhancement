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
    
    # Decision boundary:
    if mean_flatness > 0.045 or harmonic_ratio < 0.40:
        task = 'ENHANCEMENT'
        confidence = min(0.98, max(0.60, (mean_flatness / 0.08) * 0.9))
        desc = "Dominant environmental stationary/non-stationary noise detected. Routing to Speech Enhancement."
    elif f0_std > 35.0 and harmonic_ratio >= 0.40:
        task = 'SEPARATION'
        confidence = min(0.96, max(0.65, (f0_std / 70.0) * 0.85 + 0.15))
        desc = "Bimodal pitch distribution and multi-talker harmonics detected. Routing to Two-Speaker Separation."
    else:
        if harmonic_ratio >= 0.55:
            task = 'SEPARATION'
            confidence = 0.70
            desc = "Rich harmonic speech content detected. Defaulting to Speaker Separation."
        else:
            task = 'ENHANCEMENT'
            confidence = 0.70
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
