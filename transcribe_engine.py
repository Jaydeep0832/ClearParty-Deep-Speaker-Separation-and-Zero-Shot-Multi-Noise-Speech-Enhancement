# transcribe_engine.py
"""
Whisper-Powered Diarized Transcription Engine
Generates:
1. Speaker-attributed dialogue: Speaker-1 -> Sentence, Speaker-2 -> Sentence
2. Standard SubRip (.srt) subtitle files
3. Synchronized timestamps for interactive UI playback
"""

import os
import torch
from datetime import timedelta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transcribe_separated_audio(s1_audio_path, s2_audio_path, model_name="small.en"):
    """
    Transcribes separated audio channels and produces an interleaved dialogue timeline.
    """
    import whisper
    import srt
    
    print(f"🎙️ Loading Whisper Model ('{model_name}') on {DEVICE}...")
    asr_model = whisper.load_model(model_name, device=DEVICE)
    
    print("Transcribing Speaker 1 stream...")
    res1 = asr_model.transcribe(s1_audio_path, language="en", fp16=(DEVICE.type == "cuda"))
    
    print("Transcribing Speaker 2 stream...")
    res2 = asr_model.transcribe(s2_audio_path, language="en", fp16=(DEVICE.type == "cuda"))
    
    dialogue_events = []
    
    for seg in res1.get('segments', []):
        text = seg['text'].strip()
        if len(text) > 0:
            dialogue_events.append({
                'speaker': 'Speaker 1',
                'start': seg['start'],
                'end': seg['end'],
                'text': text
            })
            
    for seg in res2.get('segments', []):
        text = seg['text'].strip()
        if len(text) > 0:
            dialogue_events.append({
                'speaker': 'Speaker 2',
                'start': seg['start'],
                'end': seg['end'],
                'text': text
            })
            
    dialogue_events.sort(key=lambda x: x['start'])
    
    subtitles = []
    for idx, ev in enumerate(dialogue_events, start=1):
        sub = srt.Subtitle(
            index=idx,
            start=timedelta(seconds=ev['start']),
            end=timedelta(seconds=ev['end']),
            content=f"[{ev['speaker']}]: {ev['text']}"
        )
        subtitles.append(sub)
        
    srt_content = srt.compose(subtitles)
    
    return {
        'events': dialogue_events,
        'srt': srt_content,
        'speaker1_raw': res1['text'].strip(),
        'speaker2_raw': res2['text'].strip()
    }

def transcribe_enhanced_audio(enhanced_audio_path, model_name="small.en"):
    """
    Transcribes clean enhanced audio for live-style subtitle display.
    """
    import whisper
    import srt
    
    print(f"🎙️ Loading Whisper Model ('{model_name}') on {DEVICE}...")
    asr_model = whisper.load_model(model_name, device=DEVICE)
    
    res = asr_model.transcribe(enhanced_audio_path, language="en", fp16=(DEVICE.type == "cuda"))
    
    subtitles = []
    events = []
    for idx, seg in enumerate(res.get('segments', []), start=1):
        text = seg['text'].strip()
        if len(text) > 0:
            events.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': text
            })
            sub = srt.Subtitle(
                index=idx,
                start=timedelta(seconds=seg['start']),
                end=timedelta(seconds=seg['end']),
                content=text
            )
            subtitles.append(sub)
            
    srt_content = srt.compose(subtitles)
    
    return {
        'full_text': res['text'].strip(),
        'events': events,
        'srt': srt_content
    }

if __name__ == "__main__":
    import sys, json
    print("=" * 60)
    print("🎙️ ClearParty Diarized Whisper Speech-to-Text Engine (GPU)")
    print("=" * 60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    s1_candidates = [
        os.path.join(BASE_DIR, "separated_outputs", "speaker1_separated.wav"),
        os.path.join(BASE_DIR, "speaker_separation", "samples", "Separated Speaker 1 (Model Output).wav"),
        os.path.join(BASE_DIR, "audio", "separation", "Separated Speaker 1 (Model Output).wav")
    ]
    s2_candidates = [
        os.path.join(BASE_DIR, "separated_outputs", "speaker2_separated.wav"),
        os.path.join(BASE_DIR, "speaker_separation", "samples", "Separated Speaker 2 (Model Output).wav"),
        os.path.join(BASE_DIR, "audio", "separation", "Separated Speaker 2 (Model Output).wav")
    ]
    
    s1 = sys.argv[1] if len(sys.argv) > 1 else next((c for c in s1_candidates if os.path.exists(c)), None)
    s2 = sys.argv[2] if len(sys.argv) > 2 else next((c for c in s2_candidates if os.path.exists(c)), None)
    
    if s1 and s2 and os.path.exists(s1) and os.path.exists(s2):
        print(f"Transcribing Stream 1: {s1}")
        print(f"Transcribing Stream 2: {s2}")
        
        # Use medium.en if VRAM allows, else small.en
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        model_name = "medium.en" if vram_gb >= 10.0 else "small.en"
        print(f"Using Whisper model: '{model_name}' on {DEVICE}")
        
        result = transcribe_separated_audio(s1, s2, model_name=model_name)
        
        print("\n" + "=" * 60)
        print("📝 DIARIZED CONVERSATION TRANSCRIPTION:")
        print("=" * 60)
        for ev in result['events']:
            m_s = int(ev['start'] // 60)
            s_s = ev['start'] % 60
            m_e = int(ev['end'] // 60)
            s_e = ev['end'] % 60
            t_str = f"[{m_s:02d}:{s_s:05.2f} - {m_e:02d}:{s_e:05.2f}]"
            tag = "🔵 Speaker 1" if ev['speaker'] == "Speaker 1" else "🟢 Speaker 2"
            print(f"{tag} {t_str}: {ev['text']}")
            
        print("=" * 60)
        # Save SRT
        srt_path = os.path.join(BASE_DIR, "diarized_dialogue.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(result['srt'])
        print(f"💾 Saved Subtitles to: {srt_path}")
    else:
        print("⚠️ Separated audio streams not found. Please run `python3 long_audio_engine.py` first.")

