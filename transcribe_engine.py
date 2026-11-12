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
