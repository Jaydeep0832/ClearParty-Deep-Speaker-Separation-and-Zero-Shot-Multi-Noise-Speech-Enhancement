# webapp_v2.py
"""
ClearParty AI v2.0: Unified Audio Separation, Enhancement & Diarized Transcription WebApp
Supports 5-minute audio files with GPU acceleration and optional Whisper ASR.
"""

import os
import sys
import io
import time
import json
import numpy as np
import soundfile as sf
import streamlit as st
import torch

from task_classifier import classify_audio_task
from long_audio_engine import process_5min_separation, process_5min_enhancement

st.set_page_config(
    page_title="ClearParty AI v2.0 - Diarized Speech Intelligence",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ ClearParty AI v2.0: Deep Audio Processing & Speech Intelligence")
st.markdown("**Dual-Track Speech Separation & Enhancement with Automatic Task Routing & Diarized Whisper ASR**")

# Sidebar
st.sidebar.title("⚡ Engine Status")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
st.sidebar.success(f"**Compute Device:**\n{device_name}")
vram_free = round(torch.cuda.mem_get_info()[0] / 1e9, 2) if torch.cuda.is_available() else 0.0
st.sidebar.info(f"**Available VRAM:** {vram_free} GB / 8.00 GB")

auto_detect = st.sidebar.checkbox("🤖 Enable Automatic Task Classification", value=True,
                                  help="Automatically analyzes audio to choose between Separation and Enhancement.")

mode = st.sidebar.radio("Force Manual Task Override:", ["Auto-Route", "Speaker Separation", "Speech Enhancement"])

st.markdown("---")
st.subheader("1. Audio Input (Maximum 5.0 Minutes)")

uploaded_file = st.file_uploader("Upload audio file (.wav, .mp3, .flac, .m4a):", type=["wav", "mp3", "flac", "m4a"])

if uploaded_file is not None:
    temp_path = "temp_input_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    data, sr = sf.read(temp_path)
    dur_sec = len(data) / sr
    st.audio(temp_path)
    st.write(f"⏱️ **File Duration:** {dur_sec:.1f} seconds ({dur_sec/60:.2f} minutes) | **Sampling Rate:** {sr} Hz")
    
    if dur_sec > 310.0:
        st.error("⚠️ File exceeds 5.0 minutes limit. Please upload audio under 5 minutes.")
        st.stop()
        
    if auto_detect and mode == "Auto-Route":
        with st.spinner("Analyzing acoustic features (flatness, harmonicity, F0 divergence)..."):
            clf = classify_audio_task(temp_path)
            selected_task = clf['task']
            st.info(f"🎯 **Auto-Classified Task:** **{selected_task}** (Confidence: {clf['confidence']}%) — *{clf['description']}*")
    else:
        selected_task = "SEPARATION" if mode == "Speaker Separation" else "ENHANCEMENT"
        
    if st.button("🚀 Process Long Audio Pipeline", type="primary", use_container_width=True):
        if selected_task == "SEPARATION":
            with st.spinner("Executing 5-minute Overlap-Add Separation..."):
                t0 = time.time()
                sep_res = process_5min_separation(temp_path)
                t1 = time.time()
                st.success(f"✅ Separation finished in {t1-t0:.2f}s (Real-Time Factor: {sep_res['rtf']:.3f}x)")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 👤 Separated Speaker 1")
                    st.audio(sep_res['s1_path'])
                    with open(sep_res['s1_path'], "rb") as f:
                        st.download_button("⬇️ Download Speaker 1 WAV", f, "speaker1_isolated.wav", "audio/wav")
                        
                with c2:
                    st.markdown("### 👤 Separated Speaker 2")
                    st.audio(sep_res['s2_path'])
                    with open(sep_res['s2_path'], "rb") as f:
                        st.download_button("⬇️ Download Speaker 2 WAV", f, "speaker2_isolated.wav", "audio/wav")
                        
            try:
                from transcribe_engine import transcribe_separated_audio
                with st.spinner("Running Whisper ASR transcription with speaker attribution..."):
                    asr_res = transcribe_separated_audio(sep_res['s1_path'], sep_res['s2_path'])
                    
                    st.markdown("---")
                    st.subheader("📝 Speaker-Attributed Diarized Transcript")
                    
                    for ev in asr_res['events']:
                        t_str = f"[{int(ev['start']//60):02d}:{ev['start']%60:05.2f} - {int(ev['end']//60):02d}:{ev['end']%60:05.2f}]"
                        if ev['speaker'] == "Speaker 1":
                            st.markdown(f"🔵 **{ev['speaker']}** `{t_str}`: {ev['text']}")
                        else:
                            st.markdown(f"🟢 **{ev['speaker']}** `{t_str}`: {ev['text']}")
                            
                    st.markdown("---")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button("⬇️ Download Diarized Subtitles (.SRT)", asr_res['srt'], "dialogue_subtitles.srt", "text/plain")
                    with d2:
                        st.download_button("⬇️ Download Structured Transcript (.JSON)", json.dumps(asr_res['events'], indent=2), "transcript.json", "application/json")
            except Exception as e:
                st.warning(f"Whisper transcription skipped: {e}. (Install `openai-whisper` and `srt` to enable).")
                    
        else:
            with st.spinner("Executing Speech Enhancement..."):
                t0 = time.time()
                enh_res = process_5min_enhancement(temp_path)
                t1 = time.time()
                st.success(f"✅ Speech Enhancement completed in {t1-t0:.2f}s (RTF: {enh_res['rtf']:.3f}x)")
                
                st.markdown("### 🌟 Enhanced Clean Restored Speech")
                st.audio(enh_res['enhanced_path'])
                with open(enh_res['enhanced_path'], "rb") as f:
                    st.download_button("⬇️ Download Clean Audio WAV", f, "clean_restored.wav", "audio/wav")
                    
            try:
                from transcribe_engine import transcribe_enhanced_audio
                with st.spinner("Generating live synchronized transcript via Whisper..."):
                    asr_enh = transcribe_enhanced_audio(enh_res['enhanced_path'])
                    st.markdown("---")
                    st.subheader("📝 Synchronized Transcript")
                    for ev in asr_enh['events']:
                        t_str = f"[{int(ev['start']//60):02d}:{ev['start']%60:05.2f} - {int(ev['end']//60):02d}:{ev['end']%60:05.2f}]"
                        st.markdown(f"🎙️ `{t_str}`: {ev['text']}")
                        
                    st.download_button("⬇️ Download Subtitles (.SRT)", asr_enh['srt'], "enhanced_subtitles.srt", "text/plain")
            except Exception as e:
                st.warning(f"Whisper transcription skipped: {e}.")
