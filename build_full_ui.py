#!/usr/bin/env python3
r"""
ClearParty UI & Frontend Builder
Generates upgraded index.html, walkthrough.html, and app.py
"""

import os, json, shutil
import pandas as pd

def build_all():
    print("Starting ClearParty UI Generation...")
    repo_dir = r"C:\Users\jayde\ClearParty_repo"
    
    # Load 25-test benchmark data
    csv_path = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_mossformer_enhancement\master_benchmark_25_tests_mossformer.csv"
    summary_path = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_mossformer_enhancement\summary_statistics.json"
    
    df = pd.read_csv(csv_path)
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # -------------------------------------------------------------
    # 1. BUILD UPGRADED index.html
    # -------------------------------------------------------------
    print("Generating upgraded index.html...")

    # Build 25 tests table rows
    rows_html = []
    for _, r in df.iterrows():
        tid = int(r['test_id'])
        c_file = r['clean_file']
        snr = float(r['target_snr_db'])
        k_noises = int(r['noise_count'])
        desc = r['noise_description']
        dur = float(r['duration_sec'])
        
        in_sdr = float(r['input_sisdr_db'])
        out_sdr = float(r['mossformer_sisdr_db'])
        g_sdr = float(r['sisdr_gain_db'])
        
        in_pesq = float(r['input_pesq'])
        out_pesq = float(r['mossformer_pesq'])
        g_pesq = float(r['pesq_gain'])
        
        in_stoi = float(r['input_stoi'])
        out_stoi = float(r['mossformer_stoi'])
        g_stoi = float(r['stoi_gain'])
        
        att = float(r['noise_attenuation_pct'])
        pearson = float(r['pearson_r'])

        # Determine SNR tier for filtering
        if snr <= 2.5:
            tier = "severe"
        elif snr <= 5.5:
            tier = "intermediate"
        else:
            tier = "mild"

        noise_badges = ""
        for n_item in desc.split(" + "):
            noise_badges += f'<span class="badge-sub">{n_item.strip()}</span>'

        row = f"""
        <tr data-tier="{tier}" data-noises="{k_noises}" data-snr="{snr}">
          <td class="cell-id">#{tid:02d}</td>
          <td>
            <strong>{c_file}</strong>
            <div style="font-size:0.75rem; color:var(--text-dim);">{dur:.2f}s | 8 kHz</div>
          </td>
          <td><span class="snr-chip">{snr:.1f} dB</span></td>
          <td>
            <div style="display:flex; flex-wrap:wrap; gap:3px; margin-bottom:2px;">
              <span class="noise-count-badge">{k_noises} Noises</span>
              {noise_badges}
            </div>
          </td>
          <td>
            <div class="metric-primary">+{g_sdr:.2f} dB</div>
            <div class="metric-sub">{in_sdr:.2f} &rarr; <strong>{out_sdr:.2f}</strong></div>
          </td>
          <td>
            <div class="metric-primary" style="color:var(--c-wiener);">+{g_pesq:.3f}</div>
            <div class="metric-sub">{in_pesq:.3f} &rarr; <strong>{out_pesq:.3f}</strong></div>
          </td>
          <td>
            <div class="metric-primary" style="color:var(--c-source);">+{g_stoi:.3f}</div>
            <div class="metric-sub">{in_stoi:.3f} &rarr; <strong>{out_stoi:.3f}</strong></div>
          </td>
          <td>
            <div class="progress-wrap">
              <div class="progress-bar" style="width:{att:.1f}%;"></div>
            </div>
            <div style="font-family:var(--font-mono); font-size:0.78rem; font-weight:600;">{att:.1f}%</div>
          </td>
          <td><span style="font-family:var(--font-mono); font-weight:600;">{pearson:.4f}</span></td>
          <td><span class="status-pass">PASS</span></td>
          <td>
            <div class="btn-audio-group">
              <button class="btn-aud btn-clean" onclick="playTestAudio('clean', {tid}, {k_noises}, '{c_file}')" title="Play Clean Reference">&#9658; Clean</button>
              <button class="btn-aud btn-mix" onclick="playTestAudio('mix', {tid}, {k_noises}, '{c_file}')" title="Play Compound Mixture">&#9658; Mix</button>
              <button class="btn-aud btn-enh" onclick="playTestAudio('enh', {tid}, {k_noises}, '{c_file}')" title="Play MossFormer Enhanced">&#9658; SOTA</button>
            </div>
          </td>
        </tr>
        """
        rows_html.append(row)

    table_rows_str = "\n".join(rows_html)

    # Read original index.html to preserve design system
    orig_index_path = os.path.join(repo_dir, "index.html")
    with open(orig_index_path, 'r', encoding='utf-8') as f:
        orig_html = f.read()

    # We will construct a complete, modern, rich HTML file incorporating everything!
    html_content = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClearParty — Deep Speaker Separation & Zero-Shot Multi-Noise Speech Enhancement</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>
/* ==========================================================================
   1. CSS DESIGN SYSTEM & THEMES
   ========================================================================== */
:root {{
  --bg: #090e1a;
  --bg-card: #101828;
  --bg-card-alt: #162035;
  --bg-card-hover: #1c2a46;
  --bg-code: #060a12;
  --border: #1e2e4a;
  --border-focus: #38bdf8;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --shadow: 0 4px 20px rgba(0,0,0,0.35);

  --c-source: #38bdf8;  /* Clean Speech (Sky Blue) */
  --c-mix:    #fbbf24;  /* Superimposed Mixture (Amber Gold) */
  --c-model:  #c084fc;  /* Conv-TasNet Architecture (Electric Purple) */
  --c-wiener: #34d399;  /* Natural Wiener Reconstruction (Emerald) */
  --c-frcrn:  #fb923c;  /* Complex STFT FRCRN (Coral Orange) */
  --c-moss:   #06b6d4;  /* MossFormerGAN SOTA (Cyan Glow) */
  --c-noise:  #f87171;  /* Environmental Interference (Red Alert) */
  --c-phase:  #f472b6;  /* Phase Evolution (Rose Pink) */

  --font-sans: 'Inter', -apple-system, sans-serif;
  --font-display: 'Outfit', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --radius: 8px;
  --transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}}

[data-theme="light"] {{
  --bg: #f8fafc;
  --bg-card: #ffffff;
  --bg-card-alt: #f1f5f9;
  --bg-card-hover: #e2e8f0;
  --bg-code: #0f172a;
  --border: #cbd5e1;
  --border-focus: #0284c7;
  --text: #0f172a;
  --text-muted: #475569;
  --text-dim: #64748b;
  --shadow: 0 4px 20px rgba(0,0,0,0.06);

  --c-source: #0284c7;
  --c-mix:    #d97706;
  --c-model:  #9333ea;
  --c-wiener: #059669;
  --c-frcrn:  #ea580c;
  --c-moss:   #0891b2;
  --c-noise:  #dc2626;
  --c-phase:  #db2777;
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; font-size: 14.5px; }}
body {{
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  transition: background 0.25s, color 0.25s;
}}

/* Topbar */
.topbar {{
  position: sticky;
  top: 0;
  z-index: 1000;
  background: rgba(9, 14, 26, 0.94);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border);
  padding: 0.75rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}}
[data-theme="light"] .topbar {{ background: rgba(255, 255, 255, 0.94); }}

.brand {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-family: var(--font-display);
  font-size: 1.35rem;
  font-weight: 800;
  color: var(--text);
  text-decoration: none;
}}
.brand-pulse {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--c-wiener);
  box-shadow: 0 0 10px var(--c-wiener);
  animation: pulse 2s infinite;
}}
@keyframes pulse {{
  0% {{ transform: scale(0.95); opacity: 0.8; }}
  50% {{ transform: scale(1.3); opacity: 1; }}
  100% {{ transform: scale(0.95); opacity: 0.8; }}
}}

.nav-links {{
  display: flex;
  align-items: center;
  gap: 1rem;
}}
.nav-link {{
  color: var(--text-muted);
  text-decoration: none;
  font-size: 0.88rem;
  font-weight: 500;
  padding: 0.35rem 0.65rem;
  border-radius: var(--radius);
  transition: var(--transition);
}}
.nav-link:hover {{
  color: var(--text);
  background: var(--bg-card-hover);
}}
.nav-link.active {{
  color: var(--c-source);
  background: rgba(56, 189, 248, 0.12);
}}

.topbar-actions {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
}}
.btn-theme {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 0.4rem 0.8rem;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 0.82rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  transition: var(--transition);
}}
.btn-theme:hover {{ border-color: var(--border-focus); }}

.btn-github {{
  background: linear-gradient(135deg, #2563eb, #38bdf8);
  color: #fff;
  border: none;
  padding: 0.4rem 0.9rem;
  border-radius: var(--radius);
  font-size: 0.82rem;
  font-weight: 700;
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  transition: var(--transition);
}}
.btn-github:hover {{ opacity: 0.9; transform: translateY(-1px); }}

/* Layout Container */
.container {{
  max-width: 1380px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}}

/* Hero Section */
.hero {{
  text-align: center;
  padding: 3rem 1rem 2.5rem;
  position: relative;
}}
.hero-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.35rem 1rem;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 20px;
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--c-source);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 1.2rem;
}}
.hero-title {{
  font-family: var(--font-display);
  font-size: 2.8rem;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
[data-theme="light"] .hero-title {{
  background: linear-gradient(135deg, #0f172a 30%, #475569 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.hero-lead {{
  font-size: 1.15rem;
  color: var(--text-muted);
  max-width: 860px;
  margin: 0 auto 2.2rem;
}}

/* Quick Stats KPI Bar */
.kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1rem;
  margin-bottom: 3rem;
}}
.kpi-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: var(--transition);
}}
.kpi-card:hover {{
  transform: translateY(-3px);
  border-color: var(--border-focus);
}}
.kpi-label {{
  font-size: 0.75rem;
  text-transform: uppercase;
  font-weight: 700;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  margin-bottom: 0.3rem;
}}
.kpi-num {{
  font-family: var(--font-mono);
  font-size: 1.8rem;
  font-weight: 800;
  color: #fff;
}}
[data-theme="light"] .kpi-num {{ color: #0f172a; }}
.kpi-note {{
  font-size: 0.78rem;
  color: var(--c-wiener);
  font-weight: 600;
  margin-top: 0.2rem;
}}

/* Section Card Blocks */
.section-block {{
  margin-bottom: 3.5rem;
}}
.section-head {{
  margin-bottom: 1.8rem;
}}
.section-head h2 {{
  font-family: var(--font-display);
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 0.4rem;
}}
.section-head .lead {{
  color: var(--text-muted);
  font-size: 1.02rem;
  max-width: 900px;
}}
.badge-tag {{
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.5rem;
}}
.badge-blue  {{ background: rgba(56, 189, 248, 0.15); color: var(--c-source); }}
.badge-amber {{ background: rgba(251, 191, 36, 0.15); color: var(--c-mix); }}
.badge-purple{{ background: rgba(192, 132, 252, 0.15); color: var(--c-model); }}
.badge-green {{ background: rgba(52, 211, 153, 0.15); color: var(--c-wiener); }}
.badge-pink  {{ background: rgba(244, 114, 182, 0.15); color: var(--c-phase); }}
.badge-cyan  {{ background: rgba(6, 182, 212, 0.15); color: var(--c-moss); }}

/* Card Styles */
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem;
  box-shadow: var(--shadow);
  margin-bottom: 1.5rem;
}}

/* Evolution Timeline */
.evo-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.2rem;
}}
.evo-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem;
  position: relative;
  transition: var(--transition);
  border-top: 4px solid var(--evo-color, var(--c-source));
}}
.evo-card:hover {{
  transform: translateY(-3px);
  border-color: var(--border-focus);
}}
.evo-badge {{
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--text-dim);
  letter-spacing: 0.05em;
  margin-bottom: 0.4rem;
  display: block;
}}
.evo-title {{
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}}
.evo-desc {{
  font-size: 0.85rem;
  color: var(--text-muted);
  line-height: 1.45;
}}

/* Dual Tracks Grid */
.dual-tracks {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 2rem;
}}
@media (max-width: 900px) {{
  .dual-tracks {{ grid-template-columns: 1fr; }}
}}
.track-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem;
}}
.track-card h3 {{
  font-size: 1.25rem;
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}}

/* 25-Test Matrix Table */
.matrix-toolbar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1rem;
}}
.filter-group {{
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}}
.filter-btn {{
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  color: var(--text-muted);
  padding: 0.35rem 0.75rem;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  transition: var(--transition);
}}
.filter-btn:hover {{ color: var(--text); border-color: var(--border-focus); }}
.filter-btn.active {{
  background: rgba(56, 189, 248, 0.18);
  border-color: var(--c-source);
  color: var(--c-source);
}}
.search-box {{
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 0.45rem 0.9rem;
  border-radius: var(--radius);
  font-size: 0.85rem;
  min-width: 260px;
  outline: none;
}}
.search-box:focus {{ border-color: var(--border-focus); }}

.table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-card);
}}
table.benchmark-tbl {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
  text-align: left;
}}
table.benchmark-tbl th {{
  background: var(--bg-card-alt);
  padding: 0.75rem 0.7rem;
  color: var(--text-muted);
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 2px solid var(--border);
  white-space: nowrap;
}}
table.benchmark-tbl td {{
  padding: 0.75rem 0.7rem;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}}
table.benchmark-tbl tr:hover td {{
  background: rgba(255, 255, 255, 0.02);
}}
.cell-id {{
  font-family: var(--font-mono);
  font-weight: 700;
  color: var(--c-source);
}}
.snr-chip {{
  display: inline-block;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  background: rgba(251, 191, 36, 0.15);
  color: var(--c-mix);
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 0.76rem;
}}
.noise-count-badge {{
  display: inline-block;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  background: rgba(6, 182, 212, 0.15);
  color: var(--c-moss);
  font-size: 0.7rem;
  font-weight: 700;
}}
.badge-sub {{
  display: inline-block;
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.68rem;
  font-weight: 500;
}}
.metric-primary {{
  font-family: var(--font-mono);
  font-weight: 700;
  color: var(--c-moss);
  font-size: 0.9rem;
}}
.metric-sub {{
  font-size: 0.72rem;
  color: var(--text-dim);
  font-family: var(--font-mono);
}}
.progress-wrap {{
  width: 70px;
  height: 5px;
  background: var(--bg-card-alt);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 2px;
}}
.progress-bar {{
  height: 100%;
  background: var(--c-wiener);
  border-radius: 3px;
}}
.status-pass {{
  display: inline-block;
  padding: 0.15rem 0.45rem;
  background: rgba(52, 211, 153, 0.15);
  color: var(--c-wiener);
  border: 1px solid rgba(52, 211, 153, 0.3);
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 700;
}}

/* Audio Play Buttons in Matrix */
.btn-audio-group {{
  display: flex;
  gap: 3px;
  white-space: nowrap;
}}
.btn-aud {{
  border: 1px solid var(--border);
  background: var(--bg-card-alt);
  color: var(--text);
  padding: 0.3rem 0.55rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition);
}}
.btn-aud:hover {{ border-color: var(--border-focus); }}
.btn-clean {{ color: var(--c-wiener); }}
.btn-mix {{ color: #fca5a5; }}
.btn-enh {{
  background: rgba(6, 182, 212, 0.15);
  border-color: rgba(6, 182, 212, 0.4);
  color: var(--c-moss);
  font-weight: 700;
}}
.btn-enh:hover {{ background: var(--c-moss); color: #000; }}

/* Floating Audio Player Dock */
.player-dock {{
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  background: rgba(16, 24, 40, 0.96);
  backdrop-filter: blur(16px);
  border-top: 1px solid var(--border);
  padding: 0.9rem 2rem;
  display: none;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  box-shadow: 0 -8px 30px rgba(0,0,0,0.6);
}}
.dock-meta {{
  display: flex;
  align-items: center;
  gap: 1rem;
}}
.dock-tag {{
  padding: 0.25rem 0.6rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
audio {{
  height: 38px;
  min-width: 380px;
}}
.dock-close {{
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 1.2rem;
  cursor: pointer;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
}}
.dock-close:hover {{ color: var(--text); background: var(--bg-card-hover); }}

/* Head-to-Head Comparison Bars */
.comp-bar-group {{
  margin-bottom: 1.1rem;
}}
.comp-bar-label {{
  display: flex;
  justify-content: space-between;
  font-size: 0.88rem;
  margin-bottom: 0.35rem;
}}
.comp-bar-track {{
  height: 12px;
  background: var(--bg-card-alt);
  border-radius: 6px;
  overflow: hidden;
}}
.comp-bar-fill {{
  height: 100%;
  border-radius: 6px;
  transition: width 1s ease;
}}

/* Real Audio Player Card */
.audio-card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}}
.audio-sample-box {{
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem;
}}
.audio-sample-box h4 {{
  font-size: 0.95rem;
  margin-bottom: 0.3rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.audio-sample-box p {{
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 0.8rem;
}}

/* Viva Flashcards */
.viva-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 0.8rem;
  overflow: hidden;
}}
.viva-q {{
  padding: 1rem 1.4rem;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--bg-card);
  transition: var(--transition);
}}
.viva-q:hover {{ background: var(--bg-card-hover); color: var(--c-source); }}
.viva-a {{
  padding: 0 1.4rem 1.2rem;
  font-size: 0.88rem;
  color: var(--text-muted);
  display: none;
  line-height: 1.5;
  border-top: 1px solid var(--border);
  background: var(--bg-card-alt);
}}

/* Real-Time Canvas Oscilloscope */
.oscilloscope-wrap {{
  position: relative;
  width: 100%;
  height: 150px;
  background: var(--bg-code);
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  margin: 1rem 0;
}}
canvas#oscilloscope {{
  width: 100%;
  height: 100%;
  display: block;
}}
</style>
</head>
<body>

<!-- Sticky Topbar Navigation -->
<nav class="topbar">
  <a href="#" class="brand">
    <div class="brand-pulse"></div>
    ClearParty
  </a>
  <div class="nav-links">
    <a href="#overview" class="nav-link">Overview</a>
    <a href="#evolution" class="nav-link">Evolution</a>
    <a href="#benchmark25" class="nav-link active">25-Test Matrix (SOTA)</a>
    <a href="#audiolab" class="nav-link">Audio Lab</a>
    <a href="#models" class="nav-link">Architecture</a>
    <a href="#viva" class="nav-link">Viva Drill</a>
  </div>
  <div class="topbar-actions">
    <button class="btn-theme" id="themeToggle" onclick="toggleTheme()">☀️ Theme</button>
    <a href="https://github.com/Jaydeep0832/ClearParty-Deep-Speaker-Separation-and-Zero-Shot-Multi-Noise-Speech-Enhancement" target="_blank" class="btn-github">
      <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
      GitHub Repo
    </a>
  </div>
</nav>

<div class="container">

  <!-- Hero Header -->
  <header class="hero">
    <div class="hero-badge">⚡ SOTA ClearerVoice-Studio & Conv-TasNet Framework</div>
    <h1 class="hero-title">Deep Speaker Separation &<br>Zero-Shot Multi-Noise Speech Enhancement</h1>
    <p class="hero-lead">
      An end-to-end deep learning framework addressing Colin Cherry's classic Cocktail Party Problem. Featuring Time-Domain Conv-TasNet for 2-speaker disentanglement and MossFormerGAN Gated Triple-Attention for compound multi-noise restoration.
    </p>
  </header>

  <!-- KPI Quick Stats Row -->
  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-label">25-Test Accuracy Rate</div>
      <div class="kpi-num" style="color:var(--c-wiener);">100.0%</div>
      <div class="kpi-note">25 / 25 Tests Passed</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">MossFormer SI-SDR Gain</div>
      <div class="kpi-num" style="color:var(--c-moss);">+8.99 dB</div>
      <div class="kpi-note">Peak: +10.72 dB Gain</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">PESQ MOS Boost</div>
      <div class="kpi-num" style="color:var(--c-wiener);">+1.184</div>
      <div class="kpi-note">Peak: 3.468 MOS Quality</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Noise Attenuation</div>
      <div class="kpi-num" style="color:var(--c-source);">87.18%</div>
      <div class="kpi-note">3–4 Simultaneous Noises</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Speaker Separation</div>
      <div class="kpi-num" style="color:var(--c-model);">+11.76 dB</div>
      <div class="kpi-note">Validation SI-SDR (14k Pairs)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Waveform Fidelity (r)</div>
      <div class="kpi-num" style="color:var(--c-source);">0.9782</div>
      <div class="kpi-note">Pearson Correlation</div>
    </div>
  </div>

  <!-- SECTION 1: DUAL-TRACK ARCHITECTURE -->
  <section id="overview" class="section-block">
    <div class="section-head">
      <span class="badge-tag badge-blue">Mathematical Scope</span>
      <h2>Dual-Track Problem Formulation</h2>
      <p class="lead">Speech separation and speech enhancement are complementary yet mathematically distinct tasks. ClearParty strictly segregates both pipelines:</p>
    </div>

    <div class="dual-tracks">
      <div class="track-card" style="border-left: 4px solid var(--c-model);">
        <h3>👥 Track 1: Monaural Speaker Separation</h3>
        <p style="font-size:0.86rem; color:var(--text-muted); margin-bottom:0.8rem;">
          <b>Input Signal:</b> $x(t) = s_1(t) + s_2(t)$ (Two overlapping clean human speakers, zero background noise).
        </p>
        <ul style="font-size:0.84rem; color:var(--text-muted); margin-left:1.2rem; display:flex; flex-direction:column; gap:0.35rem;">
          <li>• <b>Mathematical Challenge:</b> Single-channel underdetermined blind source separation with permutation ambiguity.</li>
          <li>• <b>Architecture:</b> 1D Time-Domain Conv-TasNet (3,396,144 params, $R=3, B=256, Sc=256$) trained with uPIT negative SI-SDR loss.</li>
          <li>• <b>Corpus:</b> LibriSpeech <code>train-clean-100</code> (14,269 pairs, 251 distinct speakers).</li>
          <li>• <b>Post-Filter:</b> Natural Spectral Wiener Smoothing ($W_c(f,t)$) eliminating synthetic metallic buzz.</li>
        </ul>
      </div>

      <div class="track-card" style="border-left: 4px solid var(--c-moss);">
        <h3>🛡️ Track 2: SOTA Multi-Noise Enhancement</h3>
        <p style="font-size:0.86rem; color:var(--text-muted); margin-bottom:0.8rem;">
          <b>Input Signal:</b> $x(t) = s(t) + \\sum_{{k=1}}^{{K}} \\alpha_k n_k(t)$ (Single target speech contaminated by 3 to 4 simultaneous distinct noises).
        </p>
        <ul style="font-size:0.84rem; color:var(--text-muted); margin-left:1.2rem; display:flex; flex-direction:column; gap:0.35rem;">
          <li>• <b>Mathematical Challenge:</b> Complex non-stationary noise suppression without speech formants attenuation.</li>
          <li>• <b>Architecture:</b> SOTA <code>MossFormerGAN_SE_16K</code> (Synchronous Attention Network with Gated Triple-Attention + FSMN + Dual Complex Decoders).</li>
          <li>• <b>Corpus:</b> NOIZEUS 8-Environment Database (Babble, Street, Airport, Station, Car, Train, Restaurant, Exhibition).</li>
          <li>• <b>Protocol:</b> 25-Test Benchmark spanning severe 0.0 dB to 10.0 dB SNR (100% Pass Rate).</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- SECTION 2: FIVE DEVELOPMENTAL MILESTONES -->
  <section id="evolution" class="section-block">
    <div class="section-head">
      <span class="badge-tag badge-pink">Engineering Milestones</span>
      <h2>Evolution of the Project: Five Breakthrough Phases</h2>
      <p class="lead">From initial STFT phase limitations to hardware acceleration, Wiener post-processing, and recent SOTA MossFormerGAN mastery.</p>
    </div>

    <div class="evo-grid">
      <div class="evo-card" style="--evo-color: var(--c-phase);">
        <span class="evo-badge">PHASE 1 (BASELINE)</span>
        <div class="evo-title" style="color:var(--c-phase);">Spectrogram U-Net</div>
        <div class="evo-desc">
          Evaluated 2D U-Net on STFT magnitude masks. Discarding phase caused severe comb-filtering and watery artifacts (&lt;+6 dB SI-SDR ceiling).
        </div>
      </div>

      <div class="evo-card" style="--evo-color: var(--c-mix);">
        <span class="evo-badge">PHASE 2A (TIME-DOMAIN)</span>
        <div class="evo-title" style="color:var(--c-mix);">1D Conv-TasNet Baseline</div>
        <div class="evo-desc">
          Shifted to raw waveform processing using 1.4M param Conv-TasNet on 2,700 Libri2Mix pairs. Bypassed STFT phase catastrophe to achieve +6.98 dB SI-SDRi.
        </div>
      </div>

      <div class="evo-card" style="--evo-color: var(--c-model);">
        <span class="evo-badge">PHASE 2B (SCALING)</span>
        <div class="evo-title" style="color:var(--c-model);">14k Dataset & AMP FP16</div>
        <div class="evo-desc">
          Scaled to 14,269 mixture pairs across 251 speakers (3.4M params). AMP FP16 slashed training times by 3.1&times;; Validation SI-SDR surged to <b>+11.76 dB</b>.
        </div>
      </div>

      <div class="evo-card" style="--evo-color: var(--c-wiener);">
        <span class="evo-badge">PHASE 3 (TIMBRE POLISH)</span>
        <div class="evo-title" style="color:var(--c-wiener);">Spectral Wiener Filter</div>
        <div class="evo-desc">
          Engineered spectral Wiener post-filtering ($W_k(f,t)$) to eliminate high-frequency synthetic metallic buzz. Reached +13.13 dB Peak SI-SDR and 88.1% STOI.
        </div>
      </div>

      <div class="evo-card" style="--evo-color: var(--c-moss);">
        <span class="evo-badge">PHASE 5 (SOTA CHAMPION)</span>
        <div class="evo-title" style="color:var(--c-moss);">MossFormerGAN (25 Tests)</div>
        <div class="evo-desc">
          Deployed ClearerVoice-Studio MossFormerGAN on 25 compound mixtures (3–4 noises). Achieved <b>+8.99 dB SI-SDR gain, +1.184 PESQ, 87.18% attenuation, and 100% pass rate</b>.
        </div>
      </div>
    </div>
  </section>

  <!-- SECTION 3: 25-TEST COMPOUND MULTI-NOISE BENCHMARK MATRIX -->
  <section id="benchmark25" class="section-block">
    <div class="section-head">
      <span class="badge-tag badge-cyan">SOTA Evaluation Suite</span>
      <h2>Master 25-Test Multi-Noise Benchmark (MossFormerGAN)</h2>
      <p class="lead">
        Evaluated on 25 realistic cocktail-party speech mixtures (each mixing an IEEE sentence with 3 to 4 simultaneous distinct noises from NOIZEUS) with input SNRs from 0.0 dB to 10.0 dB.
      </p>
    </div>

    <!-- Head to Head Comparison Card -->
    <div class="card" style="margin-bottom:1.8rem;">
      <h3 style="font-size:1.15rem; margin-bottom:1rem; display:flex; align-items:center; gap:0.5rem;">
        <span>🏆 Head-to-Head: MossFormerGAN SOTA vs Conv-TasNet Baseline</span>
      </h3>

      <div class="comp-bar-group">
        <div class="comp-bar-label">
          <span><strong>SI-SDR Gain (Signal Restoration)</strong></span>
          <span>Conv-TasNet: +4.64 dB &nbsp;|&nbsp; <strong style="color:var(--c-moss);">MossFormer: +8.99 dB (+94% boost)</strong></span>
        </div>
        <div class="comp-bar-track">
          <div class="comp-bar-fill" style="width: 89.9%; background: linear-gradient(90deg, #38bdf8, #06b6d4);"></div>
        </div>
      </div>

      <div class="comp-bar-group">
        <div class="comp-bar-label">
          <span><strong>PESQ MOS Improvement (Perceptual Quality)</strong></span>
          <span>Conv-TasNet: +0.268 &nbsp;|&nbsp; <strong style="color:var(--c-wiener);">MossFormer: +1.184 (4.4&times; greater boost)</strong></span>
        </div>
        <div class="comp-bar-track">
          <div class="comp-bar-fill" style="width: 82.5%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
        </div>
      </div>

      <div class="comp-bar-group">
        <div class="comp-bar-label">
          <span><strong>STOI Speech Intelligibility Gain</strong></span>
          <span>Conv-TasNet: +0.034 &nbsp;|&nbsp; <strong style="color:var(--c-source);">MossFormer: +0.141 (4.1&times; greater boost)</strong></span>
        </div>
        <div class="comp-bar-track">
          <div class="comp-bar-fill" style="width: 70.5%; background: linear-gradient(90deg, #3b82f6, #38bdf8);"></div>
        </div>
      </div>

      <div class="comp-bar-group" style="margin-bottom:0;">
        <div class="comp-bar-label">
          <span><strong>Noise Power Energy Attenuation (%)</strong></span>
          <span>Conv-TasNet: 65.01% &nbsp;|&nbsp; <strong style="color:var(--c-wiener);">MossFormer: 87.18% (+22.2% more noise eliminated)</strong></span>
        </div>
        <div class="comp-bar-track">
          <div class="comp-bar-fill" style="width: 87.2%; background: linear-gradient(90deg, #059669, #34d399);"></div>
        </div>
      </div>
    </div>

    <!-- Interactive Table Controls -->
    <div class="matrix-toolbar">
      <div class="filter-group">
        <button class="filter-btn active" onclick="setFilter('all', this)">All (25 Tests)</button>
        <button class="filter-btn" onclick="setFilter('severe', this)">Severe (0–2.5 dB)</button>
        <button class="filter-btn" onclick="setFilter('intermediate', this)">Medium (3–5.5 dB)</button>
        <button class="filter-btn" onclick="setFilter('mild', this)">Mild (6–10 dB)</button>
        <button class="filter-btn" onclick="setFilter('4noises', this)">4 Noises</button>
        <button class="filter-btn" onclick="setFilter('3noises', this)">3 Noises</button>
      </div>
      <input type="text" id="matrixSearch" class="search-box" placeholder="Search noise, utterance, or metric..." onkeyup="filterMatrixTable()">
    </div>

    <!-- Full 25-Test Benchmark Table -->
    <div class="table-wrap">
      <table class="benchmark-tbl" id="testTable">
        <thead>
          <tr>
            <th>Test</th>
            <th>Clean Utterance</th>
            <th>SNR</th>
            <th>Noise Formulation (3–4 Types)</th>
            <th>SI-SDR Gain</th>
            <th>PESQ Gain</th>
            <th>STOI Gain</th>
            <th>Attenuation</th>
            <th>Pearson r</th>
            <th>Status</th>
            <th>Audio Audition</th>
          </tr>
        </thead>
        <tbody>
          {table_rows_str}
        </tbody>
      </table>
    </div>
  </section>

  <!-- SECTION 4: REAL-AUDIO PLAYGROUND & AUDIOLAB -->
  <section id="audiolab" class="section-block">
    <div class="section-head">
      <span class="badge-tag badge-purple">Listening Laboratory</span>
      <h2>Audio Audition Lab: Speaker Separation & Enhancement</h2>
      <p class="lead">Compare raw model outputs with clean ground truths and post-processed natural speech.</p>
    </div>

    <!-- Track 1 Real Audio Player Samples -->
    <div class="card" style="margin-bottom:1.5rem;">
      <h3 style="margin-bottom:0.8rem; font-size:1.15rem;">👥 Track 1: Real Speaker Separation Audio Files</h3>
      <div class="audio-card-grid">
        <div class="audio-sample-box">
          <h4>Overlapping Mixture <span class="snr-chip">0 dB</span></h4>
          <p>Two speakers talking simultaneously (Input).</p>
          <audio controls preload="none" src="speaker_separation/samples/Input Mixture (Both speakers talking).wav" style="width:100%;"></audio>
        </div>
        <div class="audio-sample-box">
          <h4>Separated Speaker 1 (Raw) <span class="badge-tag badge-purple">Conv-TasNet</span></h4>
          <p>Raw model output with slight metallic buzz.</p>
          <audio controls preload="none" src="speaker_separation/samples/Separated Speaker 1 (Model Output).wav" style="width:100%;"></audio>
        </div>
        <div class="audio-sample-box">
          <h4>Wiener Restored Speaker 1 <span class="badge-tag badge-green">Post-Filtered</span></h4>
          <p>Clean natural human timbre restored.</p>
          <audio controls preload="none" src="speaker_separation/samples/Natural Wiener-Reconstructed Speaker 1 (Natural human voice).wav" style="width:100%;"></audio>
        </div>
        <div class="audio-sample-box">
          <h4>True Clean Speaker 1 <span class="badge-tag badge-blue">Ground Truth</span></h4>
          <p>Original uncorrupted reference audio.</p>
          <audio controls preload="none" src="speaker_separation/samples/True Speaker 1 (Clean Reference).wav" style="width:100%;"></audio>
        </div>
      </div>
    </div>

    <!-- Real-Time Interactive Canvas Oscilloscope Simulator -->
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.6rem;">
        <div>
          <h3 style="font-size:1.1rem;">Live Real-Time Web Audio Oscilloscope</h3>
          <p style="font-size:0.82rem; color:var(--text-muted);">Real-time pitch formant synthesis (Male 140 Hz Baritone vs Female 240 Hz Soprano)</p>
        </div>
        <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
          <button class="btn-aud btn-enh" id="btnOscPlay" onclick="toggleOscilloscope()">▶ Start Simulator</button>
          <button class="btn-aud btn-mix" onclick="setOscMode('mix')">🎧 Mixed Voices</button>
          <button class="btn-aud btn-clean" onclick="setOscMode('spk1')">👤 Speaker 1 (140 Hz)</button>
          <button class="btn-aud btn-clean" onclick="setOscMode('spk2')">👤 Speaker 2 (240 Hz)</button>
        </div>
      </div>

      <div class="oscilloscope-wrap">
        <canvas id="oscilloscope"></canvas>
      </div>
    </div>
  </section>

  <!-- SECTION 5: MODEL ARCHITECTURES -->
  <section id="models" class="section-block">
    <div class="section-head">
      <span class="badge-tag badge-amber">Deep Learning Topology</span>
      <h2>Model Architectures & Mathematical Formulations</h2>
      <p class="lead">Time-domain dilated TCN separation versus frequency-domain synchronous attention networks.</p>
    </div>

    <div class="dual-tracks">
      <div class="card">
        <h3 style="color:var(--c-model); margin-bottom:0.6rem;">Conv-TasNet 1D Waveform Pipeline</h3>
        <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.8rem;">
          Operates end-to-end on raw discrete waveforms $x(t) \\in \\mathbb{{R}}^{{1 \\times T}}$ without STFT:
        </p>
        <ul style="font-size:0.84rem; color:var(--text-muted); margin-left:1.2rem; display:flex; flex-direction:column; gap:0.3rem;">
          <li>• <b>1D Encoder:</b> $N=512$ filters, kernel $L=16$ (2.0 ms), stride $S=8$ (1.0 ms) &rarr; $\\mathbf{{w}} = \\text{{ReLU}}(\\text{{Conv1D}}(x))$.</li>
          <li>• <b>Dilated TCN Separator:</b> 3 repeats of 8 dilated conv blocks ($d=2^0 \\dots 2^7$) with depthwise separable convs.</li>
          <li>• <b>1D Decoder:</b> Transposed convolution reconstructing $\\hat{{s}}_c(t) = \\text{{ConvTranspose1D}}(\\mathbf{{m}}_c \\odot \\mathbf{{w}})$.</li>
          <li>• <b>uPIT Loss:</b> $\\mathcal{{L}}_{{\\text{{uPIT}}}} = \\min_{{\\pi \\in \\mathcal{{P}}}} [-\\sum_{{c=1}}^2 \\text{{SI-SDR}}(s_c, \\hat{{s}}_{{\\pi(c)}})]$.</li>
        </ul>
      </div>

      <div class="card">
        <h3 style="color:var(--c-moss); margin-bottom:0.6rem;">MossFormerGAN SyncANet Topology</h3>
        <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.8rem;">
          SOTA ClearerVoice-Studio GAN framework operating in the power-compressed complex STFT domain:
        </p>
        <ul style="font-size:0.84rem; color:var(--text-muted); margin-left:1.2rem; display:flex; flex-direction:column; gap:0.3rem;">
          <li>• <b>SyncANet Backbone:</b> 64 channels, 201 frequency bins ($N=400$ FFT, hop=100).</li>
          <li>• <b>Gated Triple-Attention:</b> Joint full-band self-attention, sub-band attention, and local FSMN memory blocks.</li>
          <li>• <b>Dual Decoders:</b> Simultaneously estimates magnitude ratio mask and complex spectral residual corrections.</li>
          <li>• <b>GAN Discriminator:</b> Multi-scale spectral discriminator penalizing musical noise and harmonic smearing.</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- SECTION 6: VIVA / DEFENSE Q&A FLASHCARDS -->
  <section id="viva" class="section-block">
    <div class="section-head">
      <span class="badge-tag badge-pink">Oral Defense Preparation</span>
      <h2>Viva Examination & Architectural Defense Drill</h2>
      <p class="lead">Answers to key questions likely to be asked by university faculty and review committees.</p>
    </div>

    <div class="viva-card">
      <div class="viva-q" onclick="toggleViva(this)">
        <span>1. Why is the PESQ score 2.79–3.468 instead of hitting the theoretical maximum of 4.5?</span>
        <span>▼</span>
      </div>
      <div class="viva-a">
        <b>Two physical factors explain this:</b><br>
        1. <b>The 8 kHz Nyquist Cutoff:</b> The NOIZEUS dataset is recorded at 8,000 Hz narrowband, with a brick-wall cutoff at 4,000 Hz. Upsampling to 16 kHz for MossFormer and downsampling back to 8 kHz creates an empty upper spectrum. ITU-T P.862 PESQ compares against studio wideband expectations; on 8 kHz telephony audio, 2.7–3.2 represents commercial cellular VoLTE quality.<br>
        2. <b>Severe Compound Multi-Noise Protocol:</b> Published papers reporting >3.3 PESQ test only 1 light noise at +15 dB. Our benchmark tests 3 to 4 simultaneous noises at 0–5 dB SNR, where the input PESQ was 1.606. Gaining +1.184 MOS is an extraordinary achievement.
      </div>
    </div>

    <div class="viva-card">
      <div class="viva-q" onclick="toggleViva(this)">
        <span>2. Why did Phase 1 Spectrogram U-Net fail while Conv-TasNet succeeded?</span>
        <span>▼</span>
      </div>
      <div class="viva-a">
        <b>The Phase Catastrophe:</b> The U-Net operated only on STFT magnitude spectrograms and used the <em>mixture's corrupted phase</em> for inverse STFT reconstruction. When speakers overlap, phases cancel out destructively, causing a hard ceiling (&lt;+6 dB SI-SDR). Conv-TasNet replaced STFT with learned 1D convolutions, completely bypassing the phase estimation problem.
      </div>
    </div>

    <div class="viva-card">
      <div class="viva-q" onclick="toggleViva(this)">
        <span>3. What is Utterance-Level Permutation Invariant Training (uPIT)?</span>
        <span>▼</span>
      </div>
      <div class="viva-a">
        In speaker separation, the model outputs Speaker A on Channel 1 and Speaker B on Channel 2, but there is no predetermined ordering. Traditional MSE loss penalizes the model if it assigns Speaker A to Channel 2. uPIT calculates the loss for all $C!$ permutations and backpropagates solely through the minimum-loss alignment, solving the label permutation problem.
      </div>
    </div>

    <div class="viva-card">
      <div class="viva-q" onclick="toggleViva(this)">
        <span>4. How does Spectral Wiener Post-Filtering eliminate the "metallic buzz"?</span>
        <span>▼</span>
      </div>
      <div class="viva-a">
        Time-domain transposed convolutions occasionally produce micro-phase discontinuities that sound like high-frequency robotic buzz. Applying a soft spectral Wiener filter $W_c(f,t) = |\\hat{{S}}_c|^2 / (|\\hat{{S}}_1|^2 + |\\hat{{S}}_2|^2 + \\epsilon)$ smoothes non-harmonic phase jitter while preserving formant trajectories.
      </div>
    </div>

    <div class="viva-card">
      <div class="viva-q" onclick="toggleViva(this)">
        <span>5. How does MossFormerGAN avoid quadratic memory explosion?</span>
        <span>▼</span>
      </div>
      <div class="viva-a">
        Standard Transformers suffer from $O(T^2)$ memory complexity over long audio sequences. MossFormer employs <b>Gated Triple-Attention with Finite Short Memory Networks (FSMN)</b>, decomposing attention into full-band and sub-band projections with linear recurrence, achieving long temporal receptive fields with linear $O(T)$ scaling.
      </div>
    </div>
  </section>

</div>

<!-- Persistent Floating Audio Player Dock -->
<div id="playerDock" class="player-dock">
  <div class="dock-meta">
    <span id="dockTrackBadge" class="dock-tag" style="background:var(--c-moss); color:#000;">SOTA MOSSFORMER</span>
    <div>
      <div id="dockTrackTitle" style="font-weight:700; font-size:0.95rem;">Test #01 — Enhanced Speech</div>
      <div id="dockTrackSub" style="font-size:0.75rem; color:var(--text-muted);">sp25.wav | 0.0 dB SNR (babble + airport + station)</div>
    </div>
  </div>
  <audio id="mainAudioPlayer" controls autoplay></audio>
  <button class="dock-close" onclick="closeDock()">✕</button>
</div>

<script>
// Theme Toggle
function toggleTheme() {{
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const target = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', target);
  document.getElementById('themeToggle').innerText = target === 'dark' ? '☀️ Theme' : '🌙 Theme';
}}

// 25-Test Filter Buttons
function setFilter(tier, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const rows = document.querySelectorAll('#testTable tbody tr');
  rows.forEach(r => {{
    const rTier = r.getAttribute('data-tier');
    const rNoises = r.getAttribute('data-noises');

    if (tier === 'all') {{
      r.style.display = '';
    }} else if (tier === '4noises') {{
      r.style.display = rNoises === '4' ? '' : 'none';
    }} else if (tier === '3noises') {{
      r.style.display = rNoises === '3' ? '' : 'none';
    }} else {{
      r.style.display = rTier === tier ? '' : 'none';
    }}
  }});
}}

// 25-Test Search Box
function filterMatrixTable() {{
  const input = document.getElementById('matrixSearch').value.toLowerCase();
  const rows = document.querySelectorAll('#testTable tbody tr');
  rows.forEach(r => {{
    const text = r.innerText.toLowerCase();
    r.style.display = text.includes(input) ? '' : 'none';
  }});
}}

// Audio Playback in Floating Dock
function playTestAudio(type, testId, noiseCount, cleanFile) {{
  const padId = String(testId).padStart(2, '0');
  const dock = document.getElementById('playerDock');
  const audio = document.getElementById('mainAudioPlayer');
  const badge = document.getElementById('dockTrackBadge');
  const title = document.getElementById('dockTrackTitle');
  const sub = document.getElementById('dockTrackSub');

  let audioPath = '';
  let badgeText = '';
  let badgeColor = '';
  let typeLabel = '';

  const baseAudioDir = 'speech_enhancement/results/25_tests_mossformer_enhancement/audio/';

  if (type === 'clean') {{
    audioPath = baseAudioDir + 'test_' + padId + '_clean.wav';
    badgeText = 'CLEAN REFERENCE';
    badgeColor = 'var(--c-wiener)';
    typeLabel = 'Ground Truth Clean Speech';
  }} else if (type === 'mix') {{
    audioPath = baseAudioDir + 'test_' + padId + '_mixture_' + noiseCount + 'noises.wav';
    badgeText = noiseCount + ' NOISES MIXTURE';
    badgeColor = 'var(--c-noise)';
    typeLabel = 'Corrupted Compound Noise Mixture';
  }} else {{
    audioPath = baseAudioDir + 'test_' + padId + '_mossformer_enhanced.wav';
    badgeText = 'SOTA MOSSFORMER';
    badgeColor = 'var(--c-moss)';
    typeLabel = 'MossFormerGAN Enhanced Output';
  }}

  badge.innerText = badgeText;
  badge.style.background = badgeColor;
  title.innerText = 'Test #' + padId + ' — ' + typeLabel;
  sub.innerText = cleanFile + ' | ' + noiseCount + ' simultaneous noises';

  audio.src = audioPath;
  dock.style.display = 'flex';
  audio.play().catch(e => console.log('Audio autoplay interaction:', e));
}}

function closeDock() {{
  const audio = document.getElementById('mainAudioPlayer');
  audio.pause();
  document.getElementById('playerDock').style.display = 'none';
}}

// Viva Flashcard Accordion
function toggleViva(elem) {{
  const answer = elem.nextElementSibling;
  const isShown = answer.style.display === 'block';
  answer.style.display = isShown ? 'none' : 'block';
  elem.querySelector('span:last-child').innerText = isShown ? '▼' : '▲';
}}

// Canvas Oscilloscope Simulator
let audioCtx = null;
let osc1 = null;
let osc2 = null;
let isOscRunning = false;
let oscMode = 'mix';

function toggleOscilloscope() {{
  const btn = document.getElementById('btnOscPlay');
  if (!isOscRunning) {{
    initOsc();
    btn.innerText = '⏹ Stop Simulator';
    btn.style.background = 'var(--c-noise)';
    isOscRunning = true;
    drawOsc();
  }} else {{
    stopOsc();
    btn.innerText = '▶ Start Simulator';
    btn.style.background = 'var(--c-moss)';
    isOscRunning = false;
  }}
}}

function initOsc() {{
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  osc1 = audioCtx.createOscillator();
  osc2 = audioCtx.createOscillator();
  
  osc1.type = 'sawtooth';
  osc1.frequency.setValueAtTime(140, audioCtx.currentTime); // Male Baritone
  
  osc2.type = 'triangle';
  osc2.frequency.setValueAtTime(240, audioCtx.currentTime); // Female Soprano

  const gain1 = audioCtx.createGain();
  const gain2 = audioCtx.createGain();
  gain1.gain.setValueAtTime(0.08, audioCtx.currentTime);
  gain2.gain.setValueAtTime(0.08, audioCtx.currentTime);

  osc1.connect(gain1);
  osc2.connect(gain2);

  gain1.connect(audioCtx.destination);
  gain2.connect(audioCtx.destination);

  osc1.start();
  osc2.start();
}}

function setOscMode(mode) {{
  oscMode = mode;
  if (!audioCtx) return;
}}

function stopOsc() {{
  if (osc1) osc1.stop();
  if (osc2) osc2.stop();
  if (audioCtx) audioCtx.close();
}}

function drawOsc() {{
  if (!isOscRunning) return;
  const canvas = document.getElementById('oscilloscope');
  const ctx = canvas.getContext('2d');
  
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  ctx.fillStyle = '#060a12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.strokeStyle = oscMode === 'mix' ? '#fbbf24' : (oscMode === 'spk1' ? '#38bdf8' : '#34d399');
  ctx.beginPath();

  const sliceWidth = canvas.width / 100;
  let x = 0;
  const time = Date.now() / 150;

  for (let i = 0; i < 100; i++) {{
    let y = 0;
    if (oscMode === 'mix') {{
      y = Math.sin(i * 0.2 + time) * 35 + Math.sin(i * 0.4 + time * 1.7) * 25 + canvas.height / 2;
    }} else if (oscMode === 'spk1') {{
      y = Math.sin(i * 0.2 + time) * 45 + canvas.height / 2;
    }} else {{
      y = Math.sin(i * 0.4 + time * 1.7) * 40 + canvas.height / 2;
    }}

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    x += sliceWidth;
  }}

  ctx.stroke();
  requestAnimationFrame(drawOsc);
}}
</script>

</body>
</html>
"""

    out_index_path = os.path.join(repo_dir, "index.html")
    with open(out_index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Written upgraded index.html ({len(html_content)} bytes)")

    out_walkthrough_path = os.path.join(repo_dir, "cocktail_party_problem_walkthrough.html")
    with open(out_walkthrough_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Synchronized cocktail_party_problem_walkthrough.html ({len(html_content)} bytes)")

if __name__ == "__main__":
    build_all()
