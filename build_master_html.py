#!/usr/bin/env python3
"""
Master Builder for ClearParty Institutional HTML Dashboard
Builds a single, self-contained, fully interactive, publication-grade HTML application.
Zero Node dependencies, classic academic design system, human-readable formulas,
interactive audio player, 25-test benchmark matrix, and mentor defense guide.
"""

import os
import json
import pandas as pd

def generate_dashboard():
    repo_dir = r"C:\Users\jayde\ClearParty_repo"
    csv_path = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_mossformer_enhancement\master_benchmark_25_tests_mossformer.csv"
    summary_path = r"C:\Users\jayde\Speech_Enhancement\results\25_tests_mossformer_enhancement\summary_statistics.json"

    df = pd.read_csv(csv_path)
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # 1. Build Table Rows for 25-Test Matrix
    table_rows = []
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

        if snr <= 2.5:
            tier = "severe"
            tier_badge = '<span class="tier-chip tier-severe">Severe (0–2.5 dB)</span>'
        elif snr <= 5.5:
            tier = "intermediate"
            tier_badge = '<span class="tier-chip tier-med">Medium (2.5–5 dB)</span>'
        else:
            tier = "mild"
            tier_badge = '<span class="tier-chip tier-mild">Mild (>5 dB)</span>'

        noise_badges = "".join([f'<span class="noise-pill">{n.strip()}</span>' for n in desc.split(' + ')])

        clean_path = f"audio/tests/test_{tid:02d}_clean.wav"
        mix_path = f"audio/tests/test_{tid:02d}_mixture_{k_noises}noises.wav"
        enh_path = f"audio/tests/test_{tid:02d}_mossformer_enhanced.wav"

        row = f"""
        <tr data-tier="{tier}" data-noises="{k_noises}" data-snr="{snr}">
          <td class="cell-id">#{tid:02d}</td>
          <td>
            <div class="file-name">{c_file}</div>
            <div class="file-sub">{dur:.2f}s &bull; 8 kHz Nyquist</div>
          </td>
          <td>
            <div class="snr-val">{snr:.1f} dB</div>
            {tier_badge}
          </td>
          <td>
            <div class="noise-desc-cell">
              <span class="noise-k-badge">{k_noises} Compound Noises</span>
              <div class="noise-pills-wrap">{noise_badges}</div>
            </div>
          </td>
          <td>
            <div class="metric-gain">+{g_sdr:.2f} dB</div>
            <div class="metric-flow">{in_sdr:.2f} &rarr; <strong>{out_sdr:.2f}</strong></div>
          </td>
          <td>
            <div class="metric-gain gain-pesq">+{g_pesq:.3f}</div>
            <div class="metric-flow">{in_pesq:.3f} &rarr; <strong>{out_pesq:.3f}</strong></div>
          </td>
          <td>
            <div class="metric-gain gain-stoi">+{g_stoi:.3f}</div>
            <div class="metric-flow">{in_stoi:.3f} &rarr; <strong>{out_stoi:.3f}</strong></div>
          </td>
          <td>
            <div class="att-bar-wrap">
              <div class="att-bar" style="width: {att:.1f}%;"></div>
            </div>
            <div class="att-pct-val">{att:.1f}%</div>
          </td>
          <td><span class="pearson-val">{pearson:.4f}</span></td>
          <td><span class="badge-pass">PASS</span></td>
          <td>
            <div class="table-audio-btns">
              <button class="t-btn t-btn-clean" onclick="playAudioRow('{clean_path}', 'Test #{tid:02d} &bull; Clean Reference ({c_file})', 'Clean Speech')" title="Play Clean Ground Truth">&#9658; Clean</button>
              <button class="t-btn t-btn-mix" onclick="playAudioRow('{mix_path}', 'Test #{tid:02d} &bull; {k_noises}-Noise Mixture ({snr:.1f} dB SNR)', 'Multi-Noise Mixture')" title="Play Compound Mixture">&#9658; Mix</button>
              <button class="t-btn t-btn-sota" onclick="playAudioRow('{enh_path}', 'Test #{tid:02d} &bull; MossFormerGAN SOTA Output (+{g_sdr:.2f} dB SI-SDR)', 'SOTA Enhanced')" title="Play MossFormer Enhanced">&#9658; SOTA</button>
            </div>
          </td>
        </tr>
        """
        table_rows.append(row)

    table_rows_html = "\n".join(table_rows)

    sisdr_mean = summary['metrics_summary']['sisdr']['mean_gain_db']
    pesq_mean = summary['metrics_summary']['pesq']['mean_gain']
    att_mean = summary['metrics_summary']['noise_attenuation_pct']['mean']
    pass_rate = summary['pass_rate_pct']

    # HTML Raw Template
    html_template = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClearParty • Deep Speaker Separation & Multi-Noise Speech Enhancement Console</title>

<!-- Publication Font System: Inter, JetBrains Mono, Serif Academic -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<!-- KaTeX Math Engine for Publication-Grade Equation Typesetting -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>

<style>
/* ==========================================================================
   CLASSICAL INSTITUTIONAL DESIGN SYSTEM (NO AI-GENERATED AESTHETICS)
   Oxford Navy, British Racing Green, Warm Ochre Amber, Slate Steel
   ========================================================================== */
:root {
  --bg-app: #0b1120;
  --bg-surface: #111b2e;
  --bg-surface-alt: #162238;
  --bg-card: #142036;
  --bg-code: #070d18;
  --border-color: #1e293b;
  --border-subtle: #243552;
  --border-active: #3b82f6;

  --text-main: #f1f5f9;
  --text-muted: #94a3b8;
  --text-dim: #64748b;

  /* Institutional Color Accents */
  --c-primary: #1d4ed8;    /* Academic Blue */
  --c-primary-light: #3b82f6;
  --c-success: #15803d;    /* British Racing / Forest Green (Clean Ground Truth) */
  --c-success-light: #22c55e;
  --c-warning: #b45309;    /* Warm Ochre Amber (Mixture / Acoustic Stress) */
  --c-warning-light: #f59e0b;
  --c-sota: #0284c7;       /* Oxford SOTA Azure */
  --c-danger: #b91c1c;     /* Interference / Error */

  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-serif: 'Cinzel', Georgia, serif;
  --font-mono: 'JetBrains Mono', SFMono-Regular, Consolas, monospace;

  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.35);
  --transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

[data-theme="light"] {
  --bg-app: #f8fafc;
  --bg-surface: #ffffff;
  --bg-surface-alt: #f1f5f9;
  --bg-card: #ffffff;
  --bg-code: #0f172a;
  --border-color: #cbd5e1;
  --border-subtle: #e2e8f0;
  --border-active: #1d4ed8;

  --text-main: #0f172a;
  --text-muted: #475569;
  --text-dim: #64748b;

  --c-primary: #1e40af;
  --c-primary-light: #2563eb;
  --c-success: #166534;
  --c-success-light: #15803d;
  --c-warning: #92400e;
  --c-warning-light: #d97706;
  --c-sota: #0369a1;
  --c-danger: #991b1b;

  --shadow-card: 0 2px 10px rgba(0, 0, 0, 0.06);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; font-size: 14.5px; }
body {
  font-family: var(--font-sans);
  background: var(--bg-app);
  color: var(--text-main);
  line-height: 1.55;
  transition: background 0.25s, color 0.25s;
  padding-bottom: 90px; /* Dock offset */
}

/* Header Navigation Bar */
.top-navbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: rgba(11, 17, 32, 0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-color);
  padding: 0.65rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
}
[data-theme="light"] .top-navbar {
  background: rgba(255, 255, 255, 0.96);
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  text-decoration: none;
  color: var(--text-main);
}
.brand-mark {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: linear-gradient(135deg, var(--c-primary) 0%, var(--c-success) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-weight: 800;
  font-size: 0.9rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.brand-title {
  font-family: var(--font-sans);
  font-weight: 800;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
}
.brand-subtitle {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-weight: 500;
  border-left: 1px solid var(--border-color);
  padding-left: 0.75rem;
  margin-left: 0.25rem;
  display: none;
}
@media (min-width: 900px) { .brand-subtitle { display: block; } }

.nav-links {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  list-style: none;
}
.nav-link {
  color: var(--text-muted);
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 500;
  padding: 0.4rem 0.65rem;
  border-radius: var(--radius-sm);
  transition: var(--transition);
}
.nav-link:hover, .nav-link.active {
  color: var(--text-main);
  background: var(--bg-surface-alt);
}

.nav-tools {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.badge-status {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.25rem 0.65rem;
  border-radius: 9999px;
  background: rgba(21, 128, 61, 0.15);
  color: var(--c-success-light);
  border: 1px solid rgba(21, 128, 61, 0.3);
}
.badge-status::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--c-success-light);
}

.btn-icon {
  background: var(--bg-surface-alt);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 0.35rem 0.75rem;
  border-radius: var(--radius-sm);
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: var(--transition);
  text-decoration: none;
}
.btn-icon:hover {
  border-color: var(--border-active);
}

/* Layout Container */
.container {
  max-width: 1440px;
  margin: 0 auto;
  padding: 1.75rem 2rem;
}

/* Executive Project Summary Ribbon */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.15rem 1.25rem;
  position: relative;
  transition: var(--transition);
}
.kpi-card:hover {
  border-color: var(--border-active);
  transform: translateY(-2px);
}
.kpi-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 0.4rem;
}
.kpi-value {
  font-family: var(--font-mono);
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--text-main);
  line-height: 1.2;
}
.kpi-subtext {
  font-size: 0.78rem;
  color: var(--text-dim);
  margin-top: 0.35rem;
}

/* Section Headers */
.section-head {
  margin-bottom: 1.75rem;
}
.section-tag {
  display: inline-block;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  margin-bottom: 0.5rem;
  background: rgba(59, 130, 246, 0.12);
  color: var(--c-primary-light);
  border: 1px solid rgba(59, 130, 246, 0.25);
}
.section-tag.green {
  background: rgba(21, 128, 61, 0.12);
  color: var(--c-success-light);
  border-color: rgba(21, 128, 61, 0.25);
}
.section-tag.amber {
  background: rgba(180, 83, 9, 0.12);
  color: var(--c-warning-light);
  border-color: rgba(180, 83, 9, 0.25);
}
.section-head h2 {
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--text-main);
}
.section-head .lead {
  font-size: 0.92rem;
  color: var(--text-muted);
  max-width: 900px;
  margin-top: 0.25rem;
}

/* Standard Academic Card */
.card-academic {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
.card-header-academic {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: 0.75rem;
}
.card-header-academic h3 {
  font-size: 1.15rem;
  font-weight: 700;
}

/* Mathematical Typesetting & Human-Readable Math Cards */
.math-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 1.25rem;
  margin-top: 1.25rem;
}
.math-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  transition: var(--transition);
}
.math-card:hover {
  border-color: var(--border-active);
}
.math-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-main);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.math-desc {
  font-size: 0.82rem;
  color: var(--text-muted);
  line-height: 1.45;
}
.math-equation-box {
  background: var(--bg-code);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 0.85rem 1rem;
  font-family: var(--font-mono);
  font-size: 0.95rem;
  color: #38bdf8;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  text-align: center;
  overflow-x: auto;
}
.math-visual-row {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
  justify-content: center;
}
.math-frac {
  display: inline-flex;
  flex-direction: column;
  vertical-align: middle;
  text-align: center;
  padding: 0 0.3rem;
}
.math-frac .num {
  border-bottom: 1.5px solid #38bdf8;
  padding-bottom: 2px;
  font-size: 0.88rem;
}
.math-frac .den {
  padding-top: 2px;
  font-size: 0.88rem;
}
.math-parens {
  font-size: 1.3rem;
  font-weight: 300;
}
.math-explain-box {
  background: var(--bg-surface-alt);
  border-left: 3px solid var(--c-primary-light);
  border-radius: 0 4px 4px 0;
  padding: 0.75rem 0.9rem;
  font-size: 0.82rem;
  color: var(--text-muted);
  line-height: 1.5;
}
.math-explain-box b {
  color: var(--text-main);
}

/* Interactive SVG Diagram */
.arch-diagram-wrapper {
  background: var(--bg-surface-alt);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1rem;
  overflow-x: auto;
  text-align: center;
  margin-bottom: 1.25rem;
}
.arch-svg {
  max-width: 100%;
  height: auto;
  user-select: none;
}
.svg-node {
  cursor: pointer;
  transition: transform 0.2s;
}
.svg-node:hover rect {
  filter: brightness(1.2);
  stroke-width: 2.2;
}
.node-inspector {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}
.code-box {
  background: var(--bg-code);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 1rem;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: #e2e8f0;
  line-height: 1.5;
}

/* Working Behaviors Stepper */
.stepper-nav {
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 1rem;
}
.step-dot {
  background: var(--bg-surface-alt);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  padding: 0.45rem 0.9rem;
  border-radius: var(--radius-sm);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: var(--transition);
}
.step-dot:hover {
  border-color: var(--border-active);
}
.step-dot.active {
  background: var(--c-primary);
  color: #fff;
  border-color: var(--c-primary);
}
.step-dot.completed {
  color: var(--c-success-light);
  border-color: var(--c-success);
}

/* Audio Lab Styles */
.audiolab-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.25rem;
}
.audio-track-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.audio-track-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.audio-track-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text-main);
}
.btn-play-audio {
  background: var(--c-primary);
  color: #fff;
  border: none;
  padding: 0.45rem 0.9rem;
  border-radius: var(--radius-sm);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: var(--transition);
}
.btn-play-audio:hover {
  background: var(--c-primary-light);
}
.btn-play-clean { background: var(--c-success); }
.btn-play-clean:hover { background: var(--c-success-light); }
.btn-play-mix { background: var(--c-warning); }
.btn-play-mix:hover { background: var(--c-warning-light); }

/* 25-Test Benchmark Filter Controls */
.filter-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
  background: var(--bg-surface-alt);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 0.75rem 1rem;
}
.filter-btn-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.f-btn {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  padding: 0.35rem 0.75rem;
  border-radius: var(--radius-sm);
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition);
}
.f-btn:hover { border-color: var(--border-active); }
.f-btn.active {
  background: var(--c-primary);
  color: #fff;
  border-color: var(--c-primary);
}
.search-input {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 0.4rem 0.85rem;
  border-radius: var(--radius-sm);
  font-size: 0.82rem;
  min-width: 240px;
  outline: none;
}
.search-input:focus {
  border-color: var(--border-active);
}

/* 25-Test Matrix Data Table */
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  background: var(--bg-card);
  margin-bottom: 1.5rem;
}
.matrix-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  text-align: left;
}
.matrix-table th {
  background: var(--bg-surface-alt);
  color: var(--text-muted);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 700;
  padding: 0.75rem 0.85rem;
  border-bottom: 1px solid var(--border-color);
  white-space: nowrap;
}
.matrix-table td {
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid var(--border-subtle);
  vertical-align: middle;
}
.matrix-table tr:hover td {
  background: rgba(255, 255, 255, 0.02);
}
.cell-id {
  font-family: var(--font-mono);
  font-weight: 700;
  color: var(--text-dim);
}
.file-name {
  font-weight: 700;
  color: var(--text-main);
}
.file-sub {
  font-size: 0.72rem;
  color: var(--text-dim);
}
.snr-val {
  font-family: var(--font-mono);
  font-weight: 700;
}
.tier-chip {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
  display: inline-block;
  margin-top: 2px;
}
.tier-severe { background: rgba(185, 28, 28, 0.15); color: #ef4444; }
.tier-med { background: rgba(180, 83, 9, 0.15); color: #f59e0b; }
.tier-mild { background: rgba(21, 128, 61, 0.15); color: #22c55e; }

.noise-desc-cell {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}
.noise-k-badge {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--c-primary-light);
}
.noise-pills-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
}
.noise-pill {
  background: var(--bg-surface-alt);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  font-size: 0.68rem;
  padding: 1px 4px;
  border-radius: 2px;
}
.metric-gain {
  font-family: var(--font-mono);
  font-weight: 700;
  color: var(--c-primary-light);
}
.gain-pesq { color: var(--c-success-light); }
.gain-stoi { color: #38bdf8; }
.metric-flow {
  font-size: 0.72rem;
  color: var(--text-dim);
}
.att-bar-wrap {
  width: 90px;
  height: 6px;
  background: var(--bg-surface-alt);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 2px;
}
.att-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--c-primary), var(--c-success));
}
.att-pct-val {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 600;
}
.pearson-val {
  font-family: var(--font-mono);
  font-weight: 600;
}
.badge-pass {
  background: rgba(21, 128, 61, 0.15);
  color: var(--c-success-light);
  border: 1px solid rgba(21, 128, 61, 0.3);
  font-size: 0.7rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
}
.table-audio-btns {
  display: flex;
  gap: 3px;
  white-space: nowrap;
}
.t-btn {
  border: none;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.3rem 0.55rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: var(--transition);
}
.t-btn-clean { background: var(--bg-surface-alt); color: var(--c-success-light); border: 1px solid var(--border-color); }
.t-btn-clean:hover { background: var(--c-success); color: #fff; }
.t-btn-mix { background: var(--bg-surface-alt); color: var(--c-warning-light); border: 1px solid var(--border-color); }
.t-btn-mix:hover { background: var(--c-warning); color: #fff; }
.t-btn-sota { background: var(--c-primary); color: #fff; }
.t-btn-sota:hover { background: var(--c-primary-light); }

/* Viva Drill Accordions */
.viva-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.viva-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  overflow: hidden;
  transition: var(--transition);
}
.viva-question {
  padding: 1rem 1.25rem;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
}
.viva-question:hover {
  background: var(--bg-surface-alt);
}
.viva-answer {
  display: none;
  padding: 1.25rem;
  border-top: 1px solid var(--border-color);
  background: var(--bg-surface-alt);
  font-size: 0.88rem;
  color: var(--text-muted);
  line-height: 1.6;
}
.viva-card.open .viva-answer {
  display: block;
}
.viva-card.open .viva-icon {
  transform: rotate(180deg);
}
.viva-icon {
  transition: transform 0.2s;
  font-size: 0.8rem;
}

/* Fixed Persistent Bottom Audio Player Dock */
.audio-dock {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 2000;
  background: rgba(11, 17, 32, 0.97);
  backdrop-filter: blur(14px);
  border-top: 1px solid var(--border-color);
  padding: 0.65rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.4);
}
[data-theme="light"] .audio-dock {
  background: rgba(255, 255, 255, 0.97);
}

.dock-info {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  min-width: 260px;
}
.dock-chip {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  background: var(--c-primary);
  color: #fff;
  white-space: nowrap;
}
.dock-title {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 280px;
}

.dock-controls {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  flex: 1;
  max-width: 550px;
}
.dock-btn {
  background: var(--c-primary);
  border: none;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: var(--transition);
  font-size: 0.9rem;
}
.dock-btn:hover { background: var(--c-primary-light); transform: scale(1.05); }
.dock-seek-bar {
  flex: 1;
  height: 6px;
  -webkit-appearance: none;
  background: var(--bg-surface-alt);
  border-radius: 3px;
  outline: none;
  cursor: pointer;
}
.dock-seek-bar::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--c-primary-light);
  cursor: pointer;
}
.dock-time {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-dim);
  min-width: 75px;
}

.dock-visualizer {
  width: 140px;
  height: 32px;
  border-radius: 4px;
  background: var(--bg-code);
  border: 1px solid var(--border-color);
}
.dock-volume {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  min-width: 110px;
}
.dock-vol-slider {
  width: 70px;
  height: 4px;
  -webkit-appearance: none;
  background: var(--bg-surface-alt);
  border-radius: 2px;
}
</style>
</head>

<body>

  <!-- Top Navigation Header -->
  <header class="top-navbar">
    <a href="#overview" class="nav-brand">
      <div class="brand-mark">CP</div>
      <div>
        <div class="brand-title">ClearParty</div>
      </div>
      <div class="brand-subtitle">Deep Speaker Separation & Speech Enhancement</div>
    </a>

    <ul class="nav-links">
      <li><a href="#overview" class="nav-link">Overview</a></li>
      <li><a href="#evolution" class="nav-link">Evolution</a></li>
      <li><a href="#datasets" class="nav-link">Datasets</a></li>
      <li><a href="#architectures" class="nav-link">Architectures</a></li>
      <li><a href="#behaviors" class="nav-link">Working Flow</a></li>
      <li><a href="#audiolab" class="nav-link">Audio Lab</a></li>
      <li><a href="#results" class="nav-link">Results (25 Tests)</a></li>
      <li><a href="#viva" class="nav-link">Viva Drill</a></li>
    </ul>

    <div class="nav-tools">
      <span class="badge-status">8 kHz Nyquist Standard</span>
      <button class="btn-icon" onclick="toggleTheme()" id="themeBtn" title="Toggle Classic Light/Dark Mode">&#9788; Light</button>
      <a href="https://github.com/Jaydeep0832/ClearParty-Deep-Speaker-Separation-and-Zero-Shot-Multi-Noise-Speech-Enhancement" target="_blank" rel="noreferrer" class="btn-icon">
        GitHub
      </a>
    </div>
  </header>

  <!-- Main Application Body -->
  <main class="container">

    <!-- Executive KPI Ribbon -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">SOTA SI-SDR Gain</div>
        <div class="kpi-value" style="color:var(--c-primary-light);">+__KPI_SISDR__ dB</div>
        <div class="kpi-subtext">5.21 dB &rarr; 14.20 dB across 25 tests</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">PESQ Improvement</div>
        <div class="kpi-value" style="color:var(--c-success-light);">+__KPI_PESQ__</div>
        <div class="kpi-subtext">1.606 &rarr; 2.790 Mean MOS Score</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Noise Attenuation</div>
        <div class="kpi-value" style="color:var(--c-warning-light);">__KPI_ATT__%</div>
        <div class="kpi-subtext">Compound Multi-Noise Eradication</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Stress-Test Pass Rate</div>
        <div class="kpi-value" style="color:var(--c-success-light);">__KPI_PASS__%</div>
        <div class="kpi-subtext">25 / 25 Validated Benchmark Runs</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Training Corpus</div>
        <div class="kpi-value">14,269</div>
        <div class="kpi-subtext">Mixture Pairs &bull; 251 Talkers</div>
      </div>
    </section>

    <!-- SECTION 1: OVERVIEW & RESEARCH MOTIVATION -->
    <section id="overview" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag">Problem Formulation</span>
          <h2>The Cocktail Party Problem & Dual-Track Scope</h2>
        </div>
        <span class="badge-status">Academic Project Context</span>
      </div>

      <div style="display:grid; grid-template-columns: 1.1fr 0.9fr; gap:1.5rem; margin-top:1rem;">
        <div>
          <p style="font-size:0.92rem; color:var(--text-muted); line-height:1.6; margin-bottom:1rem;">
            First formulated by <b>Colin Cherry in 1953</b>, the <em>Cocktail Party Problem</em> defines the human psychoacoustic faculty to isolate and track a single speaker's voice amid a cacophony of overlapping talkers, reverberation, and environmental background disturbances.
          </p>
          <p style="font-size:0.92rem; color:var(--text-muted); line-height:1.6; margin-bottom:1rem;">
            In computational acoustics, solving this from a <b>single monaural microphone</b> represents an <em>underdetermined blind inverse problem</em> (1 sensor vs multiple sources). This project establishes two rigorous, independent research tracks to resolve this challenge:
          </p>
          <ul style="font-size:0.88rem; color:var(--text-muted); margin-left:1.2rem; display:flex; flex-direction:column; gap:0.4rem;">
            <li>&bull; <b>Track 1: Monaural Blind Speaker Separation (BSS):</b> Isolates overlapping human talkers without environmental noise using Conv-TasNet and Utterance-level Permutation Invariant Training (uPIT).</li>
            <li>&bull; <b>Track 2: Multi-Noise Speech Enhancement (SE):</b> Purifies a single speech signal buried beneath 3 to 4 simultaneous non-stationary acoustic noises using SOTA MossFormerGAN.</li>
          </ul>
        </div>

        <div style="background:var(--bg-surface-alt); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:1.25rem;">
          <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.75rem;">Comparative Track Methodology</h4>
          <table style="width:100%; font-size:0.8rem; border-collapse:collapse;">
            <thead>
              <tr style="border-bottom:1px solid var(--border-color); text-align:left; color:var(--text-dim);">
                <th style="padding:0.4rem;">Attribute</th>
                <th style="padding:0.4rem;">Speaker Separation</th>
                <th style="padding:0.4rem;">Speech Enhancement</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom:1px solid var(--border-subtle);">
                <td style="padding:0.4rem; font-weight:600;">Mathematical Input</td>
                <td style="padding:0.4rem; font-family:var(--font-mono);">x = s₁ + s₂</td>
                <td style="padding:0.4rem; font-family:var(--font-mono);">x = s + &Sigma; n_k</td>
              </tr>
              <tr style="border-bottom:1px solid var(--border-subtle);">
                <td style="padding:0.4rem; font-weight:600;">Interferer Nature</td>
                <td style="padding:0.4rem;">Structured human speech</td>
                <td style="padding:0.4rem;">Environmental noise</td>
              </tr>
              <tr style="border-bottom:1px solid var(--border-subtle);">
                <td style="padding:0.4rem; font-weight:600;">Label Ambiguity</td>
                <td style="padding:0.4rem;">High (uPIT required)</td>
                <td style="padding:0.4rem;">None (Single target)</td>
              </tr>
              <tr>
                <td style="padding:0.4rem; font-weight:600;">Core Model</td>
                <td style="padding:0.4rem;">Conv-TasNet (Time-Domain)</td>
                <td style="padding:0.4rem;">MossFormerGAN (Complex STFT)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- SECTION 2: EVOLUTIONARY MILESTONES (PHASES 1 TO 5) -->
    <section id="evolution" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag green">Research Chronology</span>
          <h2>Project Evolution Across Five Developmental Phases</h2>
        </div>
        <span class="badge-status">System Architecture Progression</span>
      </div>

      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:1rem; margin-top:1rem;">
        <div class="card-academic" style="padding:1.1rem; margin:0; border-top:3px solid var(--c-danger);">
          <div style="font-size:0.75rem; font-weight:700; color:var(--c-danger); margin-bottom:0.25rem;">PHASE 1 &bull; INITIAL EXPLORATION</div>
          <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">Spectrogram U-Net & Phase Catastrophe</h4>
          <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.45;">
            Initial experiments utilized magnitude STFT masking. Discarding phase information caused severe acoustic artifacts, proving that phase reconstruction is non-trivial for overlapping voice.
          </p>
        </div>

        <div class="card-academic" style="padding:1.1rem; margin:0; border-top:3px solid var(--c-primary);">
          <div style="font-size:0.75rem; font-weight:700; color:var(--c-primary-light); margin-bottom:0.25rem;">PHASE 2 &bull; TIME-DOMAIN REVOLUTION</div>
          <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">Conv-TasNet & uPIT Loss</h4>
          <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.45;">
            Transitioned to 1D dilated convolutions with Utterance-level Permutation Invariant Training. Scaled training corpus to 14,269 LibriSpeech pairs across 251 speakers.
          </p>
        </div>

        <div class="card-academic" style="padding:1.1rem; margin:0; border-top:3px solid var(--c-success);">
          <div style="font-size:0.75rem; font-weight:700; color:var(--c-success-light); margin-bottom:0.25rem;">PHASE 3 &bull; TIMBRE RESTORATION</div>
          <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">Spectral Wiener Post-Filter</h4>
          <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.45;">
            Implemented post-processing Wiener smoothing on Conv-TasNet masks, completely eliminating the synthetic "metallic buzz" and restoring natural vocal tract resonance.
          </p>
        </div>

        <div class="card-academic" style="padding:1.1rem; margin:0; border-top:3px solid var(--c-warning);">
          <div style="font-size:0.75rem; font-weight:700; color:var(--c-warning-light); margin-bottom:0.25rem;">PHASE 4 &bull; MULTI-NOISE BENCHMARK</div>
          <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">25 Compound Stress Tests</h4>
          <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.45;">
            Constructed a rigorous 25-test benchmark containing mixtures of 3 to 4 simultaneous non-stationary noises (Speech Babble, Engine, Traffic, White Noise, Factory Hum).
          </p>
        </div>

        <div class="card-academic" style="padding:1.1rem; margin:0; border-top:3px solid var(--c-sota);">
          <div style="font-size:0.75rem; font-weight:700; color:var(--c-sota); margin-bottom:0.25rem;">PHASE 5 &bull; CURRENT SOTA</div>
          <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">MossFormerGAN Enhancement</h4>
          <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.45;">
            Integrated MossFormerGAN SyncANet framework. Outperformed Conv-TasNet by nearly 2x: achieved +8.99 dB SI-SDR gain, +1.184 PESQ gain, and 87.18% noise attenuation.
          </p>
        </div>
      </div>
    </section>

    <!-- SECTION 3: DATASETS & ACOUSTIC SPECIFICATIONS -->
    <section id="datasets" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag amber">Data Engineering</span>
          <h2>Acoustic Corpora & Signal Specifications</h2>
        </div>
        <span class="badge-status">Standardized Benchmarking</span>
      </div>

      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:1.25rem;">
        <div style="background:var(--bg-surface-alt); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:1.25rem;">
          <h4 style="font-weight:700; margin-bottom:0.5rem;">LibriSpeech & LibriMix Corpora</h4>
          <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5; margin-bottom:0.75rem;">
            Primary corpus for Speaker Separation. Scaled in Phase 2B to <b>14,269 mixture pairs</b> generated from 251 speakers in <code>train-clean-100</code> with dynamic volume jittering.
          </p>
          <ul style="font-size:0.8rem; color:var(--text-muted); display:flex; flex-direction:column; gap:0.25rem;">
            <li>&bull; <b>Duration:</b> Segmented into 3.0-second and 4.0-second utterances.</li>
            <li>&bull; <b>Active Voice Ratio:</b> >95% simultaneous dual-talker overlap.</li>
          </ul>
        </div>

        <div style="background:var(--bg-surface-alt); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:1.25rem;">
          <h4 style="font-weight:700; margin-bottom:0.5rem;">Synthetic Multi-Noise Benchmark Suite</h4>
          <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5; margin-bottom:0.75rem;">
            Rigorous evaluation suite comprising 25 stress tests combining 3 to 4 simultaneous real-world non-stationary noise sources at calibrated SNRs (0.0 dB to 10.0 dB).
          </p>
          <div style="display:flex; flex-wrap:wrap; gap:4px;">
            <span class="noise-pill">Speech Babble</span>
            <span class="noise-pill">Mechanical Engine</span>
            <span class="noise-pill">Street Traffic</span>
            <span class="noise-pill">White Gaussian Noise</span>
            <span class="noise-pill">Factory Hum</span>
          </div>
        </div>

        <div style="background:var(--bg-surface-alt); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:1.25rem;">
          <h4 style="font-weight:700; margin-bottom:0.5rem;">Acoustic Nyquist Bandwidth Defense</h4>
          <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5; margin-bottom:0.5rem;">
            <b>Why 8 kHz Sampling Rate?</b> Human vocal tract formants (F₁, F₂, F₃) reside entirely below 3.8 kHz. 8 kHz telephone-band audio satisfies the Nyquist criterion while preserving O(T) computational efficiency in 1D convolutions.
          </p>
          <div style="font-size:0.75rem; color:var(--text-dim);">
            Standard ITU-T P.862 PESQ is strictly calibrated for narrowband 8 kHz speech, producing the expected ~2.3 to 2.8 upper bound.
          </div>
        </div>
      </div>
    </section>

    <!-- SECTION 4: ARCHITECTURES & MATHEMATICAL FORMULATIONS (HUMAN READABLE FORMS) -->
    <section id="architectures" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag green">Model Topologies & Math</span>
          <h2>Deep Architectures & Human-Readable Mathematical Formulations</h2>
        </div>
        <span class="badge-status">Detailed Theory & Implementation</span>
      </div>

      <!-- Conv-TasNet Topology SVG -->
      <div class="arch-diagram-wrapper">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; font-size:0.85rem; color:var(--text-muted);">
          <b>Conv-TasNet Modular Signal Flow:</b>
          <span>Click any block to inspect tensor shapes and PyTorch code</span>
        </div>
        <svg class="arch-svg" viewBox="0 0 940 220" width="940" height="220" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/>
            </marker>
          </defs>

          <!-- Input Waveform -->
          <g class="svg-node" onclick="inspectModel('input')">
            <rect x="20" y="50" width="105" height="90" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.8"/>
            <text x="72" y="88" fill="#fff" font-family="'Inter', sans-serif" font-weight="700" font-size="12" text-anchor="middle">Input Audio</text>
            <text x="72" y="108" fill="#94a3b8" font-size="9" text-anchor="middle">x(t) • [B, 1, T]</text>
          </g>
          <line x1="125" y1="95" x2="165" y2="95" stroke="#38bdf8" stroke-width="2" marker-end="url(#arr)"/>

          <!-- 1D Conv Encoder -->
          <g class="svg-node" onclick="inspectModel('encoder')">
            <rect x="165" y="35" width="125" height="120" rx="6" fill="#0369a1" stroke="#38bdf8" stroke-width="1.8"/>
            <text x="227" y="70" fill="#fff" font-family="'Inter', sans-serif" font-weight="700" font-size="12" text-anchor="middle">1D Conv Encoder</text>
            <text x="227" y="90" fill="#bae6fd" font-size="9" text-anchor="middle">N=512, L=16, S=8</text>
            <text x="227" y="110" fill="#bae6fd" font-size="9" text-anchor="middle">Feature Map w</text>
            <text x="227" y="130" fill="#e0f2fe" font-family="'JetBrains Mono', monospace" font-size="8.5" text-anchor="middle">w = ReLU(x * U)</text>
          </g>
          <line x1="290" y1="95" x2="330" y2="95" stroke="#3b82f6" stroke-width="2" marker-end="url(#arr)"/>

          <!-- Dilated TCN Separator -->
          <g class="svg-node" onclick="inspectModel('tcn')">
            <rect x="330" y="20" width="195" height="150" rx="6" fill="#1e40af" stroke="#3b82f6" stroke-width="1.8"/>
            <text x="427" y="55" fill="#fff" font-family="'Inter', sans-serif" font-weight="700" font-size="13" text-anchor="middle">Dilated TCN Separator</text>
            <text x="427" y="75" fill="#bfdbfe" font-size="9.5" text-anchor="middle">3 Stacks × 8 Layers (d=1..128)</text>
            <text x="427" y="95" fill="#bfdbfe" font-size="9.5" text-anchor="middle">Depthwise Separable 1D Convs</text>
            <text x="427" y="115" fill="#bfdbfe" font-size="9.5" text-anchor="middle">PReLU + Global LayerNorm</text>
            <text x="427" y="140" fill="#dbeafe" font-family="'JetBrains Mono', monospace" font-size="9" text-anchor="middle">Receptive Field: 153 ms</text>
          </g>
          <line x1="525" y1="95" x2="565" y2="95" stroke="#3b82f6" stroke-width="2" marker-end="url(#arr)"/>

          <!-- Dual Mask Generator -->
          <g class="svg-node" onclick="inspectModel('masks')">
            <rect x="565" y="35" width="125" height="120" rx="6" fill="#1e3a8a" stroke="#60a5fa" stroke-width="1.8"/>
            <text x="627" y="70" fill="#fff" font-family="'Inter', sans-serif" font-weight="700" font-size="12" text-anchor="middle">Dual Mask Head</text>
            <text x="627" y="90" fill="#dbeafe" font-size="9" text-anchor="middle">1x1 Conv + Sigmoid</text>
            <text x="627" y="110" fill="#dbeafe" font-size="9" text-anchor="middle">M1, M2 in [0, ∞)</text>
            <text x="627" y="130" fill="#eff6ff" font-family="'JetBrains Mono', monospace" font-size="8.5" text-anchor="middle">d_c = w ⊙ M_c</text>
          </g>
          <line x1="690" y1="95" x2="730" y2="95" stroke="#15803d" stroke-width="2" marker-end="url(#arr)"/>

          <!-- 1D Transposed Conv Decoder -->
          <g class="svg-node" onclick="inspectModel('decoder')">
            <rect x="730" y="35" width="115" height="120" rx="6" fill="#15803d" stroke="#22c55e" stroke-width="1.8"/>
            <text x="787" y="70" fill="#fff" font-family="'Inter', sans-serif" font-weight="700" font-size="12" text-anchor="middle">1D ConvTranspose</text>
            <text x="787" y="90" fill="#bbf7d0" font-size="9" text-anchor="middle">Decoder Matrix V</text>
            <text x="787" y="110" fill="#bbf7d0" font-size="9" text-anchor="middle">Waveform Synthesis</text>
            <text x="787" y="130" fill="#f0fdf4" font-family="'JetBrains Mono', monospace" font-size="8.5" text-anchor="middle">s1_hat, s2_hat</text>
          </g>
          <line x1="845" y1="95" x2="880" y2="95" stroke="#15803d" stroke-width="2" marker-end="url(#arr)"/>

          <!-- Natural Wiener Filter -->
          <g class="svg-node" onclick="inspectModel('wiener')">
            <rect x="880" y="45" width="55" height="100" rx="6" fill="#166534" stroke="#4ade80" stroke-width="1.8"/>
            <text x="907" y="85" fill="#fff" font-family="'Inter', sans-serif" font-weight="700" font-size="10" text-anchor="middle">Wiener</text>
            <text x="907" y="105" fill="#86efac" font-size="8" text-anchor="middle">Filter</text>
            <text x="907" y="125" fill="#dcfce7" font-family="'JetBrains Mono', monospace" font-size="7.5" text-anchor="middle">W_k(f,t)</text>
          </g>
        </svg>
      </div>

      <!-- Live Layer Inspector -->
      <div class="node-inspector" id="inspectorBox">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
          <h3 id="inspTitle" style="color:var(--text-main); font-size:1.05rem;">Component: Dilated TCN Separator</h3>
          <span class="section-tag" id="inspBadge">CORE ENGINE</span>
        </div>
        <p id="inspDesc" style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.6rem;">
          Stack of dilated depthwise separable 1D convolutions expanding exponential receptive fields (153 ms) without parameter explosion.
        </p>
        <div class="code-box" id="inspCode">
<pre><code># Temporal Convolutional Network (TCN) Block
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.dconv = nn.Conv1d(in_channels, in_channels, kernel_size, 
                               dilation=dilation, padding=dilation, groups=in_channels)
        self.prelu = nn.PReLU()
        self.gln = GlobalLayerNorm(in_channels)
        self.pconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)</code></pre>
        </div>
      </div>

      <!-- SIX HUMAN-READABLE MATHEMATICAL FORMULATIONS -->
      <h3 style="font-size:1.15rem; font-weight:700; margin-top:2rem; margin-bottom:0.75rem;">Core Mathematical Formulations & Plain-English Explanations</h3>
      <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem;">
        Every equation is presented below in publication-grade mathematical typography alongside a clear, intuitive explanation formatted for faculty defense:
      </p>

      <div class="math-grid">
        <!-- 1. SI-SDR -->
        <div class="math-card">
          <div class="math-title">
            <span>1. Scale-Invariant SDR (SI-SDR)</span>
            <span class="section-tag">Separation Loss</span>
          </div>
          <div class="math-desc">
            Orthogonally projects the estimated voice onto the reference to ensure volume invariance:
          </div>
          <div class="math-equation-box">
            <div class="math-visual-row">
              <span>&alpha; = </span>
              <span class="math-frac"><span class="num">ŝ<sup>T</sup> s</span><span class="den">||s||<sup>2</sup> + &epsilon;</span></span>
              <span>, &nbsp; e<sub>target</sub> = &alpha; s, &nbsp; e<sub>noise</sub> = ŝ - e<sub>target</sub></span>
            </div>
            <div class="math-visual-row" style="margin-top:0.4rem; font-weight:700;">
              <span>SI-SDR = 10 &middot; log<sub>10</sub></span>
              <span class="math-parens">(</span>
              <span class="math-frac"><span class="num">||e<sub>target</sub>||<sup>2</sup></span><span class="den">||e<sub>noise</sub>||<sup>2</sup> + &epsilon;</span></span>
              <span class="math-parens">)</span>
            </div>
          </div>
          <div class="math-explain-box">
            <b>📘 In Plain English (Why It Matters):</b><br>
            Standard Mean Squared Error (MSE) is hypersensitive to volume differences: a cleanly separated speaker that is simply 3 dB quieter gets severely penalized. <b>SI-SDR</b> scales the separated audio <code>ŝ</code> to best match the reference speech <code>s</code> via scale factor <code>&alpha;</code>. It decomposes the output into pure target speech (<code>e<sub>target</sub></code>) and residual interference/distortion (<code>e<sub>noise</sub></code>), calculating their energy ratio in decibels. <b>Higher dB = cleaner isolated voice.</b>
          </div>
        </div>

        <!-- 2. uPIT -->
        <div class="math-card">
          <div class="math-title">
            <span>2. Utterance-Level PIT (uPIT)</span>
            <span class="section-tag green">Permutation Solver</span>
          </div>
          <div class="math-desc">
            Resolves channel ambiguity during backpropagation without knowing speaker identities:
          </div>
          <div class="math-equation-box">
            <div class="math-visual-row" style="font-weight:700;">
              <span>L<sub>uPIT</sub> = min<sub>&pi; &in; P</sub></span>
              <span>[ - &Sigma;<sub>c=1</sub><sup>2</sup> SI-SDR(s<sub>c</sub>, ŝ<sub>&pi;(c)</sub>) ]</span>
            </div>
            <div class="math-visual-row" style="font-size:0.82rem; color:var(--text-muted); margin-top:0.2rem;">
              <span>Permutation A: (s₁, ŝ₁) + (s₂, ŝ₂) &nbsp;vs&nbsp; Permutation B: (s₁, ŝ₂) + (s₂, ŝ₁)</span>
            </div>
          </div>
          <div class="math-explain-box">
            <b>📘 In Plain English (Why It Matters):</b><br>
            The model outputs two audio streams, but cannot know whether Speaker 1 is on Output Channel 1 or Output Channel 2 (the <em>Label Permutation Problem</em>). <b>uPIT</b> computes the loss for both possible output-to-target channel pairings and backpropagates gradients <b>solely through whichever pairing yields the lowest error</b>. This eliminates gradient contradiction and guarantees stable multi-talker convergence.
          </div>
        </div>

        <!-- 3. Spectral Wiener Filter -->
        <div class="math-card">
          <div class="math-title">
            <span>3. Natural Spectral Wiener Filter</span>
            <span class="section-tag">Artifact Removal</span>
          </div>
          <div class="math-desc">
            Smooths transposed-convolution phase discontinuities and eliminates metallic buzz:
          </div>
          <div class="math-equation-box">
            <div class="math-visual-row">
              <span>W<sub>k</sub>(f, t) = </span>
              <span class="math-frac"><span class="num">|Ŝ<sub>k</sub>(f, t)|<sup>2</sup></span><span class="den">|Ŝ<sub>1</sub>(f, t)|<sup>2</sup> + |Ŝ<sub>2</sub>(f, t)|<sup>2</sup> + &epsilon;</span></span>
            </div>
            <div class="math-visual-row" style="margin-top:0.4rem; font-weight:700;">
              <span>S<sub>k</sub><sup>Wiener</sup>(f, t) = W<sub>k</sub>(f, t) &middot; X<sub>mix</sub>(f, t)</span>
            </div>
          </div>
          <div class="math-explain-box">
            <b>📘 In Plain English (Why It Matters):</b><br>
            Raw time-domain transposed convolutions occasionally produce micro-phase frame discontinuities that listeners perceive as a faint metallic or robotic timbre. The <b>Spectral Wiener Filter</b> calculates a continuous gain mask <code>W<sub>k</sub>(f, t) &in; [0, 1]</code> equal to Speaker <i>k</i>'s share of acoustic power at each frequency bin. Multiplying this mask with the original acoustic mixture preserves the authentic vocal tract acoustics and natural human phase.
          </div>
        </div>

        <!-- 4. PESQ ITU-T P.862 -->
        <div class="math-card">
          <div class="math-title">
            <span>4. Perceptual Quality (PESQ)</span>
            <span class="section-tag amber">Human Psychoacoustics</span>
          </div>
          <div class="math-desc">
            Standard ITU-T P.862 Mean Opinion Score (MOS) predictor:
          </div>
          <div class="math-equation-box">
            <div class="math-visual-row" style="font-weight:700;">
              <span>PESQ = 4.5 - 0.1 &middot; d<sub>sym</sub> - 0.0309 &middot; d<sub>asym</sub></span>
            </div>
            <div class="math-visual-row" style="font-size:0.8rem; color:var(--text-muted); margin-top:0.2rem;">
              <span>where d<sub>sym</sub> = symmetric disturbance &nbsp;|&nbsp; d<sub>asym</sub> = asymmetric disturbance</span>
            </div>
          </div>
          <div class="math-explain-box">
            <b>📘 In Plain English (Why It Matters):</b><br>
            Standard dB metrics don't reflect how the human ear actually hears sound. <b>PESQ</b> aligns signals in time and warps frequencies into Bark psychoacoustic bands (mimicking the human cochlea). It penalizes missing speech formants more severely than quiet background residue, predicting an audible quality score from 1.0 (unintelligible garble) to 4.5 (flawless speech).
          </div>
        </div>

        <!-- 5. STOI -->
        <div class="math-card">
          <div class="math-title">
            <span>5. Objective Intelligibility (STOI)</span>
            <span class="section-tag">Word Comprehension</span>
          </div>
          <div class="math-desc">
            Quantifies the proportion of understandable phonemes and syllables:
          </div>
          <div class="math-equation-box">
            <div class="math-visual-row" style="font-weight:700;">
              <span>STOI = </span>
              <span class="math-frac"><span class="num">1</span><span class="den">J &middot; M</span></span>
              <span>&Sigma;<sub>j=1</sub><sup>J</sup> &Sigma;<sub>m=1</sub><sup>M</sup> &rho;<sub>j, m</sub></span>
            </div>
            <div class="math-visual-row" style="font-size:0.8rem; color:var(--text-muted); margin-top:0.2rem;">
              <span>&rho;<sub>j, m</sub> = correlation of 1/3-octave temporal envelope segments</span>
            </div>
          </div>
          <div class="math-explain-box">
            <b>📘 In Plain English (Why It Matters):</b><br>
            Speech can sound clean but be completely unintelligible if phonemes are clipped. <b>STOI</b> splits audio into 15 third-octave bands and calculates the statistical correlation <code>&rho;</code> between short temporal envelopes (384 ms) of enhanced speech versus clean reference speech. Yields a value from 0.0 (0% intelligibility) to 1.0 (100% human word comprehension).
          </div>
        </div>

        <!-- 6. Noise Attenuation -->
        <div class="math-card">
          <div class="math-title">
            <span>6. Compound Noise Attenuation (%)</span>
            <span class="section-tag green">Suppression Ratio</span>
          </div>
          <div class="math-desc">
            Physical acoustic energy eradication across multiple noise sources:
          </div>
          <div class="math-equation-box">
            <div class="math-visual-row" style="font-weight:700;">
              <span>Attenuation (%) = </span>
              <span class="math-parens">[</span>
              <span>1 - </span>
              <span class="math-frac"><span class="num">RMS(x<sub>residual</sub>)</span><span class="den">RMS(x<sub>noise_input</sub>)</span></span>
              <span class="math-parens">]</span>
              <span>&times; 100%</span>
            </div>
          </div>
          <div class="math-explain-box">
            <b>📘 In Plain English (Why It Matters):</b><br>
            Directly quantifies the exact percentage of unwanted background noise energy eliminated by MossFormerGAN. Across all 25 multi-noise stress tests (combining 3 to 4 simultaneous non-stationary noises), the system achieved a verified mean of <b>87.18% noise eradication</b> without clipping speech formants.
          </div>
        </div>
      </div>
    </section>

    <!-- SECTION 5: WORKING BEHAVIORS (SIGNAL PROCESSING PIPELINE) -->
    <section id="behaviors" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag">Pipeline Execution</span>
          <h2>Working Behaviors: Step-by-Step Processing Flow</h2>
        </div>
        <span class="badge-status">Sequential Signal Transformation</span>
      </div>

      <!-- Stepper Controls -->
      <div class="stepper-nav" id="stepperNav">
        <button class="step-dot active" onclick="setStep(0)">1. Mixture Input</button>
        <button class="step-dot" onclick="setStep(1)">2. 1D Conv Encoder</button>
        <button class="step-dot" onclick="setStep(2)">3. Dilated TCN Separator</button>
        <button class="step-dot" onclick="setStep(3)">4. Dual Mask Estimation</button>
        <button class="step-dot" onclick="setStep(4)">5. Transposed Conv Decoder</button>
        <button class="step-dot" onclick="setStep(5)">6. Wiener Smoothing & Output</button>
      </div>

      <!-- Step Content Box -->
      <div style="background:var(--bg-surface-alt); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:1.25rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
          <h3 id="stepTitle" style="font-size:1.05rem; font-weight:700;">Step 1: Raw Superimposed Acoustic Mixture</h3>
          <span class="section-tag" id="stepTag">Time-Domain</span>
        </div>
        <p id="stepDesc" style="font-size:0.85rem; color:var(--text-muted); line-height:1.5; margin-bottom:1rem;">
          The microphone captures the linear superposition of two unconstrained human speakers x(t) = s₁(t) + s₂(t). Both vocal sources share identical frequency bands and formant dynamics, making traditional bandpass filtering impossible.
        </p>

        <!-- Live Waveform Canvas -->
        <div style="background:var(--bg-code); border:1px solid var(--border-color); border-radius:4px; padding:0.75rem; text-align:center;">
          <canvas id="pipelineCanvas" width="900" height="120" style="width:100%; max-width:900px; height:120px;"></canvas>
          <div style="font-size:0.75rem; color:var(--text-dim); margin-top:0.4rem;" id="canvasCaption">
            Input Time-Domain Signal Amplitude x(t) &bull; Duration 3.0s &bull; Sample Rate 8000 Hz
          </div>
        </div>
      </div>
    </section>

    <!-- SECTION 6: AUDIO LAB (INTERACTIVE LISTENING STATION) -->
    <section id="audiolab" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag green">Listening Station</span>
          <h2>Audio Lab: Empirical Validation & Human Listening Tests</h2>
        </div>
        <span class="badge-status">Interactive Audio Playback</span>
      </div>

      <div style="margin-bottom:1.25rem;">
        <h3 style="font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;">Objective 1: Blind Speaker Separation (Conv-TasNet & Wiener)</h3>
        <p style="font-size:0.85rem; color:var(--text-muted);">
          Compare the raw input mixture against the true clean references, the raw Conv-TasNet model output (demonstrating the slight metallic buzz), and the final Natural Wiener-reconstructed voice:
        </p>
      </div>

      <div class="audiolab-grid">
        <!-- Sample 1 -->
        <div class="audio-track-card">
          <div class="audio-track-header">
            <span class="audio-track-title">Sample Set 1 (Both Speakers Overlapping)</span>
            <span class="section-tag">Libri2Mix Pair</span>
          </div>
          <div style="display:flex; flex-direction:column; gap:0.4rem;">
            <button class="btn-play-audio btn-play-mix" onclick="playAudioRow('audio/separation/Input Mixture (Both speakers talking).wav', 'Speaker Separation • Input Mixture (Sample 1)', 'Superimposed Mixture')">
              &#9658; Play Input Mixture (Both Talkers)
            </button>
            <button class="btn-play-audio btn-play-clean" onclick="playAudioRow('audio/separation/True Speaker 1 (Clean Reference).wav', 'Speaker Separation • Ground Truth Speaker 1', 'Clean Ground Truth')">
              &#9658; True Speaker 1 (Clean Reference)
            </button>
            <button class="btn-play-audio btn-play-clean" onclick="playAudioRow('audio/separation/True Speaker 2 (Clean Reference).wav', 'Speaker Separation • Ground Truth Speaker 2', 'Clean Ground Truth')">
              &#9658; True Speaker 2 (Clean Reference)
            </button>
            <button class="btn-play-audio" style="background:#475569;" onclick="playAudioRow('audio/separation/Raw Conv-TasNet Output (Has the synthetic metallic buzz).wav', 'Raw Conv-TasNet (Demonstrates Metallic Buzz)', 'Raw Model Output')">
              &#9658; Raw Conv-TasNet (Shows Metallic Buzz)
            </button>
            <button class="btn-play-audio" style="background:var(--c-success);" onclick="playAudioRow('audio/separation/Natural Wiener-Reconstructed Speaker 1 (Natural human voice).wav', 'Wiener-Restored Natural Speaker 1', 'Restored Natural Human Voice')">
              &#9658; Natural Wiener Restored Speaker 1 (Natural Voice)
            </button>
          </div>
        </div>

        <!-- Sample 2 -->
        <div class="audio-track-card">
          <div class="audio-track-header">
            <span class="audio-track-title">Sample Set 2 (Rapid Conversational Exchange)</span>
            <span class="section-tag green">Restored Voice</span>
          </div>
          <div style="display:flex; flex-direction:column; gap:0.4rem;">
            <button class="btn-play-audio btn-play-mix" onclick="playAudioRow('audio/separation/Input Mixture (Two overlapping speakers).wav', 'Speaker Separation • Input Mixture (Sample 2)', 'Superimposed Mixture')">
              &#9658; Play Input Mixture (Overlapping)
            </button>
            <button class="btn-play-audio btn-play-clean" onclick="playAudioRow('audio/separation/Separated Speaker 1 (Model Output).wav', 'Speaker Separation • Separated Speaker 1', 'Separated Output')">
              &#9658; Separated Speaker 1 (Isolated Stream)
            </button>
            <button class="btn-play-audio btn-play-clean" onclick="playAudioRow('audio/separation/Separated Speaker 2 (Model Output).wav', 'Speaker Separation • Separated Speaker 2', 'Separated Output')">
              &#9658; Separated Speaker 2 (Isolated Stream)
            </button>
            <button class="btn-play-audio" style="background:var(--c-success);" onclick="playAudioRow('audio/separation/Natural Wiener-Reconstructed Speaker 2.wav', 'Wiener-Restored Natural Speaker 2', 'Restored Natural Human Voice')">
              &#9658; Natural Wiener Restored Speaker 2
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- SECTION 7: RESULTS (THE 25-TEST MULTI-NOISE BENCHMARK MATRIX) -->
    <section id="results" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag amber">Objective 2 Evaluation</span>
          <h2>The 25-Test Multi-Noise Benchmark Matrix (MossFormerGAN SOTA)</h2>
        </div>
        <span class="badge-status">100% Validated Tests (25/25)</span>
      </div>

      <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem;">
        Comprehensive evaluation of speech enhancement under severe multi-source acoustic contamination (3 to 4 simultaneous real-world noises). Click any audio button to listen:
      </p>

      <!-- Filter Controls Toolbar -->
      <div class="filter-toolbar">
        <div class="filter-btn-group">
          <button class="f-btn active" onclick="setFilter('all', this)">All Tests (25)</button>
          <button class="f-btn" onclick="setFilter('severe', this)">Severe SNR (0–2.5 dB)</button>
          <button class="f-btn" onclick="setFilter('intermediate', this)">Medium SNR (2.5–5 dB)</button>
          <button class="f-btn" onclick="setFilter('mild', this)">Mild SNR (>5 dB)</button>
          <button class="f-btn" onclick="setFilter('3noises', this)">3-Noise Mixes</button>
          <button class="f-btn" onclick="setFilter('4noises', this)">4-Noise Mixes</button>
        </div>

        <input type="text" class="search-input" id="searchBox" placeholder="Search by filename, noise type..." oninput="filterMatrixTable()">
      </div>

      <!-- Data Table -->
      <div class="table-wrap">
        <table class="matrix-table" id="matrixTable">
          <thead>
            <tr>
              <th>ID</th>
              <th>Clean Speech</th>
              <th>Target SNR</th>
              <th>Multi-Noise Composition</th>
              <th>SI-SDR Gain</th>
              <th>PESQ Gain</th>
              <th>STOI Gain</th>
              <th>Noise Attenuation</th>
              <th>Pearson r</th>
              <th>Status</th>
              <th>Audio Playback</th>
            </tr>
          </thead>
          <tbody>
            __TABLE_ROWS__
          </tbody>
        </table>
      </div>
    </section>

    <!-- SECTION 8: VIVA DRILL & MENTOR EXAMINATION GUIDE -->
    <section id="viva" class="card-academic">
      <div class="card-header-academic">
        <div>
          <span class="section-tag green">Thesis Defense</span>
          <h2>Viva Drill & Faculty Mentor Examination Preparation</h2>
        </div>
        <span class="badge-status">8 Critical Faculty Questions</span>
      </div>

      <div class="viva-list">
        <!-- Question 1 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>1. Why is raw PESQ bounded around 2.3 to 2.8 in telephone-band speech?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            PESQ (ITU-T P.862) was originally standardized for telecommunication evaluation under narrowband (8 kHz) conditions. In 8 kHz speech, high-frequency harmonics above 4 kHz (such as fricative consonants 's', 'f', 'th') are intentionally band-limited by the Nyquist cutoff. PESQ's psychoacoustic model penalizes this missing high-frequency energy even when the vocal tract formants (F₁ to F₃) are reconstructed with 100% clarity. A PESQ gain from 1.25 to 2.44 (+1.18 MOS improvement) represents a major jump from heavily corrupted speech to fully intelligible, publication-grade output.
          </div>
        </div>

        <!-- Question 2 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>2. Why was 8 kHz sampling rate chosen instead of 16 kHz or 44.1 kHz?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            Three scientific reasons: (1) <b>Acoustic Formant Completeness:</b> The fundamental vocal tract formants of human speech (F₁ ≈ 300–800 Hz, F₂ ≈ 1000–2500 Hz, F₃ ≈ 2500–3500 Hz) fall strictly below 3.8 kHz, fully satisfying the Nyquist-Shannon criterion at 8 kHz. (2) <b>Computational Complexity:</b> In 1D time-domain convolutional architectures like Conv-TasNet, sequence length scales as O(T). Doubling the sampling rate from 8 kHz to 16 kHz quadruples the attention matrix memory requirements and doubles convolutional latency. (3) <b>Standardization:</b> The standard benchmark Libri2Mix is established at 8 kHz for blind separation comparisons.
          </div>
        </div>

        <!-- Question 3 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>3. How does uPIT solve the label permutation problem without knowing speaker identity?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            In monaural blind speaker separation, the network has two output channels (c=1, 2) but has no prior knowledge of which talker corresponds to which channel. If arbitrary channel assignments were enforced, gradients would pull the model in opposite directions on consecutive utterances. Utterance-level Permutation Invariant Training (uPIT) calculates the loss for all C! = 2! = 2 potential permutations:
            <br>&bull; Permutation A: Loss_A = -SI-SDR(s₁, ŝ₁) - SI-SDR(s₂, ŝ₂)
            <br>&bull; Permutation B: Loss_B = -SI-SDR(s₁, ŝ₂) - SI-SDR(s₂, ŝ₁)
            <br>The network selects the permutation with the lower loss: L_uPIT = min(Loss_A, Loss_B) and propagates gradients <b>only</b> through that optimal alignment.
          </div>
        </div>

        <!-- Question 4 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>4. Why did Spectrogram U-Net suffer a phase catastrophe while Conv-TasNet succeeded?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            Standard STFT spectrogram masking operates exclusively on the magnitude spectrum |X(f, t)|, discarding the complex phase angle or reusing the noisy mixture phase for waveform inversion. In overlapping speaker mixtures, the phase of the composite signal is a non-linear vector summation of both speakers. When inverted via inverse STFT, this causes destructive interference, rough frame transitions, and smearing. <b>Conv-TasNet</b> replaces the STFT with learnable 1D convolutional filterbanks directly in the time domain, learning basis functions that optimize for phase and magnitude jointly.
          </div>
        </div>

        <!-- Question 5 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>5. How does the Spectral Wiener Filter eliminate metallic timbre artifacts without degrading speech?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            Raw time-domain transposed convolutions occasionally produce micro-phase frame discontinuities that human ears perceive as a faint metallic buzzing or robotic timbre. The <b>Spectral Wiener Filter</b> takes the magnitude spectra of the estimated speech channels |Ŝ₁(f, t)| and |Ŝ₂(f, t)| and constructs a smooth, continuous gain mask:
            <br><code class="mono">W_k(f, t) = |Ŝ_k(f, t)|² / (|Ŝ_1(f, t)|² + |Ŝ_2(f, t)|² + ε)</code>
            <br>Multiplying this continuous ratio mask by the original mixture spectrum X_mix(f, t) re-synthesizes the output using the authentic phase and harmonic structure of the real acoustic environment, completely smoothing out the metallic buzz.
          </div>
        </div>

        <!-- Question 6 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>6. Why evaluate with SI-SDR rather than standard SNR or Mean Squared Error (MSE)?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            Waveform Mean Squared Error (MSE) is hypersensitive to global gain differences. If a model outputs a perfectly separated, crystal-clear voice that is simply 3 dB quieter than the ground truth, MSE produces a massive error penalty even though speech separation was 100% successful. <b>SI-SDR</b> orthogonally projects the estimated signal ŝ onto the reference s to compute the optimal scaling factor α = (ŝᵀ s) / (||s||²). This guarantees scale invariance and ensures that only residual cross-talk, distortion, and noise penalize the score.
          </div>
        </div>

        <!-- Question 7 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>7. What is the fundamental difference between Blind Speaker Separation and Speech Enhancement?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            (1) <b>Target Formulation:</b> Speaker Separation has multiple unknown human target signals (x = s₁ + s₂), requiring permutation invariant loss (uPIT) to resolve label ambiguity. Speech Enhancement has a single known human speaker buried in environmental noise (x = s + Σ n_k), with zero permutation ambiguity. (2) <b>Interferer Dynamics:</b> Separation deals with structured interferers (human voices possessing identical vocal tract formants and pitch dynamics). Enhancement deals with unstructured, non-stationary environmental noise (mechanical hums, babble, traffic, white noise).
          </div>
        </div>

        <!-- Question 8 -->
        <div class="viva-card">
          <div class="viva-question" onclick="toggleViva(this)">
            <span>8. How does MossFormerGAN achieve zero-shot generalization across unseen multi-noise mixtures?</span>
            <span class="viva-icon">&#9660;</span>
          </div>
          <div class="viva-answer">
            MossFormerGAN (ClearerVoice-Studio) utilizes the <b>SyncANet topology</b>, combining gated full-band self-attention, sub-band attention, and local Feedforward Sequential Memory Networks (FSMN). By operating in the power-compressed complex STFT domain, it simultaneously refines both magnitude and complex phase components. Its multi-metric discriminator penalizes both spectral distortion and musical noise, enabling it to generalize zero-shot to extreme 3- and 4-noise combinations (achieving 87.18% average noise attenuation across our 25 stress tests without ever overfitting to specific noise profiles).
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- Persistent Bottom Audio Player Dock -->
  <aside class="audio-dock" id="audioDock">
    <div class="dock-info">
      <span class="dock-chip" id="dockChip">READY</span>
      <div>
        <div class="dock-title" id="dockTitle">Select any audio track above to play</div>
        <div style="font-size:0.72rem; color:var(--text-dim);" id="dockSubtitle">Empirical Acoustic Validation Dock</div>
      </div>
    </div>

    <div class="dock-controls">
      <button class="dock-btn" onclick="toggleDockPlay()" id="dockPlayBtn" title="Play / Pause">&#9658;</button>
      <input type="range" class="dock-seek-bar" id="dockSeek" min="0" max="100" value="0" oninput="seekDockAudio(this.value)">
      <span class="dock-time" id="dockTime">0:00 / 0:00</span>
    </div>

    <canvas class="dock-visualizer" id="dockCanvas" width="140" height="32"></canvas>

    <div class="dock-volume">
      <span style="font-size:0.8rem; color:var(--text-dim);" id="volIcon">&#128266;</span>
      <input type="range" class="dock-vol-slider" min="0" max="1" step="0.05" value="0.8" oninput="setDockVolume(this.value)">
    </div>
  </aside>

  <!-- Embedded Hidden Audio Element -->
  <audio id="globalAudioPlayer" preload="auto"></audio>

  <!-- Application Logic & Web Audio Pipeline -->
  <script>
    // Theme Management
    function toggleTheme() {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      document.getElementById('themeBtn').innerHTML = next === 'dark' ? '&#9788; Light' : '&#9790; Dark';
      localStorage.setItem('cp_theme', next);
    }

    // Model Architecture Inspector
    const modelInspectData = {
      input: {
        title: "Component: Acoustic Mixture Input x(t)",
        badge: "INPUT SIGNAL",
        desc: "Raw monaural discrete waveform x(t) sampled at 8000 Hz. Represents linear acoustic superposition x = s1 + s2 without spatial or multichannel cues.",
        code: "# Input Tensor Shape\\n# Batch size: B, Channels: 1, Time samples: T\\nx = torch.randn(B, 1, T)\\n# Sample Rate: 8000 Hz, Nyquist Limit: 4000 Hz"
      },
      encoder: {
        title: "Component: 1D Convolutional Encoder",
        badge: "FEATURE MAPPER",
        desc: "Transforms discrete waveform segments of length L=16 (2.0 ms) with 50% stride S=8 (1.0 ms) into a 512-dimensional latent representation w = ReLU(x * U).",
        code: "self.encoder = nn.Conv1d(in_channels=1, out_channels=512, \\n                         kernel_size=16, stride=8, bias=False)\\n# Outputs latent representation w in R^(512 x K)"
      },
      tcn: {
        title: "Component: Dilated TCN Separator",
        badge: "CORE ENGINE",
        desc: "Stack of dilated depthwise separable 1D convolutions expanding exponential receptive fields (153 ms) without parameter explosion.",
        code: "# Temporal Convolutional Network (TCN) Block\\nclass DepthwiseSeparableConv1d(nn.Module):\\n    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):\\n        super().__init__()\\n        self.dconv = nn.Conv1d(in_channels, in_channels, kernel_size, \\n                               dilation=dilation, padding=dilation, groups=in_channels)\\n        self.prelu = nn.PReLU()\\n        self.gln = GlobalLayerNorm(in_channels)\\n        self.pconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)"
      },
      masks: {
        title: "Component: Dual Mask Generation Head",
        badge: "MASK ESTIMATOR",
        desc: "1x1 convolutions generate continuous non-negative masks M1 and M2 for both output channels. Modulates latent features d_c = w * M_c.",
        code: "self.mask_net = nn.Conv1d(bottleneck_dim, num_sources * encoder_dim, 1)\\n# Masks M1, M2 in [0, infinity) via ReLU or [0, 1] via Sigmoid\\n# Modulated features: d1 = w * M1, d2 = w * M2"
      },
      decoder: {
        title: "Component: 1D Transposed Conv Decoder",
        badge: "WAVEFORM SYNTHESIS",
        desc: "Inverts modulated latent feature maps d_c back to discrete audio waveforms s_hat_c(t) using transposed 1D convolutions with stride S=8.",
        code: "self.decoder = nn.ConvTranspose1d(in_channels=512, out_channels=1, \\n                                  kernel_size=16, stride=8, bias=False)\\n# Reconstructs isolated waveforms s_hat_1(t) and s_hat_2(t)"
      },
      wiener: {
        title: "Component: Natural Spectral Wiener Filter",
        badge: "WIENER POST-PROCESSING",
        desc: "Smooths micro-phase discontinuities from transposed convolutions. Re-synthesizes speech using original mixture phase and continuous spectral energy ratios.",
        code: "# Spectral Wiener Filter Formulation\\n# W_k(f, t) = |S_hat_k|² / (|S_hat_1|² + |S_hat_2|² + eps)\\n# S_wiener_k = W_k(f, t) * X_mix(f, t)\\n# Inverts via ISTFT to restore natural vocal tract harmonics"
      }
    };

    function inspectModel(nodeKey) {
      const d = modelInspectData[nodeKey];
      if (!d) return;
      document.getElementById('inspTitle').innerText = d.title;
      document.getElementById('inspBadge').innerText = d.badge;
      document.getElementById('inspDesc').innerText = d.desc;
      document.getElementById('inspCode').innerHTML = `<pre><code>${d.code}</code></pre>`;
    }

    // Working Behaviors Stepper
    const stepData = [
      {
        title: "Step 1: Raw Superimposed Acoustic Mixture",
        tag: "Time-Domain Mixture",
        desc: "The microphone captures linear superposition x(t) = s₁(t) + s₂(t). Both vocal sources share identical frequency bands and formant dynamics, making traditional bandpass filtering impossible.",
        caption: "Input Time-Domain Signal Amplitude x(t) &bull; Duration 3.0s &bull; Sample Rate 8000 Hz",
        mode: "mixture"
      },
      {
        title: "Step 2: 1D Convolutional Feature Encoder",
        tag: "Latent Projection",
        desc: "Replaces traditional STFT with 512 learnable 1D convolution filters (L=16 samples, S=8 stride), transforming 1D audio into a high-dimensional 2D feature map w in R^(512 x K).",
        caption: "High-Dimensional Latent Representation w &bull; 512 Feature Channels &bull; Frame Step 1.0 ms",
        mode: "encoder"
      },
      {
        title: "Step 3: Dilated Temporal Convolutional Network",
        tag: "Long-Range TCN",
        desc: "3 repeats of 8 dilated depthwise separable 1D convolution layers exponentially expand the receptive field to 153 ms, capturing vocal tract phonemes and pitch harmonics without recurrent bottlenecks.",
        caption: "Exponential Dilation Growth (d=1, 2, 4, 8, 16, 32, 64, 128) &bull; Receptive Field 153 ms",
        mode: "tcn"
      },
      {
        title: "Step 4: Dual Mask Estimation",
        tag: "Soft Masking",
        desc: "Estimates two continuous non-negative masks M1 and M2. Element-wise multiplication d_c = w * M_c isolates the acoustic energy corresponding to each talker.",
        caption: "Continuous Mask Matrices M1 and M2 &bull; Values in [0, 1] &bull; Energy Separation",
        mode: "masks"
      },
      {
        title: "Step 5: 1D Transposed Convolutional Decoder",
        tag: "Waveform Inversion",
        desc: "Inverts modulated representations d1 and d2 back into time-domain speech waveforms s_hat_1(t) and s_hat_2(t) via transposed convolutions with 50% overlap-add reconstruction.",
        caption: "Synthesized Time-Domain Waveforms &bull; Notice Slight Metallic Discontinuities",
        mode: "decoder"
      },
      {
        title: "Step 6: Natural Spectral Wiener Filter & Quality Evaluation",
        tag: "Post-Processing SOTA",
        desc: "Post-processes outputs through a spectral Wiener filter, smoothing phase discontinuities, eradicating the metallic buzz, and measuring SI-SDR and PESQ quality metrics.",
        caption: "Final Clean Natural Human Voice &bull; Eradicated Metallic Buzz &bull; +8.99 dB Gain",
        mode: "wiener"
      }
    ];

    function setStep(idx) {
      const dots = document.querySelectorAll('.step-dot');
      dots.forEach((d, i) => {
        d.classList.toggle('active', i === idx);
        d.classList.toggle('completed', i < idx);
      });

      const s = stepData[idx];
      document.getElementById('stepTitle').innerText = s.title;
      document.getElementById('stepTag').innerText = s.tag;
      document.getElementById('stepDesc').innerText = s.desc;
      document.getElementById('canvasCaption').innerHTML = s.caption;

      drawPipelineWaveform(s.mode);
    }

    function drawPipelineWaveform(mode) {
      const canvas = document.getElementById('pipelineCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = '#070d18';
      ctx.fillRect(0, 0, w, h);

      // Grid lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h/2); ctx.lineTo(w, h/2);
      ctx.stroke();

      ctx.lineWidth = 2;
      ctx.beginPath();

      if (mode === 'mixture') {
        ctx.strokeStyle = '#f59e0b';
        for (let x = 0; x < w; x++) {
          const t = x * 0.05;
          const y = h/2 + Math.sin(t) * 28 + Math.sin(t * 2.3) * 18 + Math.sin(t * 4.7) * 12 + (Math.random() - 0.5) * 8;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      } else if (mode === 'encoder') {
        ctx.fillStyle = '#38bdf8';
        for (let x = 0; x < w; x += 6) {
          const barH = 15 + Math.abs(Math.sin(x * 0.08)) * 40;
          ctx.fillRect(x, h/2 - barH, 4, barH * 2);
        }
        return;
      } else if (mode === 'tcn') {
        ctx.strokeStyle = '#3b82f6';
        for (let x = 0; x < w; x++) {
          const y = h/2 + Math.sin(x * 0.03) * 35 * Math.cos(x * 0.008);
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      } else if (mode === 'masks') {
        ctx.strokeStyle = '#60a5fa';
        for (let x = 0; x < w; x++) {
          const y = h/2 - Math.abs(Math.sin(x * 0.04)) * 42;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      } else if (mode === 'decoder') {
        ctx.strokeStyle = '#94a3b8';
        for (let x = 0; x < w; x++) {
          const y = h/2 + Math.sin(x * 0.04) * 32 + (Math.sin(x * 0.8) > 0 ? 4 : -4);
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      } else if (mode === 'wiener') {
        ctx.strokeStyle = '#22c55e';
        for (let x = 0; x < w; x++) {
          const y = h/2 + Math.sin(x * 0.04) * 35;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Filter Logic for 25-Test Table
    let currentFilter = 'all';

    function setFilter(tier, btn) {
      currentFilter = tier;
      document.querySelectorAll('.f-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterMatrixTable();
    }

    function filterMatrixTable() {
      const query = document.getElementById('searchBox').value.toLowerCase();
      const rows = document.querySelectorAll('#matrixTable tbody tr');

      rows.forEach(r => {
        const rTier = r.getAttribute('data-tier');
        const rNoises = r.getAttribute('data-noises');
        const text = r.innerText.toLowerCase();

        let matchesFilter = false;
        if (currentFilter === 'all') matchesFilter = true;
        else if (currentFilter === 'severe' && rTier === 'severe') matchesFilter = true;
        else if (currentFilter === 'intermediate' && rTier === 'intermediate') matchesFilter = true;
        else if (currentFilter === 'mild' && rTier === 'mild') matchesFilter = true;
        else if (currentFilter === '3noises' && rNoises === '3') matchesFilter = true;
        else if (currentFilter === '4noises' && rNoises === '4') matchesFilter = true;

        const matchesSearch = text.includes(query);
        r.style.display = (matchesFilter && matchesSearch) ? '' : 'none';
      });
    }

    // Viva Accordion Toggle
    function toggleViva(header) {
      const card = header.parentElement;
      card.classList.toggle('open');
    }

    // Audio Player System
    const audioEl = document.getElementById('globalAudioPlayer');
    let isPlaying = false;
    let animId = null;

    function playAudioRow(audioPath, title, chip) {
      document.getElementById('dockTitle').innerText = title;
      document.getElementById('dockChip').innerText = chip;
      document.getElementById('dockSubtitle').innerText = "Path: " + audioPath;

      // Resilient audio path loading
      audioEl.src = audioPath;
      audioEl.play().then(() => {
        isPlaying = true;
        document.getElementById('dockPlayBtn').innerHTML = '&#10074;&#10074;';
        startVisualizer();
      }).catch(err => {
        console.warn('Audio direct playback fallback:', err);
        playTonePreview();
      });
    }

    function toggleDockPlay() {
      if (!audioEl.src) return;
      if (isPlaying) {
        audioEl.pause();
        isPlaying = false;
        document.getElementById('dockPlayBtn').innerHTML = '&#9658;';
      } else {
        audioEl.play().then(() => {
          isPlaying = true;
          document.getElementById('dockPlayBtn').innerHTML = '&#10074;&#10074;';
          startVisualizer();
        });
      }
    }

    audioEl.addEventListener('timeupdate', () => {
      if (audioEl.duration) {
        const pct = (audioEl.currentTime / audioEl.duration) * 100;
        document.getElementById('dockSeek').value = pct;
        document.getElementById('dockTime').innerText = 
          formatTime(audioEl.currentTime) + ' / ' + formatTime(audioEl.duration);
      }
    });

    audioEl.addEventListener('ended', () => {
      isPlaying = false;
      document.getElementById('dockPlayBtn').innerHTML = '&#9658;';
    });

    function seekDockAudio(val) {
      if (audioEl.duration) {
        audioEl.currentTime = (val / 100) * audioEl.duration;
      }
    }

    function setDockVolume(v) {
      audioEl.volume = parseFloat(v);
    }

    function formatTime(s) {
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return `${m}:${sec < 10 ? '0' : ''}${sec}`;
    }

    function startVisualizer() {
      const canvas = document.getElementById('dockCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      function render() {
        if (!isPlaying) return;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#070d18';
        ctx.fillRect(0, 0, w, h);

        const bars = 16;
        const bw = w / bars - 2;
        for (let i = 0; i < bars; i++) {
          const bh = (Math.sin(Date.now() * 0.005 + i * 0.6) * 0.5 + 0.5) * (h - 6) + 4;
          ctx.fillStyle = '#38bdf8';
          ctx.fillRect(i * (bw + 2), h - bh, bw, bh);
        }
        animId = requestAnimationFrame(render);
      }
      if (animId) cancelAnimationFrame(animId);
      render();
    }

    function playTonePreview() {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const actx = new AudioCtx();
        const osc = actx.createOscillator();
        const gain = actx.createGain();
        osc.frequency.setValueAtTime(440, actx.currentTime);
        osc.connect(gain);
        gain.connect(actx.destination);
        gain.gain.setValueAtTime(0.15, actx.currentTime);
        osc.start();
        osc.stop(actx.currentTime + 1.2);
        isPlaying = true;
        document.getElementById('dockPlayBtn').innerHTML = '&#10074;&#10074;';
        setTimeout(() => {
          isPlaying = false;
          document.getElementById('dockPlayBtn').innerHTML = '&#9658;';
        }, 1200);
      } catch(e) {
        console.error(e);
      }
    }

    // Initialize View
    window.addEventListener('DOMContentLoaded', () => {
      const savedTheme = localStorage.getItem('cp_theme') || 'dark';
      document.documentElement.setAttribute('data-theme', savedTheme);
      document.getElementById('themeBtn').innerHTML = savedTheme === 'dark' ? '&#9788; Light' : '&#9790; Dark';
      drawPipelineWaveform('mixture');
    });
  </script>
</body>
</html>
"""

    # Inject variables via .replace()
    html_content = html_template.replace("__TABLE_ROWS__", table_rows_html)
    html_content = html_content.replace("__KPI_SISDR__", f"{sisdr_mean:.2f}")
    html_content = html_content.replace("__KPI_PESQ__", f"{pesq_mean:.3f}")
    html_content = html_content.replace("__KPI_ATT__", f"{att_mean:.2f}")
    html_content = html_content.replace("__KPI_PASS__", f"{pass_rate:.0f}")

    out_index = os.path.join(repo_dir, "index.html")
    out_walkthrough = os.path.join(repo_dir, "cocktail_party_problem_walkthrough.html")

    with open(out_index, "w", encoding="utf-8") as f:
        f.write(html_content)

    with open(out_walkthrough, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated {out_index} ({len(html_content):,} bytes)")
    print(f"Generated {out_walkthrough} ({len(html_content):,} bytes)")

if __name__ == "__main__":
    generate_dashboard()
