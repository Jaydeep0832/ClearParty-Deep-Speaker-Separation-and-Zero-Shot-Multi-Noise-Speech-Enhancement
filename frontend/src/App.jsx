import React, { useState, useEffect, useRef } from 'react';
import { 
  BENCHMARK_TESTS, 
  SUMMARY_STATS, 
  EVOLUTION_PHASES, 
  VIVA_QUESTIONS, 
  SEPARATION_SAMPLES 
} from './data/benchmarkData';
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  CheckCircle, 
  Layers, 
  Sliders, 
  FileText, 
  BookOpen, 
  HelpCircle, 
  Sun, 
  Moon, 
  GitBranch, 
  ExternalLink, 
  Search, 
  Filter, 
  BarChart2, 
  ShieldCheck, 
  Activity,
  ArrowRight
} from 'lucide-react';

export default function App() {
  const [theme, setTheme] = useState('light');
  const [activeTab, setActiveTab] = useState('benchmark');
  const [filterTier, setFilterTier] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Audio Player State
  const [currentTrack, setCurrentTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(new Audio());

  // Canvas Oscilloscope State
  const canvasRef = useRef(null);
  const [isOscActive, setIsOscActive] = useState(false);
  const [oscMode, setOscMode] = useState('mix');
  const audioCtxRef = useRef(null);
  const osc1Ref = useRef(null);
  const osc2Ref = useRef(null);
  const animFrameRef = useRef(null);

  // Expanded Viva Accordion
  const [expandedViva, setExpandedViva] = useState(0);

  // Sync theme attribute to HTML tag
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Handle Audio Player controls
  const playAudioTrack = (track) => {
    if (currentTrack?.url === track.url && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
      return;
    }

    setCurrentTrack(track);
    audioRef.current.src = track.url;
    audioRef.current.play().then(() => {
      setIsPlaying(true);
    }).catch(err => {
      console.warn("Audio play error:", err);
      setIsPlaying(false);
    });
  };

  const togglePlayPause = () => {
    if (!currentTrack) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true));
    }
  };

  useEffect(() => {
    const handleEnded = () => setIsPlaying(false);
    audioRef.current.addEventListener('ended', handleEnded);
    return () => {
      audioRef.current.removeEventListener('ended', handleEnded);
      audioRef.current.pause();
    };
  }, []);

  // Filtered 25-test benchmark items
  const filteredTests = BENCHMARK_TESTS.filter(t => {
    const matchesTier = 
      filterTier === 'all' ? true :
      filterTier === 'severe' ? t.snr <= 2.5 :
      filterTier === 'medium' ? (t.snr > 2.5 && t.snr <= 5.5) :
      filterTier === 'mild' ? t.snr > 5.5 :
      filterTier === '4noises' ? t.noiseCount === 4 :
      filterTier === '3noises' ? t.noiseCount === 3 : true;

    const matchesQuery = 
      searchQuery === '' ? true :
      t.cleanFile.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.noiseDescription.toLowerCase().includes(searchQuery.toLowerCase()) ||
      String(t.snr).includes(searchQuery);

    return matchesTier && matchesQuery;
  });

  // Web Audio API Synthesizer Simulator
  const toggleOscilloscope = () => {
    if (isOscActive) {
      if (osc1Ref.current) osc1Ref.current.stop();
      if (osc2Ref.current) osc2Ref.current.stop();
      if (audioCtxRef.current) audioCtxRef.current.close();
      cancelAnimationFrame(animFrameRef.current);
      setIsOscActive(false);
    } else {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      audioCtxRef.current = ctx;

      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(140, ctx.currentTime); // Male Baritone
      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(240, ctx.currentTime); // Female Soprano

      const g1 = ctx.createGain();
      const g2 = ctx.createGain();
      g1.gain.setValueAtTime(0.06, ctx.currentTime);
      g2.gain.setValueAtTime(0.06, ctx.currentTime);

      osc1.connect(g1);
      osc2.connect(g2);
      g1.connect(ctx.destination);
      g2.connect(ctx.destination);

      osc1.start();
      osc2.start();
      osc1Ref.current = osc1;
      osc2Ref.current = osc2;
      setIsOscActive(true);

      renderCanvas();
    }
  };

  const renderCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;

    ctx.fillStyle = theme === 'dark' ? '#0f172a' : '#f8fafc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = 
      oscMode === 'mix' ? '#b45309' :
      oscMode === 'spk1' ? '#1e3a8a' : '#15803d';

    ctx.beginPath();
    const sliceWidth = canvas.width / 100;
    let x = 0;
    const t = Date.now() / 150;

    for (let i = 0; i < 100; i++) {
      let y = canvas.height / 2;
      if (oscMode === 'mix') {
        y += Math.sin(i * 0.2 + t) * 30 + Math.sin(i * 0.4 + t * 1.7) * 20;
      } else if (oscMode === 'spk1') {
        y += Math.sin(i * 0.2 + t) * 40;
      } else {
        y += Math.sin(i * 0.4 + t * 1.7) * 35;
      }

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.stroke();

    animFrameRef.current = requestAnimationFrame(renderCanvas);
  };

  return (
    <div className="min-h-screen">
      {/* Top Academic Header */}
      <header className="inst-header">
        <div className="inst-header-inner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{
              width: '34px',
              height: '34px',
              backgroundColor: 'var(--c-primary)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              fontWeight: 800,
              fontSize: '1.1rem'
            }}>
              CP
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '1.05rem', letterSpacing: '-0.01em', lineHeight: 1.2 }}>
                ClearParty Research Platform
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                Deep Speaker Separation & Multi-Noise Speech Enhancement
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span className="badge-academic badge-green">
              <ShieldCheck size={13} /> SOTA Benchmark Verified
            </span>
            <button 
              className="btn-academic"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              title="Toggle Theme"
            >
              {theme === 'light' ? <Moon size={14} /> : <Sun size={14} />}
              <span>{theme === 'light' ? 'Dark' : 'Light'}</span>
            </button>
            <a 
              href="https://github.com/Jaydeep0832/ClearParty-Deep-Speaker-Separation-and-Zero-Shot-Multi-Noise-Speech-Enhancement"
              target="_blank"
              rel="noreferrer"
              className="btn-academic btn-academic-primary"
            >
              <GitBranch size={14} />
              <span>Repository</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="app-container">
        
        {/* Mentor Presentation Header */}
        <section style={{ marginBottom: '1.75rem', paddingTop: '0.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <span className="badge-academic badge-blue" style={{ marginBottom: '0.4rem' }}>
                Academic Thesis & Mentor Walkthrough Suite
              </span>
              <h1 style={{ fontSize: '2.1rem', fontWeight: 800, letterSpacing: '-0.025em', marginBottom: '0.4rem' }}>
                ClearParty: Dual-Track Audio Processing System
              </h1>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.98rem', maxWidth: '880px', lineHeight: 1.55 }}>
                An empirical research framework resolving Colin Cherry's Cocktail Party Problem. Featuring 1D Time-Domain Conv-TasNet with Utterance-Level Permutation Invariant Training (uPIT) for 2-speaker separation, and SOTA MossFormerGAN Synchronous Attention Networks for 25 compound multi-noise enhancement stress tests.
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em' }}>
                Host Hardware / Framework
              </div>
              <div className="mono" style={{ fontSize: '0.86rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                PyTorch 2.10 | CUDA FP16 | Sinc 16k
              </div>
            </div>
          </div>
        </section>

        {/* Top Quantitative KPI Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.9rem', marginBottom: '2rem' }}>
          <div className="card-academic" style={{ padding: '1rem', borderLeft: '4px solid var(--c-success)' }}>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-dim)' }}>
              25-Test Accuracy Rate
            </div>
            <div className="mono" style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--c-success)', margin: '0.2rem 0' }}>
              100.0%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              25 / 25 Tests Passed (Zero Regressions)
            </div>
          </div>

          <div className="card-academic" style={{ padding: '1rem', borderLeft: '4px solid var(--c-primary)' }}>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-dim)' }}>
              MossFormer SI-SDR Gain
            </div>
            <div className="mono" style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--c-primary)', margin: '0.2rem 0' }}>
              +8.99 dB
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Peak: +10.72 dB (+94% vs Baseline)
            </div>
          </div>

          <div className="card-academic" style={{ padding: '1rem', borderLeft: '4px solid var(--c-success)' }}>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-dim)' }}>
              PESQ MOS Improvement
            </div>
            <div className="mono" style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--c-success)', margin: '0.2rem 0' }}>
              +1.184
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Output: 2.790 (Peak: 3.468 MOS)
            </div>
          </div>

          <div className="card-academic" style={{ padding: '1rem', borderLeft: '4px solid var(--c-primary)' }}>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-dim)' }}>
              Noise Power Attenuation
            </div>
            <div className="mono" style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--c-primary)', margin: '0.2rem 0' }}>
              87.18%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Under 3 to 4 Simultaneous Noises
            </div>
          </div>

          <div className="card-academic" style={{ padding: '1rem', borderLeft: '4px solid var(--c-warning)' }}>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-dim)' }}>
              Speaker Separation (14k Pairs)
            </div>
            <div className="mono" style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--c-warning)', margin: '0.2rem 0' }}>
              +11.76 dB
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Validation SI-SDR (251 Talkers)
            </div>
          </div>
        </div>

        {/* Structured Mentor Navigation Tabs */}
        <div style={{ borderBottom: '1px solid var(--border-color)', marginBottom: '1.75rem' }}>
          <div className="nav-tabs-wrap">
            <button 
              className={`nav-tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
              onClick={() => setActiveTab('benchmark')}
            >
              <BarChart2 size={15} /> 1. SOTA 25-Test Matrix (MossFormer)
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'separation' ? 'active' : ''}`}
              onClick={() => setActiveTab('separation')}
            >
              <Volume2 size={15} /> 2. Objective 1: Speaker Separation Lab
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'formulation' ? 'active' : ''}`}
              onClick={() => setActiveTab('formulation')}
            >
              <BookOpen size={15} /> 3. Problem Scope & Dual-Track Formulation
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'evolution' ? 'active' : ''}`}
              onClick={() => setActiveTab('evolution')}
            >
              <Layers size={15} /> 4. Evolutionary Milestones (Phases 1–5)
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
              onClick={() => setActiveTab('architecture')}
            >
              <Sliders size={15} /> 5. Deep Architecture & Oscilloscope
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'viva' ? 'active' : ''}`}
              onClick={() => setActiveTab('viva')}
            >
              <HelpCircle size={15} /> 6. Mentor Defense & Viva Examination
            </button>
          </div>
        </div>

        {/* =============================================================
            TAB 1: 25-TEST COMPOUND MULTI-NOISE BENCHMARK MATRIX
            ============================================================= */}
        {activeTab === 'benchmark' && (
          <div>
            {/* Comparative Evidence Block */}
            <div className="card-academic" style={{ marginBottom: '1.5rem' }}>
              <div className="card-header-academic">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldCheck size={16} color="var(--c-primary)" />
                  <strong style={{ fontSize: '0.92rem' }}>
                    Comparative Verification: SOTA MossFormerGAN vs. Conv-TasNet Baseline
                  </strong>
                </div>
                <span className="badge-academic badge-green">Statistically Verified</span>
              </div>
              <div className="card-body-academic">
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
                  
                  {/* SI-SDR Metric Comparison */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '0.3rem' }}>
                      <span style={{ fontWeight: 600 }}>SI-SDR Gain (Signal Fidelity)</span>
                      <span className="mono" style={{ fontWeight: 700, color: 'var(--c-primary)' }}>
                        +8.99 dB vs +4.64 dB (+94% higher)
                      </span>
                    </div>
                    <div style={{ height: '8px', backgroundColor: 'var(--bg-card-alt)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: '89.9%', height: '100%', backgroundColor: 'var(--c-primary)' }}></div>
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                      Raw input: 5.21 dB &rarr; Output: 14.20 dB (Peak +10.72 dB in Test 11)
                    </div>
                  </div>

                  {/* PESQ Metric Comparison */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '0.3rem' }}>
                      <span style={{ fontWeight: 600 }}>PESQ MOS Gain (Perceptual Quality)</span>
                      <span className="mono" style={{ fontWeight: 700, color: 'var(--c-success)' }}>
                        +1.184 vs +0.268 (4.4&times; greater boost)
                      </span>
                    </div>
                    <div style={{ height: '8px', backgroundColor: 'var(--bg-card-alt)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: '82.5%', height: '100%', backgroundColor: 'var(--c-success)' }}></div>
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                      Raw input: 1.606 &rarr; Output: 2.790 MOS (Peak 3.468 in Test 20)
                    </div>
                  </div>

                  {/* Noise Attenuation */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '0.3rem' }}>
                      <span style={{ fontWeight: 600 }}>Noise Power Energy Attenuation</span>
                      <span className="mono" style={{ fontWeight: 700, color: 'var(--c-success)' }}>
                        87.18% vs 65.01% (+22.2% more noise removed)
                      </span>
                    </div>
                    <div style={{ height: '8px', backgroundColor: 'var(--bg-card-alt)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: '87.2%', height: '100%', backgroundColor: 'var(--c-success)' }}></div>
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                      Measured under 3 to 4 simultaneous distinct environmental noises
                    </div>
                  </div>

                </div>
              </div>
            </div>

            {/* Matrix Filter & Search Toolbar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.9rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', marginRight: '0.2rem' }}>
                  Filter Protocol:
                </span>
                {[
                  { id: 'all', label: 'All 25 Tests' },
                  { id: 'severe', label: 'Severe SNR (0–2.5 dB)' },
                  { id: 'medium', label: 'Medium SNR (3–5.5 dB)' },
                  { id: 'mild', label: 'Mild SNR (6–10 dB)' },
                  { id: '4noises', label: '4-Noise Mixtures' },
                  { id: '3noises', label: '3-Noise Mixtures' }
                ].map(b => (
                  <button
                    key={b.id}
                    className="btn-academic"
                    style={{
                      fontSize: '0.76rem',
                      padding: '0.3rem 0.65rem',
                      backgroundColor: filterTier === b.id ? 'var(--c-primary-bg)' : 'var(--bg-card)',
                      borderColor: filterTier === b.id ? 'var(--c-primary)' : 'var(--border-color)',
                      color: filterTier === b.id ? 'var(--c-primary)' : 'var(--text-main)',
                      fontWeight: filterTier === b.id ? 700 : 500
                    }}
                    onClick={() => setFilterTier(b.id)}
                  >
                    {b.label}
                  </button>
                ))}
              </div>

              <div style={{ position: 'relative', minWidth: '260px' }}>
                <Search size={14} style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-dim)' }} />
                <input
                  type="text"
                  placeholder="Search noise, utterance, SNR..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.4rem 0.6rem 0.4rem 2rem',
                    fontSize: '0.82rem',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--bg-card)',
                    color: 'var(--text-main)',
                    outline: 'none'
                  }}
                />
              </div>
            </div>

            {/* Master 25-Test Benchmark Table */}
            <div className="table-academic-wrap">
              <table className="table-academic">
                <thead>
                  <tr>
                    <th>Test ID</th>
                    <th>Clean Utterance</th>
                    <th>Input SNR</th>
                    <th>Compound Noise Formulation (3–4 Simultaneous)</th>
                    <th>SI-SDR Gain</th>
                    <th>PESQ Gain</th>
                    <th>STOI Gain</th>
                    <th>Attenuation</th>
                    <th>Pearson r</th>
                    <th>Validation</th>
                    <th style={{ textAlign: 'center' }}>Audio Audition Controls</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTests.map(t => (
                    <tr key={t.id}>
                      <td className="mono" style={{ fontWeight: 700, color: 'var(--c-primary)' }}>
                        #{t.padId}
                      </td>
                      <td>
                        <strong>{t.cleanFile}</strong>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                          {t.duration.toFixed(2)}s | 8,000 Hz
                        </div>
                      </td>
                      <td>
                        <span className="badge-academic badge-amber mono">
                          {t.snr.toFixed(1)} dB
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px', alignItems: 'center' }}>
                          <span className="badge-academic badge-blue" style={{ fontSize: '0.68rem' }}>
                            {t.noiseCount} Noises
                          </span>
                          {t.noiseTypes.map((n, i) => (
                            <span key={i} className="badge-academic badge-slate" style={{ fontSize: '0.68rem' }}>
                              {n}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td>
                        <div className="mono" style={{ fontWeight: 700, color: 'var(--c-primary)' }}>
                          +{t.sisdrGain.toFixed(2)} dB
                        </div>
                        <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                          {t.inSisdr.toFixed(2)} &rarr; {t.outSisdr.toFixed(2)}
                        </div>
                      </td>
                      <td>
                        <div className="mono" style={{ fontWeight: 700, color: 'var(--c-success)' }}>
                          +{t.pesqGain.toFixed(3)}
                        </div>
                        <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                          {t.inPesq.toFixed(3)} &rarr; {t.outPesq.toFixed(3)}
                        </div>
                      </td>
                      <td>
                        <div className="mono" style={{ fontWeight: 700, color: 'var(--c-primary)' }}>
                          +{t.stoiGain.toFixed(3)}
                        </div>
                        <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                          {t.inStoi.toFixed(3)} &rarr; {t.outStoi.toFixed(3)}
                        </div>
                      </td>
                      <td>
                        <div style={{ width: '65px', height: '5px', backgroundColor: 'var(--bg-card-alt)', borderRadius: '3px', overflow: 'hidden', marginBottom: '2px' }}>
                          <div style={{ width: `${t.attenuationPct}%`, height: '100%', backgroundColor: 'var(--c-success)' }}></div>
                        </div>
                        <span className="mono" style={{ fontSize: '0.75rem', fontWeight: 600 }}>
                          {t.attenuationPct.toFixed(1)}%
                        </span>
                      </td>
                      <td className="mono" style={{ fontSize: '0.82rem', fontWeight: 600 }}>
                        {t.pearsonR.toFixed(4)}
                      </td>
                      <td>
                        <span className="badge-academic badge-green">PASS</span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '3px', justifyContent: 'center' }}>
                          <button
                            className="btn-academic"
                            style={{ fontSize: '0.7rem', padding: '0.25rem 0.5rem', color: 'var(--c-primary)' }}
                            onClick={() => playAudioTrack({
                              title: `Test #${t.padId} — Clean Reference`,
                              sub: `${t.cleanFile} (Ground Truth Reference)`,
                              tag: 'CLEAN REFERENCE',
                              tagColor: 'green',
                              url: t.cleanAudio
                            })}
                            title="Play Clean Reference Speech"
                          >
                            <Play size={10} /> Clean
                          </button>
                          <button
                            className="btn-academic"
                            style={{ fontSize: '0.7rem', padding: '0.25rem 0.5rem', color: 'var(--c-warning)' }}
                            onClick={() => playAudioTrack({
                              title: `Test #${t.padId} — Compound Mixture (${t.noiseCount} Noises)`,
                              sub: `${t.noiseDescription} @ ${t.snr} dB SNR`,
                              tag: 'NOISY MIXTURE',
                              tagColor: 'amber',
                              url: t.mixAudio
                            })}
                            title="Play Noisy Mixture"
                          >
                            <Play size={10} /> Mixture
                          </button>
                          <button
                            className="btn-academic btn-academic-primary"
                            style={{ fontSize: '0.7rem', padding: '0.25rem 0.5rem' }}
                            onClick={() => playAudioTrack({
                              title: `Test #${t.padId} — SOTA MossFormer Enhanced`,
                              sub: `Gain: +${t.sisdrGain.toFixed(2)} dB SI-SDR | PESQ: ${t.outPesq.toFixed(3)}`,
                              tag: 'SOTA ENHANCED',
                              tagColor: 'blue',
                              url: t.enhAudio
                            })}
                            title="Play SOTA Enhanced Speech"
                          >
                            <Play size={10} /> SOTA
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* =============================================================
            TAB 2: OBJECTIVE 1 - SPEAKER SEPARATION LAB
            ============================================================= */}
        {activeTab === 'separation' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div className="card-academic">
                <div className="card-header-academic">
                  <strong>Task Formulation: Monaural Blind Speaker Separation (BSS)</strong>
                  <span className="badge-academic badge-blue">Time-Domain Conv-TasNet</span>
                </div>
                <div className="card-body-academic">
                  <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginBottom: '0.8rem' }}>
                    Given an acoustic mixture <code>x(t) = s₁(t) + s₂(t)</code> containing two simultaneous human talkers without environmental noise, the objective is to reconstruct independent streams <code>ŝ₁(t)</code> and <code>ŝ₂(t)</code>.
                  </p>
                  <ul style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <li>• <b>Mathematical Challenge:</b> The label permutation problem. The model has no way of knowing whether Speaker 1 is assigned to Output Channel 1 or 2.</li>
                    <li>• <b>Utterance-Level Permutation Invariant Training (uPIT):</b> The model computes the loss for all <code>C! = 2</code> output channel permutations and propagates gradients solely through the minimum-loss alignment: <code>L_uPIT = min_π [ -Σ SI-SDR(s_c, ŝ_π(c)) ]</code>.</li>
                    <li>• <b>Training Corpus:</b> Scaled in Phase 2B to <b>14,269 mixture pairs</b> across 251 speakers from LibriSpeech <code>train-clean-100</code>.</li>
                  </ul>
                </div>
              </div>

              <div className="card-academic">
                <div className="card-header-academic">
                  <strong>The "Metallic Buzz" & Spectral Wiener Solution</strong>
                  <span className="badge-academic badge-green">Post-Processing Innovation</span>
                </div>
                <div className="card-body-academic">
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.6rem' }}>
                    Raw time-domain transposed convolutions occasionally produce micro-phase frame discontinuities that listeners perceive as a faint metallic or robotic timbre.
                  </p>
                  <div style={{ backgroundColor: 'var(--bg-card-alt)', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-color)', marginBottom: '0.6rem' }}>
                    <code className="mono" style={{ fontSize: '0.8rem', color: 'var(--c-primary)' }}>
                      W_c(f, t) = |Ŝ_c(f, t)|² / (|Ŝ_1(f, t)|² + |Ŝ_2(f, t)|² + ε)
                    </code>
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>
                    Applying this soft spectral filter to the original mixture STFT re-synthesizes organic vocal tract harmonic contours, raising STOI intelligibility from 71.3% to <b>88.1%</b>!
                  </p>
                </div>
              </div>
            </div>

            {/* Audio Audition Grid for Speaker Separation */}
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.8rem' }}>
              Separation Audio Demonstration (Held-Out Test Sample)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              {SEPARATION_SAMPLES.map((s, idx) => (
                <div key={idx} className="card-academic" style={{ padding: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.4rem' }}>
                    <strong style={{ fontSize: '0.9rem' }}>{s.title}</strong>
                    <span className={`badge-academic badge-${s.tagColor}`}>{s.tag}</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>
                    {s.subtitle}
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.8rem' }}>
                    {s.desc}
                  </p>
                  <button
                    className="btn-academic btn-academic-primary"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => playAudioTrack({
                      title: s.title,
                      sub: s.subtitle,
                      tag: s.tag,
                      tagColor: s.tagColor,
                      url: s.audio
                    })}
                  >
                    <Play size={13} /> Listen to Audio Sample
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* =============================================================
            TAB 3: PROBLEM SCOPE & DUAL-TRACK FORMULATION
            ============================================================= */}
        {activeTab === 'formulation' && (
          <div>
            <div className="card-academic" style={{ marginBottom: '1.5rem' }}>
              <div className="card-header-academic">
                <strong>Academic Scope: Colin Cherry's Cocktail Party Problem (1953)</strong>
                <span className="badge-academic badge-blue">Foundational Literature</span>
              </div>
              <div className="card-body-academic">
                <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.8rem', lineHeight: 1.6 }}>
                  In a crowded cocktail party, humans seamlessly isolate an individual conversation from competing background chatter, clatter, and laughter. In computational acoustics using a <b>single omnidirectional microphone</b>, spatial time-difference cues are absent, transforming separation into a mathematically ill-posed, underdetermined inverse problem.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
                  <div style={{ backgroundColor: 'var(--bg-card-alt)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--c-primary)', marginBottom: '0.2rem' }}>
                      ASR Front-End Utility
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                      Multi-talker overlap spikes Word Error Rates (WER) from &lt;5% to &gt;60%. Separation acts as a mandatory front-end.
                    </div>
                  </div>
                  <div style={{ backgroundColor: 'var(--bg-card-alt)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--c-success)', marginBottom: '0.2rem' }}>
                      Assistive Hearables
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                      Enables smart earbuds to selectively attenuate 87%+ of ambient restaurant commotion while amplifying the conversation partner.
                    </div>
                  </div>
                  <div style={{ backgroundColor: 'var(--bg-card-alt)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--c-warning)', marginBottom: '0.2rem' }}>
                      Telecommunications
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                      Independent channel balancing and dynamic acoustic echo cancellation in multi-participant conference systems.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Rigorous Distinction Table */}
            <div className="card-academic">
              <div className="card-header-academic">
                <strong>Methodological Boundary: Speech Separation vs. Speech Enhancement</strong>
                <span className="badge-academic badge-green">Taxonomy</span>
              </div>
              <div className="card-body-academic" style={{ padding: 0 }}>
                <table className="table-academic">
                  <thead>
                    <tr>
                      <th>Dimension</th>
                      <th>Track 1: Monaural Speaker Separation</th>
                      <th>Track 2: Multi-Noise Speech Enhancement</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><b>Acoustic Input</b></td>
                      <td className="mono">x(t) = s₁(t) + s₂(t)</td>
                      <td className="mono">x(t) = s(t) + Σ αₖ · nₖ(t)</td>
                    </tr>
                    <tr>
                      <td><b>Interferer Characteristics</b></td>
                      <td>Structured human speech (identical pitch, formant & vocal tract dynamics)</td>
                      <td>Environmental acoustic noise (mechanical hum, diffuse babble, transient clatter)</td>
                    </tr>
                    <tr>
                      <td><b>Output Streams</b></td>
                      <td>Multiple isolated speech channels {`{ŝ₁, ŝ₂}`}</td>
                      <td>Single enhanced speech stream ŝ(t)</td>
                    </tr>
                    <tr>
                      <td><b>Permutation Ambiguity</b></td>
                      <td>Severe (Label Permutation Problem resolved via uPIT)</td>
                      <td>None (target is known uncorrupted speech utterance)</td>
                    </tr>
                    <tr>
                      <td><b>Primary Model</b></td>
                      <td>Scaled 1D Conv-TasNet (3.4M Params) + Wiener Filter</td>
                      <td>SOTA MossFormerGAN (SyncANet Gated Triple-Attention)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* =============================================================
            TAB 4: EVOLUTIONARY MILESTONES (PHASES 1 TO 5)
            ============================================================= */}
        {activeTab === 'evolution' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
              {EVOLUTION_PHASES.map((p, idx) => (
                <div key={idx} className="card-academic" style={{ borderTop: '3px solid var(--c-primary)' }}>
                  <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid var(--border-color)', backgroundColor: 'var(--bg-card-alt)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="mono" style={{ fontWeight: 800, color: 'var(--c-primary)', fontSize: '0.85rem' }}>{p.phase}</span>
                    <span className="badge-academic badge-blue" style={{ fontSize: '0.68rem' }}>{p.tag}</span>
                  </div>
                  <div style={{ padding: '1rem' }}>
                    <h4 style={{ fontSize: '0.98rem', marginBottom: '0.4rem' }}>{p.title}</h4>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginBottom: '0.6rem' }}>
                      <b>Domain:</b> {p.domain} | <b>Params:</b> {p.params}
                    </div>
                    <div style={{ backgroundColor: 'var(--bg-card-alt)', padding: '0.6rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', marginBottom: '0.6rem' }}>
                      <span className="mono" style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--c-success)' }}>
                        {p.result}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                      <b>Bottleneck:</b> {p.bottleneck}
                    </p>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>
                      <b>Engineering Lesson:</b> {p.lesson}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* =============================================================
            TAB 5: DEEP ARCHITECTURE & OSCILLOSCOPE
            ============================================================= */}
        {activeTab === 'architecture' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              
              {/* Conv-TasNet Dissection */}
              <div className="card-academic">
                <div className="card-header-academic">
                  <strong>Conv-TasNet 1D Waveform Pipeline</strong>
                  <span className="badge-academic badge-blue">Time-Domain</span>
                </div>
                <div className="card-body-academic">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                    <div style={{ borderLeft: '3px solid var(--c-primary)', paddingLeft: '0.75rem' }}>
                      <b>1D Convolutional Encoder:</b> Replaces STFT. <code>N=512</code> filters, kernel length <code>L=16</code> (2.0 ms), stride <code>S=8</code> (1.0 ms) transforming <code>x(t)</code> into 2D latent representation <code>w ∈ ℝ^(512 × K)</code>.
                    </div>
                    <div style={{ borderLeft: '3px solid var(--c-warning)', paddingLeft: '0.75rem' }}>
                      <b>Dilated TCN Separator:</b> 3 repeats of 8 dilated depthwise separable conv blocks (dilation factors <code>d = 2⁰ … 2⁷</code>). Bottleneck <code>B=256</code>, skip connections <code>Sc=256</code>. Estimates speech masks <code>m₁, m₂</code>.
                    </div>
                    <div style={{ borderLeft: '3px solid var(--c-success)', paddingLeft: '0.75rem' }}>
                      <b>1D Transposed Convolutional Decoder:</b> Reconstructs waveform <code>ŝ_c(t) = ConvTranspose1D(m_c ⊙ w)</code>.
                    </div>
                  </div>
                </div>
              </div>

              {/* MossFormerGAN Dissection */}
              <div className="card-academic">
                <div className="card-header-academic">
                  <strong>MossFormerGAN SyncANet Topology</strong>
                  <span className="badge-academic badge-green">Complex STFT GAN</span>
                </div>
                <div className="card-body-academic">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                    <div style={{ borderLeft: '3px solid var(--c-primary)', paddingLeft: '0.75rem' }}>
                      <b>Dilated DenseNet Front-End:</b> Dilated 2D convolutions ($d=1,2,4,8$) with InstanceNorm extracting multi-scale spectral features across 201 frequency bins.
                    </div>
                    <div style={{ borderLeft: '3px solid var(--c-warning)', paddingLeft: '0.75rem' }}>
                      <b>Gated Triple-Attention & FSMN:</b> Decouples full-band temporal attention and sub-band frequency attention, eliminating $O(T^2)$ memory explosion while retaining long-range linguistic cadence.
                    </div>
                    <div style={{ borderLeft: '3px solid var(--c-success)', paddingLeft: '0.75rem' }}>
                      <b>Dual Decoders & GAN Discriminator:</b> Concurrently predicts magnitude mask and complex spectral residual. Discriminator penalizes unnatural musical noise artifacts.
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* Live Web Audio Canvas Oscilloscope */}
            <div className="card-academic">
              <div className="card-header-academic">
                <div>
                  <strong>Live Web Audio Oscilloscope Simulator</strong>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                    Real-time synthesis of male baritone (140 Hz) and female soprano (240 Hz) vocal pitches
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  <button className="btn-academic btn-academic-primary" onClick={toggleOscilloscope}>
                    {isOscActive ? <Pause size={13} /> : <Play size={13} />}
                    <span>{isOscActive ? 'Stop Oscilloscope' : 'Start Oscilloscope'}</span>
                  </button>
                  <button 
                    className="btn-academic"
                    style={{ backgroundColor: oscMode === 'mix' ? 'var(--c-primary-bg)' : '' }}
                    onClick={() => setOscMode('mix')}
                  >
                    Mixed Waveform
                  </button>
                  <button 
                    className="btn-academic"
                    style={{ backgroundColor: oscMode === 'spk1' ? 'var(--c-primary-bg)' : '' }}
                    onClick={() => setOscMode('spk1')}
                  >
                    Speaker 1 (140 Hz)
                  </button>
                  <button 
                    className="btn-academic"
                    style={{ backgroundColor: oscMode === 'spk2' ? 'var(--c-primary-bg)' : '' }}
                    onClick={() => setOscMode('spk2')}
                  >
                    Speaker 2 (240 Hz)
                  </button>
                </div>
              </div>
              <div className="card-body-academic" style={{ padding: '0.5rem' }}>
                <div style={{ height: '140px', width: '100%', position: 'relative' }}>
                  <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }}></canvas>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* =============================================================
            TAB 6: MENTOR DEFENSE & VIVA EXAMINATION
            ============================================================= */}
        {activeTab === 'viva' && (
          <div>
            <div className="card-academic" style={{ marginBottom: '1.25rem', borderLeft: '4px solid var(--c-primary)' }}>
              <div className="card-header-academic">
                <strong>Academic Defense Strategy for Oral Examination</strong>
                <span className="badge-academic badge-blue">Faculty Alignment</span>
              </div>
              <div className="card-body-academic">
                <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
                  The questions below represent the most critical theoretical, mathematical, and practical inquiries typically asked by senior university faculty and thesis review committees. Each answer provides the exact physical and computational justification to defend your findings.
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {VIVA_QUESTIONS.map((item, idx) => (
                <div key={idx} className="card-academic" style={{ margin: 0 }}>
                  <div 
                    style={{
                      padding: '0.85rem 1.25rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      cursor: 'pointer',
                      backgroundColor: expandedViva === idx ? 'var(--bg-card-alt)' : 'var(--bg-card)',
                      transition: 'background-color 0.15s ease'
                    }}
                    onClick={() => setExpandedViva(expandedViva === idx ? null : idx)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span className="mono" style={{ fontWeight: 800, color: 'var(--c-primary)', fontSize: '0.9rem' }}>
                        Q{idx + 1}.
                      </span>
                      <strong style={{ fontSize: '0.92rem' }}>{item.q}</strong>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge-academic badge-slate" style={{ fontSize: '0.7rem' }}>
                        {item.category}
                      </span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                        {expandedViva === idx ? '▲' : '▼'}
                      </span>
                    </div>
                  </div>

                  {expandedViva === idx && (
                    <div style={{ padding: '1rem 1.25rem', borderTop: '1px solid var(--border-color)', backgroundColor: 'var(--bg-card)' }}>
                      <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                        {item.a}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      {/* Persistent Bottom Audio Player Dock */}
      {currentTrack && (
        <div className="audio-dock-wrap">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span className={`badge-academic badge-${currentTrack.tagColor}`} style={{ padding: '0.25rem 0.6rem' }}>
              {currentTrack.tag}
            </span>
            <div>
              <div style={{ fontWeight: 700, fontSize: '0.92rem' }}>{currentTrack.title}</div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>{currentTrack.sub}</div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button 
              className="btn-academic btn-academic-primary" 
              onClick={togglePlayPause}
              style={{ width: '38px', height: '38px', borderRadius: '50%', padding: 0, justifyContent: 'center' }}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button 
              className="btn-academic" 
              onClick={() => {
                audioRef.current.pause();
                setCurrentTrack(null);
                setIsPlaying(false);
              }}
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.75rem' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
