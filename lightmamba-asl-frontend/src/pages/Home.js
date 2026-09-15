import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getClasses, getModelInfo } from '../services/api';
import { FALLBACK_CLASSES } from '../utils/constants';
import MetricCard    from '../components/MetricCard';
import ArchitectureFlow from '../components/ArchitectureFlow';
import LoadingSpinner from '../components/LoadingSpinner';

/* ── SVG icons ─────────────────────────────────────────── */
const IconLayers = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>
  </svg>
);
const IconFilm = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
    <line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/>
    <line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/>
    <line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/>
    <line x1="17" y1="17" x2="22" y2="17"/>
  </svg>
);
const IconMonitor = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
    <line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
  </svg>
);
const IconCpu = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>
    <path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>
  </svg>
);
const IconActivity = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);
const IconArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
  </svg>
);

export default function Home() {
  const navigate = useNavigate();
  const [classes, setClasses]     = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    Promise.allSettled([getClasses(), getModelInfo()])
      .then(([clsRes, infoRes]) => {
        if (clsRes.status === 'fulfilled') setClasses(clsRes.value.classes);
        if (infoRes.status === 'fulfilled') setModelInfo(infoRes.value);
      })
      .finally(() => setLoading(false));
  }, []);

  const displayClasses = classes || FALLBACK_CLASSES;

  const stats = [
    {
      icon: <IconLayers />,
      value: modelInfo?.num_classes ?? displayClasses.length,
      label: 'ASL Classes',
      sub: 'PopSign 4-class model',
      color: 'var(--accent-blue)',
    },
    {
      icon: <IconFilm />,
      value: modelInfo?.frames_per_video ?? 32,
      label: 'Frames / Video',
      sub: 'Temporal sequence',
      color: 'var(--accent-purple)',
    },
    {
      icon: <IconMonitor />,
      value: modelInfo?.input_resolution ?? '224×224',
      label: 'Frame Resolution',
      sub: 'RGB input size',
      color: 'var(--accent-cyan)',
    },
    {
      icon: <IconCpu />,
      value: 'HMS-Mamba',
      label: 'Temporal Model',
      sub: 'Hierarchical Multi-Scale',
      color: 'var(--accent-green)',
    },
    {
      icon: <IconActivity />,
      value: 'MediaPipe',
      label: 'Skeleton Tracking',
      sub: 'Holistic landmark tracking',
      color: 'var(--accent-amber)',
    },
  ];

  return (
    <div>
      {/* ── Hero ─────────────────────────────────────────── */}
      <div className="hero">
        <div className="hero-eyebrow">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
          </svg>
          Academic Research Project — Video-Level ASL Recognition
        </div>
        <h1 className="hero-title">
          <span className="gradient-text">LightMamba-ASL</span>
          <br />
          Hierarchical Multi-Scale Mamba for ASL Recognition
        </h1>
        <p className="hero-description">
          LightMamba-ASL recognizes dynamic ASL words from complete video sequences by combining
          lightweight RGB appearance features, MediaPipe skeletal landmarks, explicit motion
          information, and hierarchical multi-scale temporal modelling.
        </p>
        <div className="hero-actions">
          <button className="btn btn-primary btn-lg" onClick={() => navigate('/live')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            Try Live ASL Recognition
          </button>
        </div>
      </div>

      {/* ── Stats ────────────────────────────────────────── */}
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '24px' }}>
          <LoadingSpinner label="Loading system info…" />
        </div>
      ) : (
        <div className="stats-grid">
          {stats.map((s) => (
            <MetricCard key={s.label} icon={s.icon} value={s.value} label={s.label} sub={s.sub} accentColor={s.color} />
          ))}
        </div>
      )}

      {/* ── Pipeline ─────────────────────────────────────── */}
      <div className="pipeline-section">
        <div style={{ marginBottom: '16px' }}>
          <div className="section-label">System Architecture</div>
          <h2 className="section-title" style={{ fontSize: '1.3rem' }}>Complete Recognition Pipeline</h2>
          <p className="section-subtitle">
            Each MP4 video is treated as one complete temporal sample — never as individual frames.
          </p>
        </div>
        <div className="pipeline-container">
          <ArchitectureFlow />
        </div>
      </div>

      {/* ── Supported Classes ────────────────────────────── */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div className="section-label">PopSign Dataset</div>
            <h2 className="section-title" style={{ fontSize: '1.2rem' }}>Supported ASL Signs</h2>
            {!classes && <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>Showing fallback list — backend offline</p>}
          </div>
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/live')}>
            Try Live Recognition <IconArrowRight />
          </button>
        </div>
        <div className="classes-grid">
          {displayClasses.map((cls) => (
            <span key={cls} className="class-chip">{cls}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
