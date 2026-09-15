import React, { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader';
import { getClasses } from '../services/api';
import { FALLBACK_CLASSES } from '../utils/constants';

const NOVELTIES = [
  {
    num: '01',
    title: 'Hierarchical Multi-Scale Temporal Modelling',
    color: 'var(--accent-blue)',
    desc: 'Instead of processing the complete sign sequence at a single temporal resolution, HMS-Mamba learns fine (T=32), intermediate (T=16), and global (T=8) temporal representations simultaneously.',
  },
  {
    num: '02',
    title: 'RGB-Skeletal Multimodal Learning',
    color: 'var(--accent-purple)',
    desc: 'MobileNetV3-Large provides visual appearance features while MediaPipe Holistic landmarks provide explicit geometric information. The two modalities are fused to form a complementary representation.',
  },
  {
    num: '03',
    title: 'Explicit Motion-Aware Landmark Encoding',
    color: 'var(--accent-cyan)',
    desc: 'First-order temporal differences (Δt = Lt − Lt−1) are computed from normalized landmark coordinates to explicitly encode movement direction, magnitude, and temporal displacement.',
  },
  {
    num: '04',
    title: 'Missing-Landmark Awareness',
    color: 'var(--accent-green)',
    desc: 'A per-frame validity mask distinguishes actual MediaPipe detections from zero-filled missing landmarks, preventing the model from treating detection failures as genuine coordinate values.',
  },
  {
    num: '05',
    title: 'Reliability-Aware Multimodal Fusion',
    color: 'var(--accent-amber)',
    desc: 'An optional gating mechanism allows the network to learn whether RGB or skeletal information is more reliable for a given temporal region, based on motion blur or landmark occlusion.',
  },
  {
    num: '06',
    title: 'Confidence-Aware Recognition',
    color: 'var(--accent-blue)',
    desc: 'Low-confidence predictions are reported as UNCERTAIN rather than forcing every unknown gesture into a known class, improving practical reliability.',
  },
  {
    num: '07',
    title: 'Confusion-Aware Evaluation',
    color: 'var(--accent-purple)',
    desc: 'The system identifies frequently confused ASL class pairs for targeted analysis, motivating future hard-example training and class-pair investigation.',
  },
  {
    num: '08',
    title: 'Real-Time Temporal Stabilization',
    color: 'var(--accent-cyan)',
    desc: 'Webcam predictions use probability-based temporal smoothing over a rolling window to reduce unstable label switching during live demonstration.',
  },
  {
    num: '09',
    title: 'Edge-Oriented Lightweight Architecture',
    color: 'var(--accent-green)',
    desc: 'MobileNetV3-Large and efficient temporal modelling are combined for practical real-time inference on supported hardware.',
  },
];

function CompareCard({ title, color, items }) {
  return (
    <div style={{ padding: '20px', borderRadius: 'var(--radius-lg)', background: `${color}08`, border: `1px solid ${color}25` }}>
      <div style={{ fontSize: '0.85rem', fontWeight: 700, color, marginBottom: '14px' }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {items.map((item) => (
          <div key={item} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: color, flexShrink: 0 }} />
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function About() {
  const [classes, setClasses] = useState(null);

  useEffect(() => {
    getClasses().then((d) => setClasses(d.classes)).catch(() => {});
  }, []);

  const displayClasses = classes || FALLBACK_CLASSES;

  return (
    <div>
      <PageHeader
        label="About"
        title="About LightMamba-ASL"
        subtitle="Academic research project — video-level American Sign Language word recognition."
      />

      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

        {/* ── Project Overview ─────────────────────── */}
        <div className="glass-card" style={{ padding: '28px' }}>
          <div className="section-label" style={{ marginBottom: '12px' }}>Project Overview</div>
          <p style={{ fontSize: '0.92rem', color: 'var(--text-secondary)', lineHeight: 1.8, maxWidth: '800px' }}>
            LightMamba-ASL is a lightweight multimodal video recognition framework for dynamic American Sign Language
            word recognition. Unlike static image-based approaches, the system analyzes complete temporal gesture
            sequences. RGB appearance features extracted using MobileNetV3-Large are combined with MediaPipe skeletal
            landmarks and explicit temporal motion features. The fused sequence is processed by Hierarchical Multi-Scale
            Mamba to learn fine, intermediate, and global gesture patterns before producing a video-level ASL word prediction.
          </p>
        </div>

        {/* ── Video vs Image ───────────────────────── */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div className="section-label" style={{ marginBottom: '16px' }}>Research Motivation</div>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
            Why Video Instead of Image?
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <CompareCard
              title="Static Image ASL"
              color="var(--accent-red)"
              items={['Single frame only', 'Hand appearance only', 'No motion history', 'No trajectory information', 'No temporal context']}
            />
            <CompareCard
              title="LightMamba-ASL (Video)"
              color="var(--accent-green)"
              items={['Complete MP4 sequence', '32 ordered frames', 'RGB appearance + skeleton', 'Explicit motion encoding', 'Hierarchical temporal modelling', 'Word-level recognition']}
            />
          </div>
        </div>

        {/* ── Why RGB + Skeleton ───────────────────── */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div className="section-label" style={{ marginBottom: '16px' }}>Multimodal Design</div>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
            Why RGB + Skeleton?
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <CompareCard
              title="RGB Features"
              color="var(--accent-blue)"
              items={['Visual appearance', 'Hand texture', 'Background context', 'Shape information', 'Lighting variation']}
            />
            <CompareCard
              title="Skeleton Features"
              color="var(--accent-purple)"
              items={['Joint geometry', 'Finger positions', 'Hand trajectory', 'Body-relative motion', 'Two-hand coordination']}
            />
          </div>
          <div style={{ marginTop: '16px', padding: '14px', borderRadius: 'var(--radius-md)', background: 'rgba(6,182,212,0.08)', border: '1px solid rgba(6,182,212,0.2)', textAlign: 'center', fontSize: '0.82rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>
            RGB + Skeleton → Complementary Multimodal Representation
          </div>
        </div>

        {/* ── Research Novelty ─────────────────────── */}
        <div>
          <div className="section-label" style={{ marginBottom: '8px' }}>Proposed Components</div>
          <h2 className="section-title" style={{ fontSize: '1.2rem', marginBottom: '4px' }}>Research Novelty</h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
            These are proposed research components. Their actual benefit must be demonstrated through ablation experiments.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '14px' }}>
            {NOVELTIES.map(({ num, title, color, desc }) => (
              <div key={num} className="glass-card glass-card-hover" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                  <div style={{ fontSize: '1.2rem', fontWeight: 900, color, opacity: 0.5, fontFamily: "'JetBrains Mono', monospace", flexShrink: 0, lineHeight: 1 }}>{num}</div>
                  <div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px', lineHeight: 1.3 }}>{title}</div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{desc}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Supported Signs ──────────────────────── */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div className="section-label" style={{ marginBottom: '12px' }}>Phase 1 Dataset</div>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
            Supported ASL Signs
            {!classes && <span style={{ fontSize: '0.7rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: '8px' }}>(fallback — backend offline)</span>}
          </h3>
          <div className="classes-grid">
            {displayClasses.map((cls) => (
              <span key={cls} className="class-chip">{cls}</span>
            ))}
          </div>
        </div>

        {/* ── Tech Stack ───────────────────────────── */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div className="section-label" style={{ marginBottom: '14px' }}>Technology Stack</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '10px' }}>
            {[
              { label: 'PyTorch',          sub: 'Deep learning framework',    color: 'var(--accent-red)' },
              { label: 'MobileNetV3-Large',sub: 'RGB feature extraction',     color: 'var(--accent-blue)' },
              { label: 'MediaPipe',        sub: 'Landmark detection',         color: 'var(--accent-green)' },
              { label: 'Mamba / SSM',      sub: 'Temporal sequence model',    color: 'var(--accent-purple)' },
              { label: 'Flask',            sub: 'REST API backend',           color: 'var(--accent-cyan)' },
              { label: 'React CRA',        sub: 'Frontend interface',         color: 'var(--accent-amber)' },
              { label: 'PopSign Dataset',  sub: '4 supported ASL signs',      color: 'var(--accent-blue)' },
              { label: 'OpenCV',           sub: 'Video decoding',             color: 'var(--accent-green)' },
            ].map(({ label, sub, color }) => (
              <div key={label} style={{ padding: '12px', borderRadius: 'var(--radius-md)', background: `${color}08`, border: `1px solid ${color}20` }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{label}</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '2px' }}>{sub}</div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
