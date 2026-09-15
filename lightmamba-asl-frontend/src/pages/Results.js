import React, { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { API_BASE_URL } from '../services/api';

/* ── Helpers ───────────────────────────────────────────── */
function MetricBox({ label, value, color = 'var(--accent-blue)' }) {
  return (
    <div style={{ padding: '16px', borderRadius: 'var(--radius-md)', background: `${color}10`, border: `1px solid ${color}25`, textAlign: 'center' }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 800, color }}>{value ?? '—'}</div>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px' }}>{label}</div>
    </div>
  );
}

function EmptyState({ message }) {
  return (
    <div className="empty-state">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <p>{message}</p>
    </div>
  );
}

const ABLATION_CONFIGS = [
  { name: 'RGB Only',                  rgb: true,  lm: false, motion: false, multiscale: false, reliability: false },
  { name: 'Landmark Only',             rgb: false, lm: true,  motion: false, multiscale: false, reliability: false },
  { name: 'RGB + Landmark',            rgb: true,  lm: true,  motion: false, multiscale: false, reliability: false },
  { name: 'RGB + Landmark + Motion',   rgb: true,  lm: true,  motion: true,  multiscale: false, reliability: false },
  { name: 'Single-Scale Mamba',        rgb: true,  lm: true,  motion: true,  multiscale: false, reliability: false },
  { name: 'HMS-Mamba (Proposed)',      rgb: true,  lm: true,  motion: true,  multiscale: true,  reliability: false },
  { name: 'HMS-Mamba + Reliability',   rgb: true,  lm: true,  motion: true,  multiscale: true,  reliability: true  },
];

const Tick = ({ v }) => v
  ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-green)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
  : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>;

export default function Results() {
  const [history, setHistory]   = useState(null);
  const [metrics, setMetrics]   = useState(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [hRes, mRes] = await Promise.allSettled([
          fetch(`${API_BASE_URL}/outputs/metrics/training_history.json`).then((r) => r.ok ? r.json() : null).catch(() => null),
          fetch(`${API_BASE_URL}/outputs/metrics/test_metrics.json`).then((r) => r.ok ? r.json() : null).catch(() => null),
        ]);
        // Try local static files as fallback
        if (hRes.status === 'fulfilled' && hRes.value) setHistory(hRes.value);
        if (mRes.status === 'fulfilled' && mRes.value) setMetrics(mRes.value);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  /* Build recharts data from history */
  const chartData = history?.train_loss?.map((_, i) => ({
    epoch: i + 1,
    trainLoss: parseFloat(history.train_loss[i]?.toFixed(4)),
    valLoss:   parseFloat(history.val_loss[i]?.toFixed(4)),
    trainAcc:  parseFloat((history.train_acc[i] * 100)?.toFixed(2)),
    valAcc:    parseFloat((history.val_acc[i] * 100)?.toFixed(2)),
  }));

  const fmt = (v) => v != null ? `${(v * 100).toFixed(2)}%` : '—';

  return (
    <div>
      <PageHeader
        label="Experimental Results"
        title="Research Results"
        subtitle="All metrics are generated from actual model training and evaluation. No values are fabricated."
      />

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '48px' }}>
          <LoadingSpinner label="Loading results…" />
        </div>
      )}

      {!loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

          {/* ── Overall Metrics ──────────────────────── */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <div className="section-label" style={{ marginBottom: '16px' }}>Test Set Performance</div>
            {metrics ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px' }}>
                <MetricBox label="Top-1 Accuracy"  value={fmt(metrics.accuracy)}          color="var(--accent-blue)" />
                <MetricBox label="Top-5 Accuracy"  value={fmt(metrics.top5_accuracy)}     color="var(--accent-purple)" />
                <MetricBox label="Macro F1"         value={metrics.f1_macro?.toFixed(4)}   color="var(--accent-cyan)" />
                <MetricBox label="Weighted F1"      value={metrics.f1_weighted?.toFixed(4)}color="var(--accent-green)" />
                <MetricBox label="Macro Precision"  value={metrics.precision_macro?.toFixed(4)} color="var(--accent-amber)" />
                <MetricBox label="Macro Recall"     value={metrics.recall_macro?.toFixed(4)}    color="var(--accent-blue)" />
              </div>
            ) : (
              <EmptyState message="No test metrics available yet. Run: python -m backend.evaluation.evaluate" />
            )}
          </div>

          {/* ── Per-Class Accuracy ───────────────────── */}
          {metrics?.per_class_accuracy && (
            <div className="glass-card" style={{ padding: '24px' }}>
              <div className="section-label" style={{ marginBottom: '16px' }}>Per-Class Accuracy</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(metrics.per_class_accuracy).map(([cls, acc]) => (
                  <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ width: '80px', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', textTransform: 'capitalize' }}>{cls}</span>
                    <div style={{ flex: 1, height: '6px', background: 'var(--bg-secondary)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${(acc * 100).toFixed(1)}%`, background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', borderRadius: '3px', transition: 'width 0.5s ease' }} />
                    </div>
                    <span style={{ width: '48px', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textAlign: 'right' }}>{(acc * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Efficiency ───────────────────────────── */}
          {metrics?.efficiency && (
            <div className="glass-card" style={{ padding: '24px' }}>
              <div className="section-label" style={{ marginBottom: '16px' }}>Model Efficiency</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
                <MetricBox label="Total Parameters"  value={metrics.efficiency.total_parameters?.toLocaleString()} color="var(--accent-blue)" />
                <MetricBox label="Model Size (MB)"   value={metrics.efficiency.model_file_size_mb?.toFixed(2)}    color="var(--accent-purple)" />
                <MetricBox label="Latency (ms)"      value={metrics.efficiency.inference_latency_ms?.toFixed(1)}  color="var(--accent-cyan)" />
                <MetricBox label="Approx. FPS"       value={metrics.efficiency.approximate_fps?.toFixed(1)}       color="var(--accent-green)" />
                <MetricBox label="Device"            value={metrics.efficiency.execution_device}                  color="var(--accent-amber)" />
              </div>
            </div>
          )}

          {/* ── Training Curves ──────────────────────── */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <div className="section-label" style={{ marginBottom: '16px' }}>Training Curves</div>
            {chartData ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                {/* Loss */}
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px' }}>Loss</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="epoch" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                      <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: '8px', color: 'var(--text-primary)' }} />
                      <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="valLoss"   name="Val Loss"   stroke="#8b5cf6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Accuracy */}
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px' }}>Accuracy (%)</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="epoch" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                      <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} domain={[0, 100]} />
                      <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: '8px', color: 'var(--text-primary)' }} />
                      <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="trainAcc" name="Train Acc" stroke="#10b981" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="valAcc"   name="Val Acc"   stroke="#f59e0b" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <EmptyState message="Training curves will appear after model training completes. Run: python -m backend.training.train" />
            )}
          </div>

          {/* ── Confusion Matrix ─────────────────────── */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <div className="section-label" style={{ marginBottom: '16px' }}>Confusion Matrix</div>
            <div style={{ textAlign: 'center' }}>
              <img
                src={`${API_BASE_URL}/outputs/confusion_matrix/confusion_matrix.png`}
                alt="Confusion Matrix"
                style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-card)' }}
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
              />
              <div style={{ display: 'none' }}>
                <EmptyState message="Confusion matrix will appear after evaluation. Run: python -m backend.evaluation.evaluate" />
              </div>
            </div>
          </div>

          {/* ── Ablation Table ───────────────────────── */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <div className="section-label" style={{ marginBottom: '16px' }}>Ablation Study</div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
              Accuracy and F1 columns will be populated after running each ablation experiment.
            </p>
            <div className="scroll-x">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-card)' }}>
                    {['Experiment', 'RGB', 'Landmarks', 'Motion', 'Multi-Scale', 'Reliability', 'Accuracy', 'F1', 'Latency'].map((h) => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ABLATION_CONFIGS.map((cfg, i) => (
                    <tr key={cfg.name} style={{ borderBottom: '1px solid var(--border-card)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>{cfg.name}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><Tick v={cfg.rgb} /></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><Tick v={cfg.lm} /></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><Tick v={cfg.motion} /></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><Tick v={cfg.multiscale} /></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><Tick v={cfg.reliability} /></td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-muted)', fontStyle: 'italic' }}>—</td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-muted)', fontStyle: 'italic' }}>—</td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-muted)', fontStyle: 'italic' }}>—</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
