import React from 'react';
import { formatPercent, getConfidenceLevel, getConfidenceFillClass } from '../utils/formatConfidence';

export function ConfidenceBar({ confidence }) {
  const level = getConfidenceLevel(confidence);
  const fillClass = getConfidenceFillClass(confidence);
  const pct = Math.round((confidence || 0) * 100);

  return (
    <div>
      <div className="confidence-row">
        <span className="confidence-label">Confidence</span>
        <span className="confidence-value" style={{ color: level.color }}>
          {formatPercent(confidence)} — {level.label}
        </span>
      </div>
      <div className="confidence-track">
        <div className={`confidence-fill ${fillClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function PredictionCard({ result }) {
  if (!result) return null;
  const { prediction, confidence, uncertain, ambiguous, ambiguity, top_predictions, processing_time_ms } = result;

  return (
    <div className="prediction-result-card">
      <div className="prediction-label">Recognized Sign</div>
      <div className={`prediction-word gradient-text${uncertain ? ' uncertain' : ''}`}>
        {prediction?.toUpperCase()}
      </div>

      {ambiguous ? (
        <p style={{ fontSize: '0.82rem', color: 'var(--accent-amber)', marginBottom: '12px', lineHeight: 1.55 }}>
          Possible signs: <strong>{ambiguity?.labels?.join(' / ')}</strong>. {ambiguity?.message}
        </p>
      ) : uncertain && (
        <p style={{ fontSize: '0.82rem', color: 'var(--accent-amber)', marginBottom: '12px' }}>
          ⚠️ Low confidence — result may not be accurate.
        </p>
      )}

      <ConfidenceBar confidence={confidence} />

      {top_predictions && top_predictions.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <div style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '10px' }}>
            Top Predictions
          </div>
          <TopPredictions predictions={top_predictions} />
        </div>
      )}

      {processing_time_ms != null && (
        <div className="processing-time-row">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          Backend inference: <strong>{processing_time_ms.toFixed(1)} ms</strong>
        </div>
      )}
    </div>
  );
}

export function TopPredictions({ predictions }) {
  if (!predictions || predictions.length === 0) return null;
  const maxConf = predictions[0]?.confidence || 1;

  const barColors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b'];

  return (
    <div className="top-predictions">
      {predictions.map((p, i) => (
        <div key={p.class} className="top-pred-row">
          <span className="top-pred-rank">#{i + 1}</span>
          <span className="top-pred-label">{p.class}</span>
          <div className="top-pred-bar-wrap">
            <div
              className="top-pred-bar"
              style={{
                width: `${((p.confidence / maxConf) * 100).toFixed(1)}%`,
                background: barColors[i % barColors.length],
              }}
            />
          </div>
          <span className="top-pred-pct">{formatPercent(p.confidence)}</span>
        </div>
      ))}
    </div>
  );
}
