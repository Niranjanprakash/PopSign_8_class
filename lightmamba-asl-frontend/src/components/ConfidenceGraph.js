import React, { useEffect, useRef, useState } from 'react';
import { formatPercent } from '../utils/formatConfidence';

const CLASS_COLORS = [
  { bar: '#3b82f6', glow: 'rgba(59,130,246,0.35)',  track: 'rgba(59,130,246,0.08)'  },
  { bar: '#8b5cf6', glow: 'rgba(139,92,246,0.35)',  track: 'rgba(139,92,246,0.08)'  },
  { bar: '#06b6d4', glow: 'rgba(6,182,212,0.35)',   track: 'rgba(6,182,212,0.08)'   },
  { bar: '#10b981', glow: 'rgba(16,185,129,0.35)',  track: 'rgba(16,185,129,0.08)'  },
  { bar: '#f59e0b', glow: 'rgba(245,158,11,0.35)',  track: 'rgba(245,158,11,0.08)'  },
  { bar: '#ec4899', glow: 'rgba(236,72,153,0.35)',  track: 'rgba(236,72,153,0.08)'  },
  { bar: '#14b8a6', glow: 'rgba(20,184,166,0.35)',  track: 'rgba(20,184,166,0.08)'  },
  { bar: '#a78bfa', glow: 'rgba(167,139,250,0.35)', track: 'rgba(167,139,250,0.08)' },
];

// Smooth lerp for animated transitions
function lerp(a, b, t) { return a + (b - a) * t; }

export default function ConfidenceGraph({ predictions, isLive = false }) {
  const animRef   = useRef(null);
  const prevRef   = useRef({});
  const [display, setDisplay] = useState({});

  // Animate toward target values
  useEffect(() => {
    if (!predictions || predictions.length === 0) return;

    const target = {};
    predictions.forEach(p => { target[p.class] = p.confidence; });

    // Init missing keys at 0
    predictions.forEach(p => {
      if (prevRef.current[p.class] === undefined) prevRef.current[p.class] = 0;
    });

    cancelAnimationFrame(animRef.current);

    function step() {
      let stillMoving = false;
      const next = {};
      predictions.forEach(p => {
        const cur = prevRef.current[p.class] ?? 0;
        const tgt = target[p.class] ?? 0;
        const val = lerp(cur, tgt, 0.18);
        next[p.class] = val;
        if (Math.abs(val - tgt) > 0.001) stillMoving = true;
      });
      prevRef.current = next;
      setDisplay({ ...next });
      if (stillMoving) animRef.current = requestAnimationFrame(step);
    }

    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [predictions]);

  if (!predictions || predictions.length === 0) {
    return (
      <div className="cg-card">
        <div className="cg-header">
          <div className="cg-title-row">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
              <line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>
            </svg>
            <span>Confidence Graph</span>
          </div>
          {isLive && <div className="cg-live-badge"><span className="cg-live-dot" />LIVE</div>}
        </div>
        <div className="cg-empty">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.25 }}>
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>
          </svg>
          <p>Analyze a gesture to see confidence scores</p>
        </div>
      </div>
    );
  }

  // Sort by confidence descending for display
  const sorted = [...predictions].sort((a, b) => b.confidence - a.confidence);
  const maxConf = sorted[0]?.confidence || 1;
  const topClass = sorted[0]?.class;

  return (
    <div className="cg-card">
      {/* Header */}
      <div className="cg-header">
        <div className="cg-title-row">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>
          </svg>
          <span>Confidence Graph</span>
        </div>
        {isLive && <div className="cg-live-badge"><span className="cg-live-dot" />LIVE</div>}
      </div>

      {/* Bars */}
      <div className="cg-bars">
        {sorted.map((p, i) => {
          const color   = CLASS_COLORS[i % CLASS_COLORS.length];
          const animVal = display[p.class] ?? 0;
          const pct     = Math.round(animVal * 100);
          const barW    = ((animVal / maxConf) * 100).toFixed(2);
          const isTop   = p.class === topClass;

          return (
            <div key={p.class} className={`cg-row${isTop ? ' cg-row-top' : ''}`}>
              {/* Rank */}
              <div className="cg-rank" style={{ color: isTop ? color.bar : 'var(--text-muted)' }}>
                {isTop ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill={color.bar} stroke="none">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                  </svg>
                ) : (
                  <span>#{i + 1}</span>
                )}
              </div>

              {/* Label */}
              <div className="cg-label" style={{ color: isTop ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                {p.class}
              </div>

              {/* Bar track + fill */}
              <div className="cg-track" style={{ background: color.track }}>
                <div
                  className="cg-fill"
                  style={{
                    width: `${barW}%`,
                    background: isTop
                      ? `linear-gradient(90deg, ${color.bar}cc, ${color.bar})`
                      : color.bar,
                    boxShadow: isTop ? `0 0 10px ${color.glow}` : 'none',
                    opacity: isTop ? 1 : 0.55,
                  }}
                />
                {/* Animated shimmer on top bar */}
                {isTop && <div className="cg-shimmer" style={{ background: `linear-gradient(90deg, transparent, ${color.bar}40, transparent)` }} />}
              </div>

              {/* Percentage */}
              <div className="cg-pct" style={{ color: isTop ? color.bar : 'var(--text-muted)' }}>
                {pct}%
              </div>
            </div>
          );
        })}
      </div>

      {/* SVG sparkline axis */}
      <div className="cg-axis">
        {[0, 25, 50, 75, 100].map(v => (
          <span key={v} className="cg-axis-tick">{v}%</span>
        ))}
      </div>
    </div>
  );
}
