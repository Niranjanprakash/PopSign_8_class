import React, { useState, useEffect, useRef } from 'react';
import PageHeader    from '../components/PageHeader';
import VideoUploader from '../components/VideoUploader';
import VideoSkeletonPlayer from '../components/VideoSkeletonPlayer';
import { PredictionCard } from '../components/PredictionCard';
import ConfidenceGraph from '../components/ConfidenceGraph';
import ErrorMessage  from '../components/ErrorMessage';
import LoadingSpinner from '../components/LoadingSpinner';
import { predictVideo } from '../services/api';
import { createObjectURL, revokeObjectURL } from '../utils/fileUtils';

/* ── Processing stages shown during inference ─────────── */
const STAGES = [
  'Uploading Video',
  'Sampling 32 Frames',
  'Extracting RGB Features',
  'Extracting Skeletal Features',
  'Encoding Motion',
  'HMS-Mamba Temporal Analysis',
  'Generating Prediction',
];

const AnalyzeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
);
const ResetIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/>
  </svg>
);
const CheckIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

export default function VideoRecognition() {
  const [file, setFile]           = useState(null);
  const [objectUrl, setObjectUrl] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [stageIdx, setStageIdx]   = useState(-1);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const stageTimer                = useRef(null);

  /* Clean up object URL on unmount */
  useEffect(() => () => { if (objectUrl) revokeObjectURL(objectUrl); }, [objectUrl]);

  function handleFileSelected(f) {
    if (objectUrl) revokeObjectURL(objectUrl);
    setFile(f);
    setObjectUrl(createObjectURL(f));
    setResult(null);
    setError(null);
    setStageIdx(-1);
  }

  function reset() {
    if (objectUrl) revokeObjectURL(objectUrl);
    setFile(null);
    setObjectUrl(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setStageIdx(-1);
    clearInterval(stageTimer.current);
  }

  async function handleAnalyze() {
    if (!file || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setStageIdx(0);

    /* Advance stage indicator every ~600 ms for UX feedback */
    let idx = 0;
    stageTimer.current = setInterval(() => {
      idx = Math.min(idx + 1, STAGES.length - 2);
      setStageIdx(idx);
    }, 600);

    try {
      const data = await predictVideo(file);
      clearInterval(stageTimer.current);
      setStageIdx(STAGES.length - 1);
      if (data.success === false) {
        setError(data.error || 'Prediction failed.');
      } else {
        setResult(data);
      }
    } catch (err) {
      clearInterval(stageTimer.current);
      setError(err.message || 'Backend request failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        label="ASL Recognition"
        title="Video Recognition"
        subtitle="Upload a complete MP4 gesture and analyze its spatial and temporal features using LightMamba-ASL."
      >
        {file && (
          <button className="btn btn-ghost btn-sm" onClick={reset}>
            <ResetIcon /> Analyze Another Video
          </button>
        )}
      </PageHeader>

      {!file ? (
        <VideoUploader onFileSelected={handleFileSelected} />
      ) : (
        <div className="recognition-layout">
          {/* ── LEFT: Video + Skeleton ─────────────────── */}
          <div>
            <VideoSkeletonPlayer file={file} objectUrl={objectUrl} />

            <div style={{ marginTop: '16px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? <LoadingSpinner size={14} /> : <AnalyzeIcon />}
                {loading ? 'Analyzing…' : 'Analyze Sign'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={reset}>
                <ResetIcon /> Change Video
              </button>
            </div>

            {error && <div style={{ marginTop: '12px' }}><ErrorMessage message={error} onDismiss={() => setError(null)} /></div>}
          </div>

          {/* ── RIGHT: Stages + Result ─────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Processing stages */}
            {loading && (
              <div className="processing-stages">
                <div style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '4px' }}>
                  Processing Pipeline
                </div>
                {STAGES.map((s, i) => (
                  <div
                    key={s}
                    className={`processing-stage${i === stageIdx ? ' active' : i < stageIdx ? ' done' : ''}`}
                  >
                    <div className={`stage-icon${i === stageIdx ? ' active' : i < stageIdx ? ' done' : ' pending'}`}>
                      {i < stageIdx ? <CheckIcon /> : i + 1}
                    </div>
                    {s}
                  </div>
                ))}
              </div>
            )}

            {/* Prediction result */}
            {result && <PredictionCard result={result} />}
            {/* Confidence graph */}
            <ConfidenceGraph predictions={result?.top_predictions ?? []} />

            {/* Idle hint */}
            {!loading && !result && !error && (
              <div className="glass-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '8px' }}>
                  Ready
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  Click <strong style={{ color: 'var(--text-primary)' }}>Analyze Sign</strong> to send the original MP4 to the
                  LightMamba-ASL backend for temporal inference.
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                  The backend samples 32 frames, extracts RGB + MediaPipe features, and runs HMS-Mamba.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
