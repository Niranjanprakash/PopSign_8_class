import React, { useRef, useState, useCallback, useEffect } from 'react';
import PageHeader     from '../components/PageHeader';
import { LandmarkStatus, SkeletonLegend } from '../components/LandmarkStatus';
import { PredictionCard } from '../components/PredictionCard';
import ConfidenceGraph from '../components/ConfidenceGraph';
import ErrorMessage   from '../components/ErrorMessage';
import LoadingSpinner from '../components/LoadingSpinner';
import { useCamera }  from '../hooks/useCamera';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { predictVideo } from '../services/api';
import { useBackendStatus } from '../hooks/useBackendStatus';

const CamIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);
const StopIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
  </svg>
);
const RecordIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="currentColor"/>
  </svg>
);
const SendIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

export default function LiveRecognition() {
  const videoRef         = useRef(null);
  const canvasRef        = useRef(null);
  const wrapRef          = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);

  const [skeletonOn, setSkeletonOn]     = useState(true);
  const [recording, setRecording]       = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedMime, setRecordedMime] = useState('');
  const [analyzing, setAnalyzing]       = useState(false);
  const [result, setResult]             = useState(null);
  const [error, setError]               = useState(null);
  const [fps, setFps]                   = useState(0);
  const fpsRef = useRef({ count: 0, last: performance.now() });

  const { active, error: camError, startCamera, stopCamera } = useCamera();
  const { tracking, mpReady } = useMediaPipe(videoRef, canvasRef, active && skeletonOn);
  const { online } = useBackendStatus();

  /* FPS counter */
  useEffect(() => {
    if (!active) { setFps(0); return; }
    const id = setInterval(() => {
      const now     = performance.now();
      const elapsed = (now - fpsRef.current.last) / 1000;
      setFps(Math.round(fpsRef.current.count / elapsed));
      fpsRef.current = { count: 0, last: now };
    }, 1000);
    return () => clearInterval(id);
  }, [active]);

  /* Canvas size sync via wrapper div */
  useEffect(() => {
    const wrap   = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width  = wrap.offsetWidth;
      canvas.height = wrap.offsetHeight;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  const handleStartCamera = useCallback(() => startCamera(videoRef.current), [startCamera]);
  const handleStopCamera  = useCallback(() => {
    stopCamera(videoRef.current);
    if (recording) stopRecording(); // eslint-disable-line
  }, [stopCamera, recording]); // eslint-disable-line

  function startRecording() {
    if (!active || !videoRef.current?.srcObject) return;
    chunksRef.current = [];
    const stream = videoRef.current.srcObject;

    const mimeType = MediaRecorder.isTypeSupported('video/mp4')
      ? 'video/mp4'
      : MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : 'video/webm';

    setRecordedMime(mimeType);
    const mr = new MediaRecorder(stream, { mimeType });
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      setRecordedBlob(blob);
    };
    mr.start(100);
    mediaRecorderRef.current = mr;
    setRecording(true);
    setResult(null);
    setError(null);
  }

  function stopRecording() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  }

  async function handleAnalyze() {
    if (!recordedBlob) return;
    setAnalyzing(true);
    setError(null);
    try {
      const ext  = recordedMime.includes('mp4') ? 'mp4' : 'webm';
      const file = new File([recordedBlob], `gesture.${ext}`, { type: recordedMime });
      const data = await predictVideo(file);
      if (data.success === false) setError(data.error || 'Prediction failed.');
      else setResult(data);
    } catch (err) {
      setError(err.message || 'Backend request failed.');
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <div>
      <PageHeader
        label="Live Recognition"
        title="Real-Time ASL Recognition"
        subtitle="Perform a complete ASL gesture while LightMamba-ASL tracks your hand and upper-body motion."
      />

      <div className="live-layout">
        {/* ── LEFT: Webcam ──────────────────────────── */}
        <div>
          <div
            ref={wrapRef}
            className="webcam-container"
            style={{ position: 'relative', overflow: 'hidden' }}
          >
            {/* Video mirrored via CSS — canvas is NOT mirrored */}
            <video
              ref={videoRef}
              playsInline muted autoPlay
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', transform: 'scaleX(-1)' }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute', top: 0, left: 0,
                width: '100%', height: '100%',
                display: skeletonOn ? 'block' : 'none',
                pointerEvents: 'none',
                transform: 'scaleX(-1)',
              }}
            />

            {!active && (
              <div className="webcam-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                  <circle cx="12" cy="13" r="4"/>
                </svg>
                <p>Click Start Camera to begin</p>
              </div>
            )}

            {active && (
              <>
                <div className="webcam-overlay-badge">
                  <span className="pulse-dot green" />
                  {skeletonOn && mpReady ? 'Skeleton Active' : 'Camera Active'}
                </div>
                <div className="webcam-fps-badge">{fps} FPS</div>
                {recording && (
                  <div style={{ position: 'absolute', bottom: 12, left: 12 }}>
                    <div className="recording-indicator">
                      <div className="recording-dot" />
                      Recording…
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Controls */}
          <div className="live-controls">
            {!active ? (
              <button className="btn btn-primary btn-sm" onClick={handleStartCamera}>
                <CamIcon /> Start Camera
              </button>
            ) : (
              <button className="btn btn-danger btn-sm" onClick={handleStopCamera}>
                <StopIcon /> Stop Camera
              </button>
            )}

            {active && !recording && (
              <button className="btn btn-success btn-sm" onClick={startRecording}>
                <RecordIcon /> Start Recording
              </button>
            )}
            {recording && (
              <button className="btn btn-danger btn-sm" onClick={stopRecording}>
                <StopIcon /> Stop Recording
              </button>
            )}
            {recordedBlob && !recording && (
              <button className="btn btn-primary btn-sm" onClick={handleAnalyze} disabled={analyzing || !online}>
                {analyzing ? <LoadingSpinner size={13} /> : <SendIcon />}
                {analyzing ? 'Analyzing…' : 'Analyze Gesture'}
              </button>
            )}
            <button
              className={`btn btn-sm ${skeletonOn ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setSkeletonOn((s) => !s)}
              disabled={!active}
            >
              {skeletonOn ? 'Hide Skeleton' : 'Show Skeleton'}
            </button>
          </div>

          {camError && <div style={{ marginTop: '10px' }}><ErrorMessage message={camError} /></div>}
          {error    && <div style={{ marginTop: '10px' }}><ErrorMessage message={error} onDismiss={() => setError(null)} /></div>}

          {recordedBlob && !recording && (
            <div style={{ marginTop: '10px', padding: '10px 14px', borderRadius: 'var(--radius-md)', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.25)', fontSize: '0.78rem', color: 'var(--accent-green)' }}>
              Gesture clip ready — {(recordedBlob.size / 1024).toFixed(1)} KB · {recordedMime}
            </div>
          )}
        </div>

        {/* ── RIGHT: Status + Result ─────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

          <div className="tracking-card">
            <div className="tracking-card-title">System Status</div>
            {[
              { label: 'Camera',    value: active    ? 'Active'     : 'Inactive',  ok: active },
              { label: 'MediaPipe', value: mpReady   ? 'Ready'      : 'Loading',   ok: mpReady },
              { label: 'Recording', value: recording ? 'Recording'  : 'Ready',     ok: !recording },
              { label: 'Backend',   value: online    ? 'Online'     : 'Offline',   ok: online },
            ].map(({ label, value, ok }) => (
              <div key={label} className="tracking-item">
                <span className="tracking-item-left">{label}</span>
                <span className={`tracking-item-right ${ok ? 'tracking-detected' : 'tracking-missing'}`}>{value}</span>
              </div>
            ))}
          </div>

          <LandmarkStatus tracking={tracking} />
          <SkeletonLegend />

          {result && <PredictionCard result={result} />}
          <ConfidenceGraph predictions={result?.top_predictions ?? []} isLive />

          {!result && (
            <div className="glass-card" style={{ padding: '16px' }}>
              <div className="section-label" style={{ marginBottom: '8px' }}>Workflow</div>
              {[
                'Start Camera',
                'MediaPipe tracks skeleton live',
                'Click Start Recording',
                'Perform complete ASL gesture',
                'Click Stop Recording',
                'Click Analyze Gesture',
                'View prediction result',
              ].map((s, i) => (
                <div key={s} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '5px 0', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.62rem', fontWeight: 700, color: 'var(--accent-blue)', flexShrink: 0 }}>
                    {i + 1}
                  </span>
                  {s}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
