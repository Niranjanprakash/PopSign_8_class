import React, { useRef, useState, useCallback, useEffect } from 'react';
import PageHeader    from '../components/PageHeader';
import { LandmarkStatus, SkeletonLegend } from '../components/LandmarkStatus';
import ConfidenceGraph from '../components/ConfidenceGraph';
import ErrorMessage  from '../components/ErrorMessage';
import LoadingSpinner from '../components/LoadingSpinner';
import { useCamera }  from '../hooks/useCamera';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { predictVideo } from '../services/api';
import { useBackendStatus } from '../hooks/useBackendStatus';

const RecordIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="currentColor"/>
  </svg>
);
const StopIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
  </svg>
);
const CamIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);
const SpeakIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
    <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
  </svg>
);
const TrashIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/>
    <path d="M9 6V4h6v2"/>
  </svg>
);
const UndoIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/>
  </svg>
);
const AddIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);

export default function SentenceBuilder() {
  const videoRef         = useRef(null);
  const canvasRef        = useRef(null);
  const wrapRef          = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);

  const [recording, setRecording]       = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedMime, setRecordedMime] = useState('');
  const [analyzing, setAnalyzing]       = useState(false);
  const [lastResult, setLastResult]     = useState(null);
  const [error, setError]               = useState(null);
  const [words, setWords]               = useState([]);
  const [speaking, setSpeaking]         = useState(false);
  const [skeletonOn, setSkeletonOn]     = useState(true);

  const { active, error: camError, startCamera, stopCamera } = useCamera();
  const { tracking, mpReady } = useMediaPipe(videoRef, canvasRef, active && skeletonOn);
  const { online } = useBackendStatus();

  /* Canvas size sync */
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width  = wrap.offsetWidth;
      canvas.height = wrap.offsetHeight;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  /* Stop speech on unmount */
  useEffect(() => () => window.speechSynthesis?.cancel(), []);

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
    mr.onstop = () => setRecordedBlob(new Blob(chunksRef.current, { type: mimeType }));
    mr.start(100);
    mediaRecorderRef.current = mr;
    setRecording(true);
    setLastResult(null);
    setError(null);
  }

  function stopRecording() {
    if (mediaRecorderRef.current?.state !== 'inactive') {
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
      if (data.success === false) {
        setError(data.error || 'Prediction failed.');
      } else {
        setLastResult(data);
        // Auto-add if confident and not uncertain
        if (!data.uncertain && data.prediction !== 'UNCERTAIN') {
          setWords(prev => [...prev, { word: data.prediction, confidence: data.confidence, id: Date.now() }]);
        }
      }
    } catch (err) {
      setError(err.message || 'Backend request failed.');
    } finally {
      setAnalyzing(false);
      setRecordedBlob(null);
    }
  }

  function handleSpeak() {
    if (!words.length || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const sentence = words.map(w => w.word).join(' ');
    const utt = new SpeechSynthesisUtterance(sentence);
    utt.rate = 0.85;
    utt.onstart = () => setSpeaking(true);
    utt.onend   = () => setSpeaking(false);
    utt.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utt);
  }

  function handleUndo() {
    setWords(prev => prev.slice(0, -1));
  }

  function handleClear() {
    window.speechSynthesis?.cancel();
    setWords([]);
    setLastResult(null);
    setSpeaking(false);
  }

  function handleAddManual(word) {
    setWords(prev => [...prev, { word, confidence: 1, id: Date.now() }]);
  }

  const sentence = words.map(w => w.word).join(' ');

  return (
    <div>
      <PageHeader
        label="Sentence Builder"
        title="Build ASL Sentences"
        subtitle="Sign each word one at a time — detected signs are added to your sentence automatically."
      />

      <div className="sb-layout">
        {/* ── LEFT: Webcam ── */}
        <div>
          <div ref={wrapRef} className="webcam-container" style={{ position: 'relative', overflow: 'hidden' }}>
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
              <button className="btn btn-primary btn-sm" onClick={handleStartCamera}><CamIcon /> Start Camera</button>
            ) : (
              <button className="btn btn-danger btn-sm" onClick={handleStopCamera}><StopIcon /> Stop Camera</button>
            )}
            {active && !recording && (
              <button className="btn btn-success btn-sm" onClick={startRecording} disabled={analyzing}><RecordIcon /> Record Sign</button>
            )}
            {recording && (
              <button className="btn btn-danger btn-sm" onClick={stopRecording}><StopIcon /> Stop</button>
            )}
            {recordedBlob && !recording && (
              <button className="btn btn-primary btn-sm" onClick={handleAnalyze} disabled={analyzing || !online}>
                {analyzing ? <LoadingSpinner size={13} /> : <AddIcon />}
                {analyzing ? 'Detecting…' : 'Add to Sentence'}
              </button>
            )}
            <button
              className={`btn btn-sm ${skeletonOn ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setSkeletonOn(s => !s)}
              disabled={!active}
            >
              {skeletonOn ? 'Hide Skeleton' : 'Show Skeleton'}
            </button>
          </div>

          {camError && <div style={{ marginTop: '10px' }}><ErrorMessage message={camError} /></div>}
          {error    && <div style={{ marginTop: '10px' }}><ErrorMessage message={error} onDismiss={() => setError(null)} /></div>}

          {/* Last detection result */}
          {lastResult && (
            <div className={`sb-last-result ${lastResult.uncertain ? 'uncertain' : 'confident'}`}>
              <span className="sb-last-label">Last detected:</span>
              <span className="sb-last-word">{lastResult.prediction?.toUpperCase()}</span>
              <span className="sb-last-conf">{Math.round(lastResult.confidence * 100)}%</span>
              {lastResult.uncertain && (
                <span className="sb-uncertain-badge">Low confidence — not added</span>
              )}
              {!lastResult.uncertain && (
                <span className="sb-added-badge">✓ Added</span>
              )}
            </div>
          )}

          <LandmarkStatus tracking={tracking} />
          <SkeletonLegend />
          <ConfidenceGraph predictions={lastResult?.top_predictions ?? []} isLive />
        </div>

        {/* ── RIGHT: Sentence Panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* Sentence display */}
          <div className="sb-sentence-card">
            <div className="sb-sentence-header">
              <span className="section-label">Sentence</span>
              <span className="sb-word-count">{words.length} word{words.length !== 1 ? 's' : ''}</span>
            </div>

            <div className="sb-sentence-display">
              {words.length === 0 ? (
                <span className="sb-placeholder">Start signing to build your sentence…</span>
              ) : (
                <span className="sb-sentence-text">{sentence}</span>
              )}
            </div>

            {/* Word chips */}
            {words.length > 0 && (
              <div className="sb-chips">
                {words.map((w, i) => (
                  <div key={w.id} className="sb-chip">
                    <span className="sb-chip-index">{i + 1}</span>
                    <span className="sb-chip-word">{w.word}</span>
                    <span className="sb-chip-conf">{Math.round(w.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Actions */}
            <div className="sb-actions">
              <button
                className="btn btn-primary btn-sm"
                onClick={handleSpeak}
                disabled={!words.length || speaking}
              >
                {speaking ? <LoadingSpinner size={13} /> : <SpeakIcon />}
                {speaking ? 'Speaking…' : 'Speak'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={handleUndo} disabled={!words.length}>
                <UndoIcon /> Undo
              </button>
              <button className="btn btn-danger btn-sm" onClick={handleClear} disabled={!words.length}>
                <TrashIcon /> Clear
              </button>
            </div>
          </div>

          {/* Quick add supported signs */}
          <div className="glass-card" style={{ padding: '16px' }}>
            <div className="section-label" style={{ marginBottom: '10px' }}>Quick Add (Supported Signs)</div>
            <div className="sb-quick-add">
              {['after','airplane','bird','cloud','cry','dog','drink','elephant'].map(sign => (
                <button key={sign} className="sb-quick-btn" onClick={() => handleAddManual(sign)}>
                  {sign}
                </button>
              ))}
            </div>
          </div>

          {/* System status */}
          <div className="tracking-card">
            <div className="tracking-card-title">System Status</div>
            {[
              { label: 'Camera',    value: active    ? 'Active'    : 'Inactive', ok: active },
              { label: 'MediaPipe', value: mpReady   ? 'Ready'     : 'Loading',  ok: mpReady },
              { label: 'Backend',   value: online    ? 'Online'    : 'Offline',  ok: online },
              { label: 'Words',     value: `${words.length} added`,              ok: words.length > 0 },
            ].map(({ label, value, ok }) => (
              <div key={label} className="tracking-item">
                <span className="tracking-item-left">{label}</span>
                <span className={`tracking-item-right ${ok ? 'tracking-detected' : 'tracking-missing'}`}>{value}</span>
              </div>
            ))}
          </div>

        </div>
      </div>
    </div>
  );
}
