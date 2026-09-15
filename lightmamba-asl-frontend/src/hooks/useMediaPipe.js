import { useState, useRef, useCallback, useEffect } from 'react';
import { initMediaPipe, detectFrame, isReady, resetSmoothing } from '../services/mediapipeService';
import { drawSkeleton, syncCanvasToVideo, getContainedVideoRect, clearCanvas } from '../utils/canvasUtils';

/**
 * useMediaPipe — non-blocking detection loop
 *
 * Design:
 *  • requestAnimationFrame drives the DRAW loop at ~60 fps.
 *  • MediaPipe detection runs inside that loop but is guarded by an
 *    `isProcessing` flag so we never queue a second detect call while
 *    the first one is still running (avoids stacking microtasks and
 *    causing timestamp-order errors in VIDEO mode).
 *  • While detection is in-flight the canvas is re-drawn with the
 *    LAST known result (already EMA-smoothed in mediapipeService) so
 *    the skeleton keeps moving at display frame-rate even when detection
 *    takes longer than one frame.
 */
export function useMediaPipe(videoRef, canvasRef, enabled) {
  const [mpReady, setMpReady]   = useState(false);
  const [mpError, setMpError]   = useState(null);
  const [tracking, setTracking] = useState({
    leftHand: null, rightHand: null, pose: null,
    leftCount: 0, rightCount: 0, poseCount: 0,
  });

  const rafRef          = useRef(null);
  const runningRef      = useRef(false);
  const isProcessingRef = useRef(false);          // guard: only 1 detect in-flight
  const lastResultRef   = useRef(null);           // last smoothed result for draw

  // Initialize MediaPipe once
  useEffect(() => {
    let cancelled = false;
    initMediaPipe()
      .then(() => { if (!cancelled) setMpReady(true); })
      .catch((e) => { if (!cancelled) setMpError(`MediaPipe init failed: ${e.message}`); });
    return () => { cancelled = true; };
  }, []);

  const processLoop = useCallback(() => {
    if (!runningRef.current) return;

    const video  = videoRef.current;
    const canvas = canvasRef.current;

    if (
      video &&
      canvas &&
      isReady() &&
      !video.paused &&
      !video.ended &&
      video.readyState >= 2
    ) {
      syncCanvasToVideo(canvas, video);
      const drawRect = getContainedVideoRect(canvas, video);

      if (!isProcessingRef.current) {
        // ── Kick off detection (synchronous MediaPipe call wrapped so we
        //    can set the flag safely around it) ──────────────────────────
        isProcessingRef.current = true;
        try {
          const ts     = performance.now();
          const result = detectFrame(video, ts);
          lastResultRef.current = result;
          setTracking(result);
        } catch (_) {
          // frame not ready — keep last result
        } finally {
          isProcessingRef.current = false;
        }
      }

      // Always draw at RAF rate using the latest available (smoothed) result
      drawSkeleton(canvas, lastResultRef.current, drawRect);
    }

    rafRef.current = requestAnimationFrame(processLoop);
  }, [videoRef, canvasRef]);

  const startTracking = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;
    rafRef.current = requestAnimationFrame(processLoop);
  }, [processLoop]);

  const stopTracking = useCallback(() => {
    runningRef.current      = false;
    isProcessingRef.current = false;
    lastResultRef.current   = null;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (canvasRef.current) clearCanvas(canvasRef.current);
    resetSmoothing();
    setTracking({ leftHand: null, rightHand: null, pose: null,
                  leftCount: 0, rightCount: 0, poseCount: 0 });
  }, [canvasRef]);

  // Auto-start/stop based on enabled flag
  useEffect(() => {
    if (enabled && mpReady) {
      startTracking();
    } else {
      stopTracking();
    }
    return () => stopTracking();
  }, [enabled, mpReady, startTracking, stopTracking]);

  return { mpReady, mpError, tracking, startTracking, stopTracking };
}
