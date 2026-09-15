/**
 * mediapipeService.js
 *
 * Loads @mediapipe/tasks-vision from CDN via a <script> tag so webpack
 * never tries to bundle the WASM module. This eliminates:
 *   - "the request of a dependency is an expression" warning
 *   - missing vision_bundle_mjs.js.map source-map error
 *
 * Smoothing:
 *   - EMA (Exponential Moving Average) on each landmark coordinate.
 *     alpha = 0.6  →  smooth enough for slow gestures, still responsive for fast ones.
 *   - Ghost-frame hold: if detection returns null for up to GHOST_FRAMES frames we
 *     keep the last known landmarks instead of flashing blank, avoiding the flicker
 *     that is very visible during fast movement.
 */

const WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

let handLandmarker = null;
let poseLandmarker  = null;
let initPromise     = null;
let mediaPipeTasksVision = null;

// ─── Smoothing state ────────────────────────────────────────────────────────

/**
 * EMA alpha: how much weight the NEW detection gets.
 *   α closer to 1  →  very responsive, less smooth
 *   α closer to 0  →  very smooth, more lag
 * 0.6 is a good default for webcam hand-tracking.
 */
const EMA_ALPHA = 0.6;

/**
 * How many consecutive "no detection" frames we tolerate before we clear
 * the displayed skeleton. Prevents single-frame flicker during fast moves.
 */
const GHOST_FRAMES = 3;

// Per-landmark-set EMA buffers (array of {x,y,z} or null)
let smoothLeft  = null;
let smoothRight = null;
let smoothPose  = null;

// Ghost-frame counters
let ghostLeft  = 0;
let ghostRight = 0;
let ghostPose  = 0;

/**
 * Apply EMA smoothing to a newly detected landmark array against a previous buffer.
 * @param {Array<{x,y,z}>|null} prev  – smoothed buffer from last frame
 * @param {Array<{x,y,z}>}      next  – raw detection this frame
 * @param {number}              alpha – EMA weight for the new frame
 * @returns {Array<{x,y,z}>}
 */
function applyEMA(prev, next, alpha) {
  if (!prev || prev.length !== next.length) return next.map(lm => ({ ...lm }));
  return next.map((lm, i) => ({
    x: alpha * lm.x + (1 - alpha) * prev[i].x,
    y: alpha * lm.y + (1 - alpha) * prev[i].y,
    z: alpha * lm.z + (1 - alpha) * prev[i].z,
  }));
}

/**
 * Resets all smoothing buffers (call when camera stops).
 */
export function resetSmoothing() {
  smoothLeft  = null;
  smoothRight = null;
  smoothPose  = null;
  ghostLeft   = 0;
  ghostRight  = 0;
  ghostPose   = 0;
}

// ─── MediaPipe init ──────────────────────────────────────────────────────────

/** Dynamically imports the ESM vision bundle from CDN. */
async function loadCDNScript() {
  if (mediaPipeTasksVision) return mediaPipeTasksVision;
  try {
    mediaPipeTasksVision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    return mediaPipeTasksVision;
  } catch (e) {
    throw new Error('Failed to load MediaPipe CDN script: ' + e.message);
  }
}

/**
 * Initialises HandLandmarker and PoseLandmarker once.
 * Safe to call multiple times — returns the same promise.
 *
 * Thresholds are intentionally kept low (0.3–0.4) so fast-moving hands are
 * not dropped by the detector mid-gesture.
 */
export async function initMediaPipe() {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const { FilesetResolver, HandLandmarker, PoseLandmarker } = await loadCDNScript();

    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      // Lowered so fast-moving hands are not lost mid-gesture
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence:  0.3,
      minTrackingConfidence:      0.4,
    });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      minPoseDetectionConfidence: 0.3,
      minPosePresenceConfidence:  0.3,
      minTrackingConfidence:      0.4,
    });
  })();

  return initPromise;
}

// ─── Per-frame detection with smoothing ──────────────────────────────────────

/**
 * Runs hand + pose detection on a single video frame, then applies EMA
 * smoothing and ghost-frame hold before returning landmarks.
 *
 * @param {HTMLVideoElement} frame
 * @param {number} timestampMs
 * @returns {{ leftHand, rightHand, pose, leftCount, rightCount, poseCount }}
 */
export function detectFrame(frame, timestampMs) {
  if (!handLandmarker || !poseLandmarker) {
    return { leftHand: null, rightHand: null, pose: null,
             leftCount: 0, rightCount: 0, poseCount: 0 };
  }

  let rawLeft  = null;
  let rawRight = null;
  let rawPose  = null;

  try {
    const handResult = handLandmarker.detectForVideo(frame, timestampMs);
    if (handResult.landmarks && handResult.handedness) {
      handResult.landmarks.forEach((lms, idx) => {
        const side = handResult.handedness[idx]?.[0]?.categoryName;
        // MediaPipe returns mirrored labels for front-facing camera
        if (side === 'Left')  rawRight = lms;
        if (side === 'Right') rawLeft  = lms;
      });
    }
  } catch (_) { /* frame not ready */ }

  try {
    const poseResult = poseLandmarker.detectForVideo(frame, timestampMs);
    if (poseResult.landmarks?.length > 0) {
      rawPose = poseResult.landmarks[0];
    }
  } catch (_) { /* frame not ready */ }

  // ── EMA smoothing + ghost-frame hold ────────────────────────────────────

  // LEFT HAND
  if (rawLeft) {
    smoothLeft = applyEMA(smoothLeft, rawLeft, EMA_ALPHA);
    ghostLeft  = 0;
  } else {
    ghostLeft++;
    if (ghostLeft > GHOST_FRAMES) smoothLeft = null; // clear after hold window
  }

  // RIGHT HAND
  if (rawRight) {
    smoothRight = applyEMA(smoothRight, rawRight, EMA_ALPHA);
    ghostRight  = 0;
  } else {
    ghostRight++;
    if (ghostRight > GHOST_FRAMES) smoothRight = null;
  }

  // POSE
  if (rawPose) {
    smoothPose = applyEMA(smoothPose, rawPose, EMA_ALPHA);
    ghostPose  = 0;
  } else {
    ghostPose++;
    if (ghostPose > GHOST_FRAMES) smoothPose = null;
  }

  return {
    leftHand:   smoothLeft,
    rightHand:  smoothRight,
    pose:       smoothPose,
    leftCount:  smoothLeft  ? smoothLeft.length  : 0,
    rightCount: smoothRight ? smoothRight.length : 0,
    poseCount:  smoothPose  ? smoothPose.length  : 0,
  };
}

export function isReady() {
  return handLandmarker !== null && poseLandmarker !== null;
}

export async function closeMediaPipe() {
  if (handLandmarker) { handLandmarker.close(); handLandmarker = null; }
  if (poseLandmarker) { poseLandmarker.close();  poseLandmarker  = null; }
  initPromise = null;
  resetSmoothing();
}
