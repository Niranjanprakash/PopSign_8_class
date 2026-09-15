import { HAND_CONNECTIONS, POSE_UPPER_CONNECTIONS, SKELETON_COLORS } from './constants';

/**
 * Clears the canvas completely.
 */
export function clearCanvas(canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Draws a single set of landmarks and their connections.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} landmarks  - array of {x, y} normalized [0..1]
 * @param {Array} connections - array of [i, j] index pairs
 * @param {string} color
 * @param {number} dotRadius
 * @param {number} lineWidth
 */
function drawLandmarkSet(ctx, landmarks, connections, color, drawRect, dotRadius = 4, lineWidth = 2) {
  if (!landmarks || landmarks.length === 0) return;

  const { x, y, width, height } = drawRect;

  // Draw connections
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = 0.75;
  for (const [i, j] of connections) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(x + a.x * width, y + a.y * height);
    ctx.lineTo(x + b.x * width, y + b.y * height);
    ctx.stroke();
  }

  // Draw dots
  ctx.globalAlpha = 1.0;
  for (const lm of landmarks) {
    if (!lm) continue;
    ctx.beginPath();
    ctx.arc(x + lm.x * width, y + lm.y * height, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    // White center
    ctx.beginPath();
    ctx.arc(x + lm.x * width, y + lm.y * height, dotRadius * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fill();
  }
}

/**
 * Draws the full skeleton from a GestureRecognizer / HandLandmarker result.
 * @param {HTMLCanvasElement} canvas
 * @param {object} result - { leftHand, rightHand, pose }
 *   leftHand / rightHand: array of {x,y,z} normalized landmarks or null
 *   pose: array of {x,y,z} normalized landmarks or null
 */
export function drawSkeleton(canvas, result, contentRect = null) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  clearCanvas(canvas);

  if (!result) return;

  const { leftHand, rightHand, pose } = result;
  const drawRect = contentRect || {
    x: 0,
    y: 0,
    width: canvas.width,
    height: canvas.height,
  };

  if (leftHand) {
    drawLandmarkSet(ctx, leftHand, HAND_CONNECTIONS, SKELETON_COLORS.LEFT_HAND, drawRect, 4, 2);
  }

  if (rightHand) {
    drawLandmarkSet(ctx, rightHand, HAND_CONNECTIONS, SKELETON_COLORS.RIGHT_HAND, drawRect, 4, 2);
  }

  if (pose) {
    // Only upper-body joints
    const upperIndices = new Set(POSE_UPPER_CONNECTIONS.flat());
    const upperPose = pose.map((lm, i) => upperIndices.has(i) ? lm : null);
    drawLandmarkSet(ctx, upperPose, POSE_UPPER_CONNECTIONS, SKELETON_COLORS.POSE, drawRect, 5, 2.5);
  }

  ctx.globalAlpha = 1.0;
}

/**
 * Syncs canvas intrinsic pixel dimensions to the video's rendered size.
 * Uses offsetWidth/offsetHeight so it works even when transform:scaleX(-1) is applied.
 */
export function syncCanvasToVideo(canvas, videoEl) {
  if (!canvas || !videoEl) return;
  const w = videoEl.offsetWidth;
  const h = videoEl.offsetHeight;
  if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
    canvas.width  = w;
    canvas.height = h;
  }
}

/**
 * Returns the actual visible video rectangle inside a canvas when the video
 * uses CSS `object-fit: contain`. Portrait videos have left/right black bars;
 * drawing normalized landmarks against the full wrapper would shift them.
 */
export function getContainedVideoRect(canvas, videoEl) {
  const canvasWidth = canvas?.width || 0;
  const canvasHeight = canvas?.height || 0;
  const videoWidth = videoEl?.videoWidth || canvasWidth;
  const videoHeight = videoEl?.videoHeight || canvasHeight;

  if (!canvasWidth || !canvasHeight || !videoWidth || !videoHeight) {
    return { x: 0, y: 0, width: canvasWidth, height: canvasHeight };
  }

  const scale = Math.min(canvasWidth / videoWidth, canvasHeight / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;

  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
}
