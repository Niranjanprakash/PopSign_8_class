// Fallback class list — used only when backend is unreachable.
// Primary source is always GET /api/classes
export const FALLBACK_CLASSES = [
  'after', 'airplane', 'bird', 'cloud', 'cry', 'dog', 'drink', 'elephant'
];

export const MAX_FILE_SIZE_MB = 100;
export const ACCEPTED_VIDEO_TYPES = ['.mp4', 'video/mp4'];

export const CONFIDENCE_LEVELS = {
  HIGH:   { min: 0.75, label: 'High Confidence',   color: '#10b981' },
  MEDIUM: { min: 0.50, label: 'Medium Confidence',  color: '#f59e0b' },
  LOW:    { min: 0.00, label: 'Low Confidence',     color: '#ef4444' },
};

export const MEDIAPIPE_WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

// Hand landmark connections (MediaPipe hand topology)
export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // Thumb
  [0,5],[5,6],[6,7],[7,8],       // Index
  [0,9],[9,10],[10,11],[11,12],  // Middle
  [0,13],[13,14],[14,15],[15,16],// Ring
  [0,17],[17,18],[18,19],[19,20],// Pinky
  [5,9],[9,13],[13,17],          // Palm
];

// Upper-body pose connections (MediaPipe pose indices)
export const POSE_UPPER_CONNECTIONS = [
  [11,12], // Shoulders
  [11,13],[13,15], // Left arm
  [12,14],[14,16], // Right arm
  [11,23],[12,24], // Torso sides
  [23,24],         // Hip line
];

export const SKELETON_COLORS = {
  LEFT_HAND:  '#3b82f6',  // Blue
  RIGHT_HAND: '#8b5cf6',  // Purple
  POSE:       '#06b6d4',  // Cyan
};

export const HEALTH_CHECK_INTERVAL_MS = 8000;
