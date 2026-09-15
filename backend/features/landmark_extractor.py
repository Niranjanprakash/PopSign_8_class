import numpy as np
import mediapipe as mp
from urllib.request import urlretrieve

from backend.config import HOLISTIC_LANDMARKER_MODEL_PATH, HOLISTIC_LANDMARKER_MODEL_URL

class LandmarkExtractor:
    """
    Extracts Left Hand, Right Hand, and Pose landmarks using MediaPipe Holistic.
    Returns:
        - raw_landmarks: list or array of shape [T, 75, 3]
        - mask: list or array of shape [T, 3] representing visibility/presence of (left_hand, right_hand, pose)
    """
    def __init__(self):
        self.mode = None
        self.holistic = None

        # Prefer the legacy API when it exists because it is fast and matches the
        # original preprocessing pipeline. Newer MediaPipe builds ship only the
        # Tasks API, so we fall back to that with an explicit model file.
        try:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mode = "legacy"
            print("[LANDMARK] Using MediaPipe legacy Holistic solution.")
            return
        except AttributeError:
            print("[LANDMARK] Legacy MediaPipe solutions API is unavailable. Switching to Tasks API.")

        self._ensure_task_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(HOLISTIC_LANDMARKER_MODEL_PATH))
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self.holistic = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
        self.mode = "tasks"
        print(f"[LANDMARK] Using MediaPipe Tasks HolisticLandmarker from {HOLISTIC_LANDMARKER_MODEL_PATH}.")

    def _ensure_task_model(self):
        if HOLISTIC_LANDMARKER_MODEL_PATH.exists() and HOLISTIC_LANDMARKER_MODEL_PATH.stat().st_size > 0:
            return

        HOLISTIC_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[LANDMARK] Downloading HolisticLandmarker model from {HOLISTIC_LANDMARKER_MODEL_URL}...")
        try:
            urlretrieve(HOLISTIC_LANDMARKER_MODEL_URL, HOLISTIC_LANDMARKER_MODEL_PATH)
        except Exception as exc:
            raise FileNotFoundError(
                "MediaPipe HolisticLandmarker model is missing and could not be downloaded. "
                "Please check network access or place the .task file at "
                f"{HOLISTIC_LANDMARKER_MODEL_PATH}."
            ) from exc

    def _extract_with_tasks(self, frame_rgb: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.holistic.detect(mp_image)

    def extract_from_frame(self, frame_rgb: np.ndarray):
        if self.mode == "tasks":
            results = self._extract_with_tasks(frame_rgb)
        else:
            results = self.holistic.process(frame_rgb)

        def landmarks_from(result, name):
            """Normalize legacy protobuf and Tasks API landmark containers."""
            value = getattr(result, name, None)
            if value is None:
                return None
            return getattr(value, "landmark", value)
        
        # Initialize empty coords
        lh_coords = np.zeros((21, 3), dtype=np.float32)
        rh_coords = np.zeros((21, 3), dtype=np.float32)
        pose_coords = np.zeros((33, 3), dtype=np.float32)
        
        # Validity mask: [left_hand_detected, right_hand_detected, pose_detected]
        mask = np.zeros(3, dtype=np.float32)
        
        left_hand = landmarks_from(results, "left_hand_landmarks")
        if left_hand:
            mask[0] = 1.0
            for i, lm in enumerate(left_hand):
                lh_coords[i] = [lm.x, lm.y, lm.z]
                
        right_hand = landmarks_from(results, "right_hand_landmarks")
        if right_hand:
            mask[1] = 1.0
            for i, lm in enumerate(right_hand):
                rh_coords[i] = [lm.x, lm.y, lm.z]
                
        pose = landmarks_from(results, "pose_landmarks")
        if pose:
            mask[2] = 1.0
            for i, lm in enumerate(pose):
                pose_coords[i] = [lm.x, lm.y, lm.z]

        # Combine all landmarks: [75, 3]
        combined = np.concatenate([lh_coords, rh_coords, pose_coords], axis=0)
        return combined, mask

    def extract_video_sequence(self, frames: np.ndarray):
        """
        Frames input shape: [T, H, W, C]
        Returns:
            - sequence_landmarks: [T, 75, 3]
            - sequence_masks: [T, 3]
        """
        seq_landmarks = []
        seq_masks = []
        
        for frame in frames:
            landmarks, mask = self.extract_from_frame(frame)
            seq_landmarks.append(landmarks)
            seq_masks.append(mask)
            
        return np.stack(seq_landmarks, axis=0), np.stack(seq_masks, axis=0)

    def close(self):
        if self.holistic is not None and hasattr(self.holistic, "close"):
            self.holistic.close()
