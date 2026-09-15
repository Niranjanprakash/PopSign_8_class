import cv2
import torch
import numpy as np
import time
from collections import deque
from torchvision import transforms

from backend.config import CHECKPOINT_DIR, CLASSES, NUM_FRAMES, IMAGE_SIZE, USE_MOTION
from backend.utils import get_device, logger
from backend.features.landmark_extractor import LandmarkExtractor
from backend.features.landmark_normalizer import normalize_landmarks
from backend.features.motion_features import compute_motion_features
from backend.models.lightmamba_asl import LightMambaASL
from backend.inference.temporal_smoothing import TemporalSmoothing
from backend.inference.confidence import calibrate_prediction, get_ambiguity_warning

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_model_inputs(frame_buffer, landmark_extractor, device):
    """
    Shared preprocessing pipeline — identical to predict_video.py.
    frame_buffer: iterable of [H, W, 3] uint8 RGB frames (length == NUM_FRAMES)
    Returns (rgb_tensor, landmark_features, mask_tensor) ready for model.forward()
    """
    frames_np = np.stack(list(frame_buffer), axis=0)  # [T, H, W, 3]

    # --- Resize all frames to IMAGE_SIZE x IMAGE_SIZE (matches training pipeline) ---
    resized = np.stack(
        [cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE)) for f in frames_np], axis=0
    )

    # --- RGB ---
    rgb_tensors = []
    for f in resized:
        f_t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        rgb_tensors.append(_NORMALIZE(f_t))
    rgb_tensor = torch.stack(rgb_tensors, dim=0).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    # --- Landmarks ---
    raw_landmarks, mask = landmark_extractor.extract_video_sequence(resized)
    normalized = normalize_landmarks(raw_landmarks, mask)
    motion = compute_motion_features(normalized, mask)  # mask-aware

    flat_landmarks = normalized.reshape(normalized.shape[0], -1)  # [T, 225]
    if USE_MOTION:
        landmark_features = torch.tensor(
            np.concatenate([flat_landmarks, motion], axis=-1), dtype=torch.float32
        )
    else:
        landmark_features = torch.tensor(flat_landmarks, dtype=torch.float32)

    landmark_features = landmark_features.unsqueeze(0).to(device)       # [1, T, L_DIM]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, 3]

    return rgb_tensor, landmark_features, mask_tensor


def main():
    device = get_device()
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    print("[WEBCAM] Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_mapping = checkpoint.get("class_mapping", CLASSES)
    model = LightMambaASL(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    landmark_extractor = LandmarkExtractor()
    smoothing = TemporalSmoothing(window_size=5)

    frame_buffer = deque(maxlen=NUM_FRAMES)
    PREDICT_EVERY_N_FRAMES = NUM_FRAMES // 2
    frames_since_last_pred = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[WEBCAM] Starting webcam feed. Press 'q' to quit, 'c' to clear buffer.")

    pred_label = "Waiting..."
    confidence = 0.0
    latency_ms = 0.0
    fps = 0.0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Store full-res frame; build_model_inputs handles resize for landmarks
        frame_buffer.append(frame_rgb)
        frames_since_last_pred += 1

        if len(frame_buffer) == NUM_FRAMES and frames_since_last_pred >= PREDICT_EVERY_N_FRAMES:
            inf_start = time.time()
            try:
                rgb_t, lm_t, mask_t = build_model_inputs(frame_buffer, landmark_extractor, device)
                with torch.no_grad():
                    logits = model(rgb_t, lm_t, mask_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                smoothed = smoothing.update(probs)
                pred_idx, confidence, top_k = calibrate_prediction(torch.tensor(smoothed).unsqueeze(0))
                ambiguity = get_ambiguity_warning(top_k, class_mapping)
                pred_label = "UNCERTAIN" if pred_idx == -1 or ambiguity else class_mapping[pred_idx]
            except Exception as e:
                logger.warning(f"[WEBCAM] Inference error: {e}")
            latency_ms = (time.time() - inf_start) * 1000
            frames_since_last_pred = 0

        fps = 1.0 / max(time.time() - start_time, 1e-6)

        cv2.rectangle(frame, (10, 10), (320, 140), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {pred_label}",      (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}",                (20, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms",  (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("LightMamba-ASL Webcam Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame_buffer.clear()
            smoothing.clear()
            pred_label = "Waiting..."
            confidence = 0.0
            frames_since_last_pred = 0

    cap.release()
    cv2.destroyAllWindows()
    landmark_extractor.close()
    print("[WEBCAM] Stream terminated.")


if __name__ == "__main__":
    main()
