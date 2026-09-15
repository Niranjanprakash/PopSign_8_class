import cv2
import numpy as np
from backend.data.frame_sampler import get_sampled_indices

def load_video(video_path: str, target_frames: int = 32, image_size: int = 224) -> np.ndarray:
    """
    Decodes a video file (mp4/webm), extracts target_frames uniformly,
    resizes to (image_size, image_size), and returns [T, H, W, C] in RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    # Read all frames first — CAP_PROP_FRAME_COUNT is unreliable for webm
    all_frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        return np.zeros((target_frames, image_size, image_size, 3), dtype=np.uint8)

    sampled_indices = get_sampled_indices(len(all_frames), target_frames)

    frames = []
    for idx in sampled_indices:
        actual_idx = min(idx, len(all_frames) - 1)
        frame_rgb = cv2.cvtColor(all_frames[actual_idx], cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, (image_size, image_size)))

    return np.stack(frames, axis=0)  # [T, H, W, C]
