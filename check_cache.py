import torch
import numpy as np
from pathlib import Path
from backend.data.video_loader import load_video
from backend.features.landmark_extractor import LandmarkExtractor
from backend.features.landmark_normalizer import normalize_landmarks
from backend.features.motion_features import compute_motion_features
from backend.config import NUM_FRAMES, IMAGE_SIZE

# Pick one test video
test_video = next(Path("dataset/popsign/test/bird").glob("*.mp4"))
video_id   = test_video.stem

print(f"Video: {test_video.name}")

# 1. Load cached features
cache_path = Path(f"dataset/processed/features/test/{video_id}.pt")
if cache_path.exists():
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    print(f"\nCached landmarks mean: {cached['landmarks'].abs().mean():.4f}")
    print(f"Cached mask sum: {cached['mask'].sum():.1f}")
    print(f"Cached lm[0,:10]: {cached['landmarks'][0,:10].tolist()}")
else:
    print("No cache found!")

# 2. Live extraction
frames = load_video(str(test_video), target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
extractor = LandmarkExtractor()
raw_lm, mask = extractor.extract_video_sequence(frames)
norm_lm = normalize_landmarks(raw_lm, mask)
motion  = compute_motion_features(norm_lm, mask)
flat_lm = norm_lm.reshape(NUM_FRAMES, -1)
live_lm = np.concatenate([flat_lm, motion], axis=-1)

print(f"\nLive landmarks mean: {np.abs(live_lm).mean():.4f}")
print(f"Live mask sum: {mask.sum():.1f}")
print(f"Live lm[0,:10]: {live_lm[0,:10].tolist()}")
