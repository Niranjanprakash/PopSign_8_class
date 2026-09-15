import re
import pickle
import hashlib
import numpy as np
from pathlib import Path
from backend.config import PROCESSED_DIR, NUM_FRAMES, IMAGE_SIZE
from backend.features.landmark_extractor import LandmarkExtractor
from backend.features.landmark_normalizer import normalize_landmarks
from backend.features.motion_features import compute_motion_features

_extractor = None

# Config fingerprint: cache is invalidated when preprocessing or extractor version changes.
_CONFIG_TAG = f"v3_f{NUM_FRAMES}_s{IMAGE_SIZE}_holistic"
_SAFE_ID_RE = re.compile(r'[^A-Za-z0-9_\-]')

def _safe_video_id(video_id: str) -> str:
    """Sanitize video_id to prevent path traversal attacks."""
    sanitized = _SAFE_ID_RE.sub('_', video_id)
    # Hard limit to avoid excessively long filenames
    return sanitized[:64]

def get_landmark_extractor():
    global _extractor
    if _extractor is None:
        _extractor = LandmarkExtractor()
    return _extractor

def get_or_extract_features(video_id: str, frames: np.ndarray = None) -> dict:
    """
    Cache path: dataset/processed/landmarks/<safe_id>_<config_tag>.pkl
    Sanitizes video_id and embeds config tag so stale caches are never reused.
    """
    cache_dir = PROCESSED_DIR / "landmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_id = _safe_video_id(video_id)
    cache_path = cache_dir / f"{safe_id}_{_CONFIG_TAG}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[CACHE] Error reading cache for {video_id}, re-extracting: {e}")

    if frames is None:
        raise ValueError(f"Features not cached for {video_id} and frames not provided!")

    extractor = get_landmark_extractor()
    raw_landmarks, mask = extractor.extract_video_sequence(frames)

    normalized = normalize_landmarks(raw_landmarks, mask)
    motion = compute_motion_features(normalized, mask)

    T, N, D = normalized.shape
    flat_landmarks = normalized.reshape(T, N * D)

    data = {
        "landmarks": flat_landmarks,  # [T, 225]
        "mask":      mask,             # [T, 3]
        "motion":    motion            # [T, 225] or [T, 450]
    }

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[CACHE] Error writing cache for {video_id}: {e}")

    return data
