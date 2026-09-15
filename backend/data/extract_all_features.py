"""
Pre-extracts and caches MobileNetV3 RGB features + MediaPipe landmarks for all splits.
Run once before training:
    python -m backend.data.extract_all_features

Saves per-video .pt files to dataset/processed/features/<split>/<video_id>.pt
Each file contains:
    {
        "rgb":       Tensor [32, 576]   - MobileNetV3-Small features
        "landmarks": Tensor [32, 675]   - normalized coords + motion (USE_SECOND_ORDER_MOTION=True)
        "mask":      Tensor [32, 3]     - landmark validity mask
        "label_id":  int
    }
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from backend.config import (
    PROCESSED_DIR, NUM_FRAMES, IMAGE_SIZE, USE_MOTION,
    CLASSES, FEATURE_VERSION
)
from backend.utils import get_device, logger
from backend.data.video_loader import load_video
from backend.features.feature_cache import get_or_extract_features
from backend.features.motion_features import compute_motion_features
from backend.features.rgb_extractor import MobileNetV3FeatureExtractor


def extract_all(splits=("train", "val", "test")):
    device = get_device()

    # Load MobileNetV3 extractor once
    extractor = MobileNetV3FeatureExtractor(pretrained=True).to(device)
    extractor.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for split in splits:
        csv_path = PROCESSED_DIR / "splits" / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"[EXTRACT] {csv_path} not found — skipping {split}")
            continue

        out_dir = PROCESSED_DIR / "features" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        logger.info(f"[EXTRACT] Processing {split}: {len(df)} videos → {out_dir}")

        skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
            video_id  = str(row["video_id"])
            video_path = str(row["video_path"])
            label_id  = int(row["label_id"])
            out_path  = out_dir / f"{video_id}.pt"

            if out_path.exists():
                try:
                    cached = torch.load(out_path, map_location="cpu", weights_only=False)
                    if cached.get("feature_version") == FEATURE_VERSION:
                        skipped += 1
                        continue
                except Exception:
                    pass

            try:
                # 1. Load frames
                frames = load_video(video_path, target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)

                # 2. RGB features [32, 576]
                rgb_tensors = []
                for frame in frames:
                    ft = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    ft = normalize(ft)
                    rgb_tensors.append(ft)
                rgb_batch = torch.stack(rgb_tensors).to(device)  # [32, 3, 224, 224]
                with torch.no_grad():
                    rgb_feats = extractor(rgb_batch).cpu()        # [32, 960]

                # 3. Landmarks + motion
                lm_data    = get_or_extract_features(video_id, frames)
                raw_lm     = np.array(lm_data["landmarks"], dtype=np.float32)  # [32, 225]
                raw_mask   = np.array(lm_data["mask"],      dtype=np.float32)  # [32, 3]
                motion     = np.array(lm_data["motion"],    dtype=np.float32)  # [32, 225|450]

                if USE_MOTION:
                    lm_features = torch.tensor(
                        np.concatenate([raw_lm, motion], axis=-1), dtype=torch.float32
                    )  # [32, 675]
                else:
                    lm_features = torch.tensor(raw_lm, dtype=torch.float32)  # [32, 225]

                torch.save({
                    "rgb":      rgb_feats,
                    "landmarks": lm_features,
                    "mask":     torch.tensor(raw_mask, dtype=torch.float32),
                    "label_id": label_id,
                    "feature_version": FEATURE_VERSION,
                }, out_path)

            except Exception as e:
                logger.warning(f"[EXTRACT] Failed {video_id}: {e}")

        logger.info(f"[EXTRACT] {split} done — {len(df) - skipped} extracted, {skipped} already cached")

    logger.info("[EXTRACT] All splits complete. Ready for fast training.")


if __name__ == "__main__":
    extract_all()
