import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from backend.config import NUM_FRAMES, IMAGE_SIZE, USE_MOTION, USE_FEATURE_CACHE, PROCESSED_DIR, FEATURE_VERSION, CLASSES, OVERSAMPLING_WEIGHTS

# Classes that get extra-strong augmentation due to low sample count
_WEAK_CLASSES = {"dog", "airplane"}
from backend.data.video_loader import load_video
from backend.data.augmentations import VideoAugmentation, apply_landmark_augmentation
from backend.features.feature_cache import get_or_extract_features
from backend.features.motion_features import compute_motion_features


class ASLVideoDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, is_training: bool = False, use_horizontal_flip: bool = False):
        self.df           = pd.read_csv(csv_path)
        self.is_training  = is_training
        self.split        = Path(csv_path).stem
        self.is_merged    = (self.split == "train_val_merged")
        self.feature_dir  = PROCESSED_DIR / "features" / self.split
        self.use_cache    = USE_FEATURE_CACHE
        self.normalize    = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augment      = VideoAugmentation(use_horizontal_flip=use_horizontal_flip) if is_training else None

        if self.use_cache and not self.is_merged:
            pt_files = list(self.feature_dir.glob("*.pt"))
            if not pt_files:
                self.use_cache = False

    def get_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler — oversamples weak classes (dog, airplane)."""
        sample_weights = [
            OVERSAMPLING_WEIGHTS.get(CLASSES[int(row["label_id"])], 1.0)
            for _, row in self.df.iterrows()
        ]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def __len__(self):
        return len(self.df)

    def _get_feature_dir(self, row) -> Path:
        """For merged dataset, find feature in original split folder."""
        if self.is_merged:
            orig_split = str(row.get("split", "train"))
            return PROCESSED_DIR / "features" / orig_split
        return self.feature_dir

    def _load_from_cache(self, video_id: str, label_id: int, feature_dir: Path) -> dict:
        """Fast path: load pre-extracted .pt features directly."""
        pt_path = feature_dir / f"{video_id}.pt"
        data    = torch.load(pt_path, map_location="cpu", weights_only=True)

        if data.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Stale feature cache: {pt_path.name}. Expected {FEATURE_VERSION!r}; "
                "run `python -m backend.data.extract_all_features`."
            )

        rgb_feats   = data["rgb"]        # [32, 576]
        lm_features = data["landmarks"]  # [32, 225] or [32, 675]
        mask        = data["mask"]       # [32, 3]

        # Lightweight landmark jitter augmentation on tensors (no video decode needed)
        if self.is_training and self.augment:
            landmark_coords = lm_features[:, :225].clone()
            if self.augment.use_horizontal_flip and torch.rand(1).item() > 0.5:
                # Swap left/right hand blocks and mirror normalized x coordinates.
                lm_np   = landmark_coords.numpy()
                mask_np = mask.numpy()
                lm_3d   = lm_np[:, :225].reshape(-1, 75, 3)
                # Cached coordinates are already wrist/shoulder centered, so a
                # horizontal flip negates x rather than applying x = 1 - x.
                lh = lm_3d[:, 0:21].copy()
                rh = lm_3d[:, 21:42].copy()
                lm_3d[:, 0:21] = rh
                lm_3d[:, 21:42] = lh
                lm_3d[:, :, 0] *= -1.0
                left_valid = mask_np[:, 0].copy()
                mask_np[:, 0] = mask_np[:, 1]
                mask_np[:, 1] = left_valid
                lm_np[:, :225] = lm_3d.reshape(-1, 225)
                landmark_coords = torch.tensor(lm_np, dtype=torch.float32)
                mask        = torch.tensor(mask_np, dtype=torch.float32)

            # Small Gaussian jitter on landmark coords — stronger for weak classes
            jitter_std = 0.012 if CLASSES[label_id] in _WEAK_CLASSES else 0.005
            landmark_coords = landmark_coords + torch.randn_like(landmark_coords) * jitter_std

        if USE_MOTION:
            coords_np = landmark_coords.numpy().reshape(-1, 75, 3)
            motion = compute_motion_features(coords_np, mask.numpy())
            lm_features = torch.cat([
                landmark_coords,
                torch.tensor(motion, dtype=torch.float32),
            ], dim=-1)
        else:
            lm_features = landmark_coords

        return {
            "video_id":  video_id,
            "rgb":       rgb_feats,
            "landmarks": lm_features,
            "mask":      mask,
            "label":     label_id,
        }

    def _load_from_video(self, row) -> dict:
        """Slow path: decode raw .mp4 and extract features on-the-fly."""
        video_path = row["video_path"]
        video_id   = str(row["video_id"])
        label_id   = int(row["label_id"])

        frames = load_video(video_path, target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)

        landmark_data = get_or_extract_features(video_id, frames)
        raw_landmarks = np.array(landmark_data["landmarks"], dtype=np.float32)
        raw_mask      = np.array(landmark_data["mask"],      dtype=np.float32)

        aug_params = None
        if self.augment:
            label_name = CLASSES[label_id]
            augmenter = VideoAugmentation(
                use_horizontal_flip=self.augment.use_horizontal_flip,
                extra_strong=(label_name in _WEAK_CLASSES),
            ) if label_name in _WEAK_CLASSES else self.augment
            frames, aug_params = augmenter(frames)
            lm_3d = raw_landmarks.reshape(-1, 75, 3)
            lm_3d, raw_mask = apply_landmark_augmentation(lm_3d, raw_mask, aug_params)
            raw_landmarks = lm_3d.reshape(-1, 225)

        if USE_MOTION:
            lm_3d_for_motion = raw_landmarks.reshape(-1, 75, 3)
            motion = compute_motion_features(lm_3d_for_motion, raw_mask)
            landmark_features = torch.tensor(
                np.concatenate([raw_landmarks, motion], axis=-1), dtype=torch.float32
            )
        else:
            landmark_features = torch.tensor(raw_landmarks, dtype=torch.float32)

        rgb_tensor = []
        for frame in frames:
            ft = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            ft = self.normalize(ft)
            rgb_tensor.append(ft)
        rgb_tensor = torch.stack(rgb_tensor, dim=0)

        return {
            "video_id":  video_id,
            "rgb":       rgb_tensor,
            "landmarks": landmark_features,
            "mask":      torch.tensor(raw_mask, dtype=torch.float32),
            "label":     label_id,
        }

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        video_id = str(row["video_id"])
        label_id = int(row["label_id"])
        feature_dir = self._get_feature_dir(row)

        cache_path = feature_dir / f"{video_id}.pt"
        if self.use_cache and cache_path.exists():
            # Reject old feature formats and transparently rebuild from video.
            # This keeps training correct even when a user retains old caches.
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                if cached.get("feature_version") == FEATURE_VERSION:
                    return self._load_from_cache(video_id, label_id, feature_dir)
            except Exception:
                pass
        return self._load_from_video(row)
