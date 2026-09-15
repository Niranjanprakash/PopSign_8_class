import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights
import numpy as np
from pathlib import Path
from backend.config import PROCESSED_DIR, RGB_FEATURE_DIM

class MobileNetV3FeatureExtractor(nn.Module):
    """
    Wraps MobileNetV3-Large to extract frame-level spatial appearance features.
    If frozen, can be used for feature extraction/caching.
    If fine-tuned, runs end-to-end as part of the model.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.mobilenet = models.mobilenet_v3_large(weights=weights)

        # MobileNetV3-Large: features -> [B, 960, 7, 7]
        self.features = self.mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [Batch * T, C, H, W]
        Returns: [Batch * T, 960]
        """
        feats = self.features(x)
        pooled = self.pool(feats)
        # Flatten to 960-dim
        flat = torch.flatten(pooled, 1)
        return flat

def extract_and_cache_rgb_features(video_id: str, rgb_tensor: torch.Tensor, extractor: MobileNetV3FeatureExtractor, device: torch.device) -> np.ndarray:
    """
    Caches extracted RGB features if using frozen backbone mode.
    rgb_tensor shape: [T, C, H, W]
    Returns: [T, 576] numpy array
    """
    cache_dir = PROCESSED_DIR / "cache" / "rgb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_id}.npy"

    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"[RGB CACHE] Error loading cache for {video_id}: {e}")

    # Extract
    extractor.eval()
    with torch.no_grad():
        # Add batch dimension: [T, C, H, W] -> T frames
        inputs = rgb_tensor.to(device)
        feats = extractor(inputs) # [T, 576]
        feats_np = feats.cpu().numpy()

    # Cache
    try:
        np.save(cache_path, feats_np)
    except Exception as e:
        print(f"[RGB CACHE] Error writing cache for {video_id}: {e}")

    return feats_np
