import torch
import torch.nn as nn
from backend.config import LANDMARK_EMBED_DIM, DROPOUT, USE_MOTION, USE_SECOND_ORDER_MOTION

class LandmarkBranch(nn.Module):
    """
    MLP Landmark Encoder. Maps raw/motion landmark features to embedding space.
    """
    def __init__(self, embed_dim: int = LANDMARK_EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        
        # Calculate input dimension
        # Left Hand (21) + Right Hand (21) + Pose (33) = 75 landmarks
        # 75 * 3 coords = 225 dimensions
        base_dim = 225
        input_dim = base_dim
        
        if USE_MOTION:
            if USE_SECOND_ORDER_MOTION:
                # Base coords + first-order + second-order
                input_dim = base_dim + base_dim * 2
            else:
                # Base coords + first-order
                input_dim = base_dim + base_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        landmarks shape: [B, T, input_dim]
        Returns: [B, T, embed_dim]
        """
        return self.net(landmarks)
