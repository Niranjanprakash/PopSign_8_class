import torch
import torch.nn as nn
from backend.features.rgb_extractor import MobileNetV3FeatureExtractor

class MobileNetBranch(nn.Module):
    """
    RGB Branch. If inputs are frames: [B, T, C, H, W], extracts features using MobileNetV3.
    If inputs are already extracted features: [B, T, 576], passes them or projects them.
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.extractor = MobileNetV3FeatureExtractor(pretrained=pretrained)
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            # Freeze parameters
            for param in self.extractor.parameters():
                param.requires_grad = False
                
    def unfreeze_final_blocks(self):
        """Allows end-to-end fine-tuning of the final layers."""
        self.freeze_backbone = False
        for param in self.extractor.features[-3:].parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, preextracted: bool = False) -> torch.Tensor:
        """
        If preextracted is False:
            x shape: [B, T, C, H, W] -> returns [B, T, 576]
        If preextracted is True:
            x shape: [B, T, 576] -> returns [B, T, 576]
        """
        if preextracted:
            return x

        B, T, C, H, W = x.shape
        # Flatten temporal and batch dimensions for CNN processing
        x_flat = x.view(B * T, C, H, W)
        
        # If frozen, run without gradient tracking for speed
        if self.freeze_backbone:
            with torch.no_grad():
                features_flat = self.extractor(x_flat)
        else:
            features_flat = self.extractor(x_flat)
            
        # Reshape back to sequence
        features = features_flat.view(B, T, -1) # [B, T, 960]
        return features
