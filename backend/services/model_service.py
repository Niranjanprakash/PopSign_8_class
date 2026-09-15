import torch
from pathlib import Path
from backend.config import CHECKPOINT_DIR, CLASSES, NUM_CLASSES, NUM_FRAMES, USE_MOTION
from backend.models.lightmamba_asl import LightMambaASL
from backend.utils import get_device

_model_instance = None
_device = None
_class_mapping = CLASSES
_checkpoint_verified = False

def get_loaded_model():
    """Lazily loads and returns the model instance in eval mode."""
    global _model_instance, _device, _class_mapping, _checkpoint_verified
    if _model_instance is not None:
        return _model_instance, _device
        
    _device = get_device()
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    # Initialize architecture
    model = LightMambaASL(pretrained=False, freeze_backbone=False)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Train a verified model before enabling prediction."
        )

    print(f"[SERVICE] Loading best model weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid model checkpoint: missing model_state_dict in {checkpoint_path}")
    if not checkpoint.get("split_signature"):
        raise ValueError(
            "Unverified model checkpoint: missing split_signature. "
            "Retrain with `python -m backend.training.train` before serving predictions."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    _class_mapping = checkpoint.get("class_mapping", CLASSES)
    if len(_class_mapping) != NUM_CLASSES:
        raise ValueError(
            f"Invalid checkpoint class mapping: expected {NUM_CLASSES} classes, got {len(_class_mapping)}"
        )
    _checkpoint_verified = True
        
    model.to(_device)
    model.eval()
    _model_instance = model
    return _model_instance, _device

def get_checkpoint_status() -> dict:
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    class_mapping = _class_mapping if _class_mapping is not None else CLASSES
    return {
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "checkpoint_verified": _checkpoint_verified,
        "class_mapping_matches_config": class_mapping == CLASSES,
        "loaded_class_mapping": class_mapping,
    }

def get_class_mapping() -> list:
    """Returns the class ordering attached to the loaded checkpoint."""
    return _class_mapping if _class_mapping is not None else CLASSES
