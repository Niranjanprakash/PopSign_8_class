import torch
from pathlib import Path
from torchvision import transforms
import numpy as np

from backend.config import NUM_FRAMES, IMAGE_SIZE, USE_MOTION, CLASSES, NUM_CLASSES
from backend.data.video_loader import load_video
from backend.features.feature_cache import get_or_extract_features
from backend.services.model_service import get_loaded_model, get_checkpoint_status, get_class_mapping
from backend.inference.confidence import calibrate_prediction, get_ambiguity_warning

def get_model_details() -> dict:
    status = get_checkpoint_status()
    return {
        "model_name": "LightMamba-ASL",
        "num_classes": NUM_CLASSES,
        "frames_per_video": NUM_FRAMES,
        "input_resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "checkpoint_status": status,
        "use_motion": USE_MOTION
    }

def predict_video_file(video_path: Path) -> dict:
    """Runs prediction on uploaded video file."""
    # 1. Load model and device
    model, device = get_loaded_model()
    class_mapping = get_class_mapping()
    
    # 2. Decode and sample video
    frames = load_video(str(video_path), target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
    
    # 3. Preprocess RGB sequence
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rgb_tensors = []
    for frame in frames:
        f_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        f_t = normalize(f_t)
        rgb_tensors.append(f_t)
    rgb_tensor = torch.stack(rgb_tensors, dim=0).unsqueeze(0).to(device) # [1, 32, 3, 224, 224]

    # 4. Extract landmarks & motion
    # Use video path stem as video_id
    video_id = video_path.stem
    landmark_data = get_or_extract_features(video_id, frames)
    
    landmarks = torch.tensor(landmark_data["landmarks"], dtype=torch.float32)
    motion = torch.tensor(landmark_data["motion"], dtype=torch.float32)
    mask = torch.tensor(landmark_data["mask"], dtype=torch.float32).unsqueeze(0).to(device)

    if USE_MOTION:
        landmark_features = torch.cat([landmarks, motion], dim=-1)
    else:
        landmark_features = landmarks
    landmark_features = landmark_features.unsqueeze(0).to(device) # [1, 32, L_DIM]

    # 5. Run inference
    with torch.no_grad():
        logits = model(rgb_tensor, landmark_features, mask)
        
    pred_idx, confidence, top_k = calibrate_prediction(logits)
    
    ambiguity = get_ambiguity_warning(top_k, class_mapping)
    if pred_idx == -1 or ambiguity:
        pred_label = "UNCERTAIN"
    else:
        pred_label = class_mapping[pred_idx]
        
    return {
        "prediction": pred_label,
        "confidence": confidence,
        "uncertain": (pred_idx == -1 or ambiguity is not None),
        "ambiguous": ambiguity is not None,
        "ambiguity": ambiguity,
        "top_predictions": [{"class": class_mapping[x["class_id"]], "confidence": x["probability"]} for x in top_k]
    }
