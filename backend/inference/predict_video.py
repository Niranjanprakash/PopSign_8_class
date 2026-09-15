import argparse
import sys
import torch
from pathlib import Path
from torchvision import transforms

from backend.config import CHECKPOINT_DIR, NUM_FRAMES, IMAGE_SIZE, USE_MOTION
from backend.utils import get_device, logger
from backend.data.video_loader import load_video
from backend.features.feature_cache import get_or_extract_features
from backend.models.lightmamba_asl import LightMambaASL
from backend.services.model_service import get_class_mapping
from backend.inference.confidence import calibrate_prediction, get_ambiguity_warning

def predict_single_video(video_path: str, checkpoint_path: str = None) -> dict:
    device = get_device()
    class_mapping = get_class_mapping()
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    else:
        checkpoint_path = Path(checkpoint_path)
        
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. Load model
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LightMambaASL(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Preprocess video frames
    logger.info("Decoding and sampling video frames...")
    frames = load_video(str(video_path), target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
    
    # 3. Extract RGB features
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rgb_tensors = []
    for frame in frames:
        f_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        f_t = normalize(f_t)
        rgb_tensors.append(f_t)
    rgb_tensor = torch.stack(rgb_tensors, dim=0).unsqueeze(0).to(device) # [1, 32, 3, 224, 224]

    # 4. Extract landmarks & motion
    logger.info("Extracting landmarks using MediaPipe...")
    video_id = video_path.stem
    landmark_data = get_or_extract_features(video_id, frames)
    
    landmarks = torch.tensor(landmark_data["landmarks"], dtype=torch.float32)
    motion = torch.tensor(landmark_data["motion"], dtype=torch.float32)
    mask = torch.tensor(landmark_data["mask"], dtype=torch.float32).unsqueeze(0).to(device) # [1, 32, 3]

    if USE_MOTION:
        landmark_features = torch.cat([landmarks, motion], dim=-1)
    else:
        landmark_features = landmarks
    landmark_features = landmark_features.unsqueeze(0).to(device) # [1, 32, L_DIM]

    # 5. Model Inference
    logger.info("Running model forward pass...")
    with torch.no_grad():
        logits = model(rgb_tensor, landmark_features, mask)
        
    pred_idx, confidence, top_k = calibrate_prediction(logits)
    
    # Format response
    ambiguity = get_ambiguity_warning(top_k, class_mapping)
    if pred_idx == -1 or ambiguity:
        pred_label = "UNCERTAIN"
    else:
        pred_label = class_mapping[pred_idx]
        
    result = {
        "prediction": pred_label,
        "confidence": confidence,
        "uncertain": (pred_idx == -1 or ambiguity is not None),
        "ambiguous": ambiguity is not None,
        "ambiguity": ambiguity,
        "top_predictions": [{"class": class_mapping[x["class_id"]], "confidence": x["probability"]} for x in top_k]
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ASL word from video clip.")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth.")
    args = parser.parse_args()
    
    try:
        res = predict_single_video(args.video, args.checkpoint)
        print("\n" + "="*50)
        print("INFERENCE RESULT")
        print("="*50)
        print(f"Predicted Sign : {res['prediction']}")
        print(f"Confidence     : {res['confidence'] * 100:.2f}%")
        print("="*50)
        print("Top Predictions:")
        for pred in res["top_predictions"]:
            print(f"  - {pred['class']:<12}: {pred['confidence'] * 100:.2f}%")
        print("="*50)
    except Exception as e:
        print(f"[ERROR] Predict video failed: {e}")
        sys.exit(1)
