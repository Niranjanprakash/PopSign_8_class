"""
Full test-set sweep — model loaded ONCE, reused for all videos.
"""
import sys
import torch
from pathlib import Path
from collections import defaultdict
from torchvision import transforms

from backend.config import CHECKPOINT_DIR, NUM_FRAMES, IMAGE_SIZE, USE_MOTION
from backend.utils import get_device
from backend.data.video_loader import load_video
from backend.features.feature_cache import get_or_extract_features
from backend.models.lightmamba_asl import LightMambaASL
from backend.services.model_service import get_class_mapping
from backend.inference.confidence import calibrate_prediction, get_ambiguity_warning

TEST_ROOT = Path("dataset/popsign/test")
CLASSES   = ["after", "airplane", "bird", "cloud", "cry", "dog", "drink", "elephant"]

# ── Load model ONCE ────────────────────────────────────────
print("Loading model...", flush=True)
device = get_device()
checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=device)
model = LightMambaASL(pretrained=False, freeze_backbone=False).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
class_mapping = get_class_mapping()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print("Model ready.\n", flush=True)

def predict(video_path):
    frames = load_video(str(video_path), target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)
    rgb_tensors = [normalize(torch.from_numpy(f).permute(2,0,1).float()/255.0) for f in frames]
    rgb_tensor = torch.stack(rgb_tensors).unsqueeze(0).to(device)

    ld = get_or_extract_features(video_path.stem, frames)
    landmarks = torch.tensor(ld["landmarks"], dtype=torch.float32)
    motion    = torch.tensor(ld["motion"],    dtype=torch.float32)
    mask      = torch.tensor(ld["mask"],      dtype=torch.float32).unsqueeze(0).to(device)
    lm_feat   = torch.cat([landmarks, motion], dim=-1).unsqueeze(0).to(device) if USE_MOTION else landmarks.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(rgb_tensor, lm_feat, mask)
    pred_idx, confidence, top_k = calibrate_prediction(logits)
    ambiguity = get_ambiguity_warning(top_k, class_mapping)
    if pred_idx == -1 or ambiguity:
        return "UNCERTAIN", confidence
    return class_mapping[pred_idx], confidence

# ── Run sweep ──────────────────────────────────────────────
per_class = defaultdict(lambda: {"total": 0, "pass": 0, "mismatches": []})
confusion  = defaultdict(lambda: defaultdict(int))

for cls in CLASSES:
    videos = sorted((TEST_ROOT / cls).glob("*.mp4"))
    total  = len(videos)
    for i, video in enumerate(videos, 1):
        print(f"\r{cls} [{i}/{total}]   ", end="", flush=True)
        try:
            pred, conf = predict(video)
        except Exception as e:
            pred, conf = "ERROR", 0.0
        per_class[cls]["total"] += 1
        if pred.lower() == cls.lower():
            per_class[cls]["pass"] += 1
        else:
            per_class[cls]["mismatches"].append((video.name, pred, conf * 100))
            confusion[cls][pred] += 1
    print(f"\r{cls}: {per_class[cls]['pass']}/{total} correct          ")

# ── Summary table ──────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Class':<12} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Acc%':>7}")
print("="*60)
total_all = pass_all = 0
for cls in CLASSES:
    t = per_class[cls]["total"]
    p = per_class[cls]["pass"]
    total_all += t; pass_all += p
    print(f"{cls:<12} {t:>6} {p:>6} {t-p:>6} {100*p/t:>6.1f}%")
print("="*60)
print(f"{'TOTAL':<12} {total_all:>6} {pass_all:>6} {total_all-pass_all:>6} {100*pass_all/total_all:>6.1f}%")

# ── Confusion breakdown ────────────────────────────────────
print("\n" + "="*60)
print("CONFUSION  (true → predicted as)")
print("="*60)
for cls in CLASSES:
    for pred, cnt in sorted(confusion[cls].items(), key=lambda x: -x[1]):
        pct = 100 * cnt / per_class[cls]["total"]
        print(f"  {cls:<12} → {pred:<12}  {cnt:>4}x  ({pct:.1f}%)")

# ── Mismatch detail ────────────────────────────────────────
print("\n" + "="*60)
print("MISMATCH DETAILS")
print("="*60)
for cls in CLASSES:
    mm = per_class[cls]["mismatches"]
    if not mm:
        print(f"\n{cls}: all correct ✅")
        continue
    print(f"\n{cls}  ({len(mm)} fails):")
    for vname, pred, conf in mm:
        print(f"  {vname[:52]:<52} → {pred:<12} {conf:.1f}%")
