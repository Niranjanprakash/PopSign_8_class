import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from backend.config import PROCESSED_DIR, OUTPUT_DIR, CHECKPOINT_DIR, CLASSES
from backend.utils import get_device, logger
from backend.data.dataset import ASLVideoDataset
from backend.models.lightmamba_asl import LightMambaASL
from backend.training.metrics import calculate_metrics
from backend.evaluation.confusion_matrix import generate_confusion_analysis
from backend.evaluation.efficiency import measure_efficiency

def main():
    device = get_device()
    
    test_csv = PROCESSED_DIR / "splits" / "test.csv"
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    if not test_csv.exists():
        logger.error("Test split file not found! Please run: python -m backend.data.prepare_dataset")
        return
        
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}! Please train a model first: python -m backend.training.train")
        return
        
    logger.info(f"Loading best checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not checkpoint.get("split_signature"):
        raise ValueError(
            "Refusing to evaluate an unverified checkpoint without split_signature. "
            "Retrain with the current leakage-safe pipeline first."
        )
    class_mapping = checkpoint.get("class_mapping", CLASSES)
    
    # Instantiate Model
    model = LightMambaASL(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load dataset
    test_dataset = ASLVideoDataset(str(test_csv), is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    logger.info("Evaluating on unseen test split...")
    
    all_logits = []
    all_targets = []
    video_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            rgb = batch["rgb"].to(device)
            landmarks = batch["landmarks"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["label"].to(device)
            
            preextracted = (rgb.dim() == 3)
            logits = model(rgb, landmarks, mask, preextracted_rgb=preextracted)
            
            all_logits.append(logits)
            all_targets.append(targets)
            video_ids.extend(batch["video_id"])
            
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 1. Calculate General Metrics
    metrics = calculate_metrics(all_logits, all_targets)
    
    # Per-Class Accuracy
    preds = torch.argmax(all_logits, dim=-1).cpu().numpy()
    targets_np = all_targets.cpu().numpy()
    
    per_class_acc = {}
    for i, cls in enumerate(class_mapping):
        cls_indices = np.where(targets_np == i)[0]
        if len(cls_indices) > 0:
            cls_acc = np.mean(preds[cls_indices] == targets_np[cls_indices])
            per_class_acc[cls] = float(cls_acc)
        else:
            per_class_acc[cls] = 0.0
            
    metrics["per_class_accuracy"] = per_class_acc
    
    # 2. Confusion Analysis
    generate_confusion_analysis(targets_np, preds)
    
    # 3. Model Efficiency Evaluation
    efficiency_metrics = measure_efficiency(model, device)
    metrics["efficiency"] = efficiency_metrics
    
    # Print Test Evaluation Results
    print("\n" + "="*50)
    print("TEST EVALUATION REPORT")
    print("="*50)
    print(f"Top-1 Test Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Top-5 Test Accuracy : {metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Macro F1-score      : {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-score   : {metrics['f1_weighted']:.4f}")
    print("="*50)
    print("Per-Class Accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  - {cls:<12}: {acc * 100:.2f}%")
    print("="*50)
    
    # 4. Save results to JSON
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save predictions details
    predictions_dir = OUTPUT_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "video_id": video_ids,
        "true_label_id": targets_np,
        "true_label": [class_mapping[t] for t in targets_np],
        "pred_label_id": preds,
        "pred_label": [class_mapping[p] for p in preds]
    })
    pred_df.to_csv(predictions_dir / "test_predictions.csv", index=False)
    
    logger.info("Evaluation pipeline complete!")

if __name__ == "__main__":
    main()
