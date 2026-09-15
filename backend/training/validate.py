import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from backend.training.metrics import calculate_metrics

def validate_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple:
    """
    Evaluates the model on the validation dataset.
    Returns:
        - avg_loss: float
        - val_metrics: dict
    """
    model.eval()
    total_loss = 0.0
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            # Move to device
            rgb = batch["rgb"].to(device)
            landmarks = batch["landmarks"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["label"].to(device)
            preextracted = (rgb.dim() == 3)

            # Forward pass
            logits = model(rgb, landmarks, mask, preextracted_rgb=preextracted)
            loss = loss_fn(logits, targets)
            
            total_loss += loss.item() * targets.size(0)
            
            all_logits.append(logits)
            all_targets.append(targets)

    # Concatenate all outputs
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_logits, all_targets)
    
    return avg_loss, metrics
