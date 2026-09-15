import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def get_scheduler(optimizer, scheduler_type: str = "CosineAnnealingLR", max_epochs: int = 150):
    if scheduler_type == "CosineAnnealingLR":
        print(f"[SCHEDULER] Using CosineAnnealingLR (T_max={max_epochs}, eta_min=1e-6).")
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    elif scheduler_type == "ReduceLROnPlateau":
        print("[SCHEDULER] Using ReduceLROnPlateau.")
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
