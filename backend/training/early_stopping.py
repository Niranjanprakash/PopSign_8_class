import torch
from pathlib import Path
from backend.config import CHECKPOINT_DIR, EARLY_STOPPING_PATIENCE

class EarlyStopping:
    """
    Early stopping helper that monitors validation loss.
    Saves model checkpoints when validation loss decreases.
    """
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.best_path = CHECKPOINT_DIR / "best_model.pth"
        self.last_path = CHECKPOINT_DIR / "last_model.pth"

    def __call__(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, class_mapping: list, split_signature: str = None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics, class_mapping, split_signature, is_best=True)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EARLY STOPPING] Validation loss did not improve. Counter: {self.counter} / {self.patience}")
            # Save last checkpoint anyway
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics, class_mapping, split_signature, is_best=False)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, metrics, class_mapping, split_signature, is_best=True)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, class_mapping: list, split_signature: str = None, is_best: bool = False):
        """Saves a model state dictionary and metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": val_loss,
            "metrics": metrics,
            "class_mapping": class_mapping,
            "split_signature": split_signature,
        }
        
        path = self.best_path if is_best else self.last_path
        torch.save(checkpoint, path)
        if self.verbose:
            status = "BEST" if is_best else "LAST"
            print(f"[CHECKPOINT] Saved {status} model checkpoint to {path.name} (Val Loss: {val_loss:.4f})")
