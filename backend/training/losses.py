import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples (drink, cry) so hard classes
    (dog, airplane, bird) get more gradient signal.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def get_loss_fn(class_counts: list = None, use_weights: bool = False,
               label_smoothing: float = 0.1, focal_gamma: float = 2.0) -> nn.Module:
    """
    Returns FocalLoss with class weights + label smoothing.
    Focal loss prevents dominant classes (drink) from overwhelming gradients.
    """
    weights_tensor = None
    if use_weights and class_counts is not None:
        total = sum(class_counts)
        weights = [total / (len(class_counts) * c) if c > 0 else 0.0 for c in class_counts]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        print(f"[LOSS] Class weights: {dict(zip(['after','airplane','bird','cloud','cry','dog','drink','elephant'], [round(w,2) for w in weights]))}")

    print(f"[LOSS] Using FocalLoss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    return FocalLoss(weight=weights_tensor, gamma=focal_gamma, label_smoothing=label_smoothing)
