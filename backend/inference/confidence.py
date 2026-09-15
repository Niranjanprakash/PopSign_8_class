import torch
from backend.config import (
    AMBIGUITY_MARGIN_THRESHOLD,
    AMBIGUOUS_CLASS_PAIRS,
    CONFIDENCE_THRESHOLD,
)

def calibrate_prediction(logits: torch.Tensor, threshold: float = CONFIDENCE_THRESHOLD) -> tuple:
    """
    Applies softmax to logits, fetches class prediction, and compares against threshold.
    Returns:
        - pred_class_idx: int (or -1 if uncertain)
        - confidence: float
        - top_k_probs: list of dicts {"class_id": int, "probability": float}
    """
    probs = torch.softmax(logits, dim=-1).squeeze(0) # [num_classes]
    
    # Sort probabilities
    top_probs, top_indices = torch.topk(probs, k=min(3, len(probs)))
    
    confidence = float(top_probs[0].item())
    pred_idx = int(top_indices[0].item())
    
    top_k_list = []
    for p, idx in zip(top_probs, top_indices):
        top_k_list.append({
            "class_id": int(idx.item()),
            "probability": float(p.item())
        })
        
    if confidence < threshold:
        return -1, confidence, top_k_list
    else:
        return pred_idx, confidence, top_k_list


def get_ambiguity_warning(
    top_predictions: list[dict],
    class_mapping: list[str],
    margin_threshold: float = AMBIGUITY_MARGIN_THRESHOLD,
) -> dict | None:
    """Abstain for a known close-probability pair instead of relabelling it."""
    if len(top_predictions) < 2:
        return None

    first, second = top_predictions[:2]
    first_label = class_mapping[first["class_id"]]
    second_label = class_mapping[second["class_id"]]
    margin = round(first["probability"] - second["probability"], 6)

    if (
        frozenset((first_label, second_label)) in AMBIGUOUS_CLASS_PAIRS
        and margin <= margin_threshold
    ):
        labels = [first_label, second_label]
        return {
            "labels": labels,
            "margin": float(margin),
            "message": (
                f"The model cannot safely distinguish {labels[0]} and {labels[1]} "
                "from this video. Keep both hands visible and record again."
            ),
        }
    return None
