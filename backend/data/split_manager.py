from pathlib import Path

from backend.config import CLASSES


def _popsign_signer_id(video_name: str) -> str:
    """Extract a stable signer identifier from a PopSign video filename."""
    return video_name.split("-", 1)[0]


def parse_popsign_splits(dataset_root: Path) -> list:
    """Read PopSign folders in nested or flat split layouts."""
    samples = []
    class_to_idx = {name: index for index, name in enumerate(CLASSES)}

    for split_name in ("train", "val", "test"):
        split_root = dataset_root / split_name / split_name
        if not split_root.exists():
            split_root = dataset_root / split_name

        for class_name in CLASSES:
            class_root = split_root / class_name
            if not class_root.exists():
                continue

            for video_path in sorted(class_root.glob("*.mp4")):
                samples.append({
                    "video_id": video_path.stem,
                    "video_path": str(video_path),
                    "label": class_name,
                    "label_id": class_to_idx[class_name],
                    "split": split_name,
                    "signer_id": _popsign_signer_id(video_path.name),
                })

    return samples
