import cv2
import pandas as pd
from pathlib import Path
from itertools import combinations
from backend.config import PROCESSED_DIR, CLASSES, POPSIGN_ROOT
from backend.data.split_manager import parse_popsign_splits


def check_video_validity(video_path: str) -> bool:
    """Returns True if OpenCV can open the video and it has > 0 frames."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames > 0
    except Exception:
        return False


def prepare_dataset():
    if POPSIGN_ROOT is None:
        raise RuntimeError(
            "POPSIGN_ROOT is required to prepare PopSign training data."
        )

    dataset_name = "PopSign 4-Class"
    print(f"[PREPARATION] Starting {dataset_name} dataset preparation...")
    print(f"[PREPARATION] Target classes: {len(CLASSES)}")

    # 1. Parse the official PopSign train, validation, and test folders.
    samples = parse_popsign_splits(POPSIGN_ROOT)
    print(f"[PREPARATION] Found {len(samples)} candidates matching {len(CLASSES)} classes. Validating files...")

    # 2. Validate each video file
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    valid_samples = []
    missing_count = 0
    corrupted_count = 0

    for s in samples:
        v_path = Path(s["video_path"])
        if not v_path.exists():
            missing_count += 1
            continue
        if check_video_validity(str(v_path)):
            s["label_id"] = class_to_idx[s["label"]]
            valid_samples.append(s)
        else:
            corrupted_count += 1

    df = pd.DataFrame(valid_samples)
    if df.empty:
        print("[ERROR] No valid videos found. Check the configured dataset root.")
        return

    # 3. Save processed splits
    splits_dir = PROCESSED_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    # Data leakage check: IDs, paths, and signer identity must be disjoint.
    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    for (left_name, left_df), (right_name, right_df) in combinations(split_frames.items(), 2):
        for column in ("video_id", "video_path", "signer_id"):
            overlap = set(left_df[column].astype(str)) & set(right_df[column].astype(str))
            assert not overlap, (
                f"Data leakage: {left_name}/{right_name} overlap in {column}: "
                f"{sorted(overlap)[:3]}"
            )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir   / "val.csv",   index=False)
    test_df.to_csv(splits_dir  / "test.csv",  index=False)

    # 4. Report
    print("\n" + "=" * 55)
    print(f"  {dataset_name.upper()} DATASET PREPARATION REPORT")
    print("=" * 55)
    print(f"  Total candidates  : {len(samples)}")
    print(f"  Valid videos      : {len(df)}")
    print(f"  Missing on disk   : {missing_count}")
    print(f"  Corrupted         : {corrupted_count}")
    print(f"  Train             : {len(train_df)}")
    print(f"  Val               : {len(val_df)}")
    print(f"  Test              : {len(test_df)}")
    print(f"  Classes           : {len(CLASSES)}")
    print("=" * 55)

    stats = []
    for cls in CLASSES:
        cls_df = df[df["label"] == cls]
        stats.append({
            "Class": cls,
            "Train": len(cls_df[cls_df["split"] == "train"]),
            "Val":   len(cls_df[cls_df["split"] == "val"]),
            "Test":  len(cls_df[cls_df["split"] == "test"]),
            "Total": len(cls_df),
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("=" * 55)

    stats_df.to_csv(PROCESSED_DIR / "dataset_stats.csv", index=False)


if __name__ == "__main__":
    prepare_dataset()
