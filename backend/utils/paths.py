from backend.config import PROJECT_ROOT, DATASET_ROOT, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR, UPLOAD_DIR

def verify_paths():
    """Checks and creates all required directories if not present."""
    directories = [
        DATASET_ROOT,
        PROCESSED_DIR,
        PROCESSED_DIR / "splits",
        PROCESSED_DIR / "landmarks",
        PROCESSED_DIR / "cache",
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        OUTPUT_DIR / "plots",
        OUTPUT_DIR / "confusion_matrix",
        OUTPUT_DIR / "metrics",
        OUTPUT_DIR / "logs",
        OUTPUT_DIR / "experiments",
        OUTPUT_DIR / "predictions",
        UPLOAD_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
