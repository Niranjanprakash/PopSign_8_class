import os
from pathlib import Path

# Paths Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
BACKEND_ASSETS_DIR = PROJECT_ROOT / "backend" / "assets"
MEDIAPIPE_ASSETS_DIR = BACKEND_ASSETS_DIR / "mediapipe"
PROCESSED_DIR = DATASET_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
HOLISTIC_LANDMARKER_MODEL_PATH = MEDIAPIPE_ASSETS_DIR / "holistic_landmarker.task"
HOLISTIC_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)

# Optional source directory used only when preparing or retraining on PopSign.
POPSIGN_ROOT = Path(os.environ["POPSIGN_ROOT"]) if os.environ.get("POPSIGN_ROOT") else None

# Ensure all critical folders exist
for folder in [
    DATASET_ROOT,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    UPLOAD_DIR,
    BACKEND_ASSETS_DIR,
    MEDIAPIPE_ASSETS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# PopSign classes supported by the shipped checkpoint.
CLASSES = ["after", "airplane", "bird", "cloud", "cry", "dog", "drink", "elephant"]
NUM_CLASSES = len(CLASSES)

# Preprocessing Hyperparameters
NUM_FRAMES = 32
IMAGE_SIZE = 224

# Training Hyperparameters
BATCH_SIZE = 8
MAX_EPOCHS = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 15   # was 10 — give weak classes more time to converge
RANDOM_SEED = 42

# Oversampling — weak classes get sampled this many times more than the majority class.
# dog (153 videos) and airplane (255 videos) are the two weakest.
OVERSAMPLING_WEIGHTS = {
    "dog":      3.5,   # 153 videos → effectively ~535
    "airplane": 2.0,   # 255 videos → effectively ~510
    "cloud":    1.5,   # 364 videos → effectively ~546
    "after":    1.3,   # 395 videos → effectively ~514
    "cry":      1.3,   # 388 videos → effectively ~504
    "elephant": 1.1,   # 459 videos → effectively ~505
    "bird":     1.0,   # 561 videos — baseline
    "drink":    1.0,   # 571 videos — baseline
}

# Model Dimensions
RGB_FEATURE_DIM = 960  # MobileNetV3-Large feature dimension (last channel before classifier)
LANDMARK_EMBED_DIM = 256
FUSION_DIM = 256
MAMBA_HIDDEN_DIM = 256
DROPOUT = 0.3

# Ablation & Novelty Configuration Switches
USE_RGB = True
USE_LANDMARKS = True
USE_MOTION = True
USE_SECOND_ORDER_MOTION = True
USE_RELIABILITY_FUSION = True
USE_HORIZONTAL_FLIP = True

# Feature Cache Config
USE_FEATURE_CACHE = True  # Load pre-extracted .pt features instead of raw video
# Bump when RGB, landmark, motion, or frame preprocessing changes. Older caches
# are rejected by the dataset instead of silently contaminating a training run.
FEATURE_VERSION = "v3_large_rgb_holistic_motion"
CONFIDENCE_THRESHOLD = 0.25
TEMPORAL_SMOOTHING_WINDOW = 5

# Inference safety: do not present one label when a known confused pair has
# nearly equal probabilities. This changes only post-processing, never weights.
AMBIGUITY_MARGIN_THRESHOLD = 0.15
AMBIGUOUS_CLASS_PAIRS = (
    frozenset(("airplane", "bird")),
    frozenset(("airplane", "cloud")),
    frozenset(("airplane", "drink")),
    frozenset(("airplane", "cry")),
    frozenset(("airplane", "after")),
    frozenset(("bird",     "drink")),
    frozenset(("bird",     "cry")),
    frozenset(("bird",     "cloud")),
    frozenset(("cloud",    "drink")),
    frozenset(("elephant", "cloud")),
    frozenset(("elephant", "after")),
    frozenset(("dog",      "after")),
)
