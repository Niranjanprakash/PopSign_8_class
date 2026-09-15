import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# MediaPipe Holistic landmark hand swap indices:
# Left hand (0-20) <-> Right hand (21-41) must be swapped on horizontal flip
_LH_SLICE = slice(0, 21)
_RH_SLICE = slice(21, 42)

class VideoAugmentation:
    """
    Strong augmentation pipeline for small datasets.
    Applies consistent transformations across all frames in a sequence [T, H, W, C].
    Returns (augmented_frames, aug_params) so landmarks can be synced in dataset.py.
    """
    def __init__(self, use_horizontal_flip: bool = True,
                 brightness_range: tuple = (0.5, 1.5),
                 contrast_range: tuple = (0.6, 1.4),
                 saturation_range: tuple = (0.7, 1.3),
                 noise_std: float = 0.03,
                 speed_jitter: bool = True,
                 rotation_range: float = 15.0,
                 zoom_range: tuple = (0.85, 1.15),
                 extra_strong: bool = False):   # True for dog / airplane
        self.use_horizontal_flip = use_horizontal_flip
        self.brightness_range = brightness_range if not extra_strong else (0.35, 1.65)
        self.contrast_range   = contrast_range   if not extra_strong else (0.45, 1.55)
        self.saturation_range = saturation_range if not extra_strong else (0.5,  1.5)
        self.noise_std        = noise_std        if not extra_strong else 0.06
        self.speed_jitter     = speed_jitter
        self.rotation_range   = rotation_range   if not extra_strong else 22.0
        self.zoom_range       = zoom_range       if not extra_strong else (0.78, 1.22)
        self.extra_strong     = extra_strong

    def __call__(self, frames: np.ndarray):
        """
        Returns:
            augmented_frames: np.ndarray [T, H, W, C]
            aug_params: dict with keys 'do_flip' and 'speed_indices' (or None)
        """
        T, H, W, C = frames.shape

        do_flip      = self.use_horizontal_flip and (random.random() > 0.5)
        brightness_f = random.uniform(*self.brightness_range)
        contrast_f   = random.uniform(*self.contrast_range)
        saturation_f = random.uniform(*self.saturation_range)
        do_noise     = random.random() > 0.4
        angle        = random.uniform(-self.rotation_range, self.rotation_range) if random.random() > 0.5 else 0.0
        zoom         = random.uniform(*self.zoom_range) if random.random() > 0.5 else 1.0

        # Temporal speed jitter — record indices for landmark sync
        speed_indices = None
        if self.speed_jitter and random.random() > 0.4:
            speed = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            raw_indices = np.linspace(0, T - 1, T) * speed
            speed_indices = np.clip(raw_indices, 0, T - 1).astype(int)
            frames = frames[speed_indices]

        M = None
        if angle != 0.0 or zoom != 1.0:
            cx, cy = W / 2, H / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)

        augmented_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageEnhance.Brightness(img).enhance(brightness_f)
            img = ImageEnhance.Contrast(img).enhance(contrast_f)
            img = ImageEnhance.Color(img).enhance(saturation_f)
            arr = np.array(img, dtype=np.float32)
            if M is not None:
                arr = cv2.warpAffine(arr, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            if do_noise:
                arr += np.random.normal(0, self.noise_std * 255, arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            augmented_frames.append(arr)

        result = np.stack(augmented_frames, axis=0)

        if random.random() > 0.7:  # less frequent temporal masking
            n_mask = max(1, int(T * 0.1))  # mask only 10% frames, not 20%
            mask_indices = random.sample(range(T), n_mask)
            result[mask_indices] = 0

        if random.random() > 0.5:
            cut_h = random.randint(H // 6, H // 3)
            cut_w = random.randint(W // 6, W // 3)
            cy = random.randint(0, H - cut_h)
            cx = random.randint(0, W - cut_w)
            result[:, cy:cy + cut_h, cx:cx + cut_w, :] = 0

        aug_params = {"do_flip": do_flip, "speed_indices": speed_indices}
        return result, aug_params


def apply_landmark_augmentation(landmarks: np.ndarray, mask: np.ndarray, aug_params: dict):
    """
    Applies the same augmentation params used on RGB frames to landmarks.
    landmarks: [T, 75, 3]  mask: [T, 3]
    Returns synced (landmarks, mask) with same shape.
    """
    speed_indices = aug_params.get("speed_indices")
    do_flip = aug_params.get("do_flip", False)

    # 1. Speed jitter sync — resample same frame indices
    if speed_indices is not None:
        landmarks = landmarks[speed_indices]
        mask = mask[speed_indices]

    # 2. Horizontal flip sync
    if do_flip:
        # Swap left hand <-> right hand landmark blocks
        lh = landmarks[:, _LH_SLICE, :].copy()
        rh = landmarks[:, _RH_SLICE, :].copy()
        landmarks[:, _LH_SLICE, :] = rh
        landmarks[:, _RH_SLICE, :] = lh
        # Mirror x-coordinate (MediaPipe x is normalized 0-1, flip = 1 - x)
        landmarks[:, :, 0] = 1.0 - landmarks[:, :, 0]
        # Swap left/right hand validity mask bits
        lh_valid = mask[:, 0].copy()
        mask[:, 0] = mask[:, 1]
        mask[:, 1] = lh_valid

    return landmarks, mask
