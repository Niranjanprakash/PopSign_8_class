import torch
from pathlib import Path
from backend.config import FEATURE_VERSION, CLASSES

for split in ['train', 'val', 'test']:
    files = list(Path(f'dataset/processed/features/{split}').glob('*.pt'))
    sample = torch.load(files[0], map_location='cpu', weights_only=True)
    version_ok = sample.get('feature_version') == FEATURE_VERSION
    rgb_shape = tuple(sample['rgb'].shape)
    lm_shape  = tuple(sample['landmarks'].shape)
    mask_shape= tuple(sample['mask'].shape)
    print(f"{split}: {len(files)} files | version_ok={version_ok} | rgb={rgb_shape} | landmarks={lm_shape} | mask={mask_shape}")
