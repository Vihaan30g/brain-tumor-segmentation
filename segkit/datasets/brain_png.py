from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from segkit.config import resolve_path
from segkit.datasets.augmentations import augment_pair
from segkit.datasets.preprocessing import preprocess_pair

try:  # progress bar is optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class BrainPNGDataset(Dataset):
    def __init__(self, cfg, sample_names=None, split_name="train"):
        self.cfg = cfg
        self.root = resolve_path(cfg.dataset.root)
        self.images_dir = self.root / cfg.dataset.images_dir
        self.masks_dir = self.root / cfg.dataset.masks_dir
        self.img_size = int(cfg.dataset.img_size)
        self.mean = float(cfg.preprocessing.mean)
        self.std = float(cfg.preprocessing.std)
        self.cache_in_ram = bool(cfg.dataset.cache_in_ram)
        self.sample_names = list(sample_names) if sample_names is not None else []
        self.split_name = split_name
        self._images = None
        self._masks = None
        if self.cache_in_ram and self.sample_names:
            self._preload()

    def _load_pair(self, name):
        image_path = self.images_dir / name
        mask_path = self.masks_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        return preprocess_pair(str(image_path), str(mask_path), self.cfg)

    def _preload(self):
        """Preprocess every sample once, in the main process, into two contiguous arrays.

        DataLoader workers are forked afterwards and share these arrays (no per-worker copies).
        """
        n = len(self.sample_names)
        self._images = np.empty((n, self.img_size, self.img_size), dtype=np.uint8)
        self._masks = np.empty((n, self.img_size, self.img_size), dtype=np.uint8)
        iterator = range(n)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"Caching {self.split_name} ({n})", leave=False)
        for i in iterator:
            self._images[i], self._masks[i] = self._load_pair(self.sample_names[i])

    def __len__(self):
        return len(self.sample_names)

    def get_raw(self, idx):
        """Preprocessed (uint8 image, uint8 mask) without augmentation."""
        if self._images is not None:
            return self._images[idx], self._masks[idx]
        return self._load_pair(self.sample_names[idx])

    def __getitem__(self, idx):
        name = self.sample_names[idx]
        image, mask = self.get_raw(idx)
        if self.split_name == "train":
            image, mask = augment_pair(image, mask, self.cfg)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = mask.astype(np.float32)[None, :, :]
        image = image[None, :, :].astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "name": name,
        }
