from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _to_uint8_grayscale(image):
    if image is None:
        raise ValueError("Image read failed.")
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype == np.uint16:
        image = (image / 257.0).astype(np.uint8)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def read_image(path):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return _to_uint8_grayscale(image)


def read_mask(path, threshold=127):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read mask: {path}")
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype == np.uint16 and image.max() <= 1:
        return (image > 0).astype(np.uint8)
    image = _to_uint8_grayscale(image)
    # Safety net: masks stored as {0,1} would be wiped out by a threshold of 127.
    if image.max() <= 1:
        threshold = 0
    return (image > threshold).astype(np.uint8)


def resize_pair(image, mask, img_size, interpolation=cv2.INTER_AREA):
    image = cv2.resize(image, (img_size, img_size), interpolation=interpolation)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)
    return image, mask


def preprocess_image(image_path, cfg):
    """Apply the configured inference preprocessing and return uint8 image plus original shape."""
    image = read_image(image_path)
    original_shape = image.shape
    image = cv2.resize(
        image,
        (int(cfg.dataset.img_size), int(cfg.dataset.img_size)),
        interpolation=cv2.INTER_AREA,
    )
    image = apply_percentile_clip(image, tuple(cfg.preprocessing.percentile_clip))
    if cfg.preprocessing.clahe.enabled:
        image = apply_clahe(
            image,
            True,
            cfg.preprocessing.clahe.clip_limit,
            cfg.preprocessing.clahe.tile_grid,
        )
    return image.astype(np.uint8), original_shape


def apply_percentile_clip(image, percentile_clip):
    low_pct, high_pct = percentile_clip
    non_zero = image[image > 0]
    if non_zero.size == 0:
        return image
    low = np.percentile(non_zero, low_pct)
    high = np.percentile(non_zero, high_pct)
    image = np.clip(image, low, high)
    scale = high - low
    if scale <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    image = ((image - low) / scale * 255.0).astype(np.uint8)
    return image


def apply_clahe(image, enabled, clip_limit=2.0, tile_grid=(8, 8)):
    if not enabled:
        return image
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
    return clahe.apply(image)


def preprocess_pair(image_path, mask_path, cfg):
    """Read, resize, percentile-clip and CLAHE one image/mask pair -> uint8 arrays (H, W)."""
    image, original_shape = preprocess_image(image_path, cfg)
    mask = read_mask(mask_path, threshold=int(cfg.dataset.mask_threshold))
    if mask.shape != original_shape:
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(
        mask,
        (int(cfg.dataset.img_size), int(cfg.dataset.img_size)),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = (mask > 0.5).astype(np.uint8)
    return image.astype(np.uint8), mask.astype(np.uint8)
