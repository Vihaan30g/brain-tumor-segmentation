from __future__ import annotations

import random

import cv2
import numpy as np


def worker_init_fn(worker_id):
    """Seed numpy/random in every DataLoader worker from torch's per-worker seed
    (which is derived from the DataLoader generator, hence reproducible)."""
    import torch

    worker_seed = int(torch.initial_seed() % (2 ** 32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_affine(image, mask, cfg):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle = np.random.uniform(-float(cfg.augmentation.rotate_deg), float(cfg.augmentation.rotate_deg))
    scale = np.random.uniform(float(cfg.augmentation.scale[0]), float(cfg.augmentation.scale[1]))
    tx = np.random.uniform(-float(cfg.augmentation.translate_frac) * w, float(cfg.augmentation.translate_frac) * w)
    ty = np.random.uniform(-float(cfg.augmentation.translate_frac) * h, float(cfg.augmentation.translate_frac) * h)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    image = image.astype(np.uint8)
    mask = (mask > 0.5).astype(np.uint8)
    return image, mask


def random_elastic(image, mask, cfg):
    h, w = image.shape[:2]
    dx = np.random.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dy = np.random.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    sigma = float(cfg.augmentation.elastic_sigma)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dx *= float(cfg.augmentation.elastic_alpha)
    dy *= float(cfg.augmentation.elastic_alpha)
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    map_x = grid_x + dx
    map_y = grid_y + dy
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image.astype(np.uint8), (mask > 0.5).astype(np.uint8)


def apply_brightness_contrast(image, cfg):
    contrast_low, contrast_high = cfg.augmentation.contrast_range
    contrast = np.random.uniform(float(contrast_low), float(contrast_high))
    brightness = np.random.uniform(-float(cfg.augmentation.brightness_delta), float(cfg.augmentation.brightness_delta))
    image = np.clip((image.astype(np.float32) - 128.0) * contrast + 128.0 + brightness * 255.0, 0, 255).astype(np.uint8)
    return image


def apply_gamma(image, cfg):
    gamma = np.random.uniform(float(cfg.augmentation.gamma_range[0]), float(cfg.augmentation.gamma_range[1]))
    image = np.clip(np.power(image.astype(np.float32) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    return image


def apply_noise(image, cfg):
    noise_std = float(cfg.augmentation.noise_std)
    noise = np.random.normal(0.0, noise_std, image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise * 255.0, 0, 255).astype(np.uint8)
    return image


def augment_pair(image, mask, cfg):
    if np.random.rand() < float(cfg.augmentation.hflip_p):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if np.random.rand() < float(cfg.augmentation.vflip_p):
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    if np.random.rand() < float(cfg.augmentation.affine_p):
        image, mask = random_affine(image, mask, cfg)
    if np.random.rand() < float(cfg.augmentation.elastic_p):
        image, mask = random_elastic(image, mask, cfg)
    if np.random.rand() < float(cfg.augmentation.brightness_contrast_p):
        image = apply_brightness_contrast(image, cfg)
    if np.random.rand() < float(cfg.augmentation.gamma_p):
        image = apply_gamma(image, cfg)
    if np.random.rand() < float(cfg.augmentation.noise_p):
        image = apply_noise(image, cfg)
    mask = (mask > 0.5).astype(np.uint8)
    return image.astype(np.uint8), mask.astype(np.uint8)
