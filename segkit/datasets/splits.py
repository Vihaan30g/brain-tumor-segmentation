from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from segkit.config import resolve_path


def _sort_key(name):
    stem = Path(name).stem
    return (0, int(stem), name) if stem.isdigit() else (1, 0, name)


def list_image_names(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {images_dir}")
    files = [p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(files, key=_sort_key)


def validate_split_inputs(images_dir, masks_dir):
    image_names = set(list_image_names(images_dir))
    mask_names = set(list_image_names(masks_dir))
    if not image_names:
        raise FileNotFoundError(f"No .png images found in {images_dir}")
    missing = sorted(image_names - mask_names, key=_sort_key)
    if missing:
        raise FileNotFoundError(f"Missing masks for {len(missing)} images, e.g. {missing[:5]}")
    missing_masks = sorted(mask_names - image_names, key=_sort_key)
    if missing_masks:
        raise FileNotFoundError(f"Missing images for {len(missing_masks)} masks, e.g. {missing_masks[:5]}")


def _dirs(cfg):
    root = resolve_path(cfg.dataset.root)
    return root / cfg.dataset.images_dir, root / cfg.dataset.masks_dir


def split_file_path(cfg):
    return resolve_path(cfg.dataset.split_file)


def build_split_file(cfg):
    images_dir, masks_dir = _dirs(cfg)
    split_file = split_file_path(cfg)
    split_file.parent.mkdir(parents=True, exist_ok=True)

    validate_split_inputs(images_dir, masks_dir)
    names = list_image_names(images_dir)
    rng = np.random.RandomState(int(cfg.experiment.seed))
    rng.shuffle(names)
    n_total = len(names)
    n_train = int(cfg.dataset.split_ratio[0] * n_total)
    n_val = int(cfg.dataset.split_ratio[1] * n_total)
    n_test = n_total - n_train - n_val
    split = {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:n_train + n_val + n_test],
        "seed": int(cfg.experiment.seed),
    }
    tmp = split_file.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    tmp.replace(split_file)
    return split


def _check_split(split, images_dir):
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    if (train & val) or (train & test) or (val & test):
        raise RuntimeError("Split file is invalid: train/val/test overlap.")
    present = set(list_image_names(images_dir))
    missing = (train | val | test) - present
    if missing:
        raise RuntimeError(
            f"Split file references {len(missing)} files that are not in {images_dir} "
            f"(e.g. {sorted(missing)[:3]}). Wrong dataset.root, or delete the split file to regenerate it."
        )


def ensure_split_file(cfg):
    """Load the split file if it exists (never overwrite it), otherwise create it."""
    images_dir, _ = _dirs(cfg)
    split_file = split_file_path(cfg)
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        _check_split(data, images_dir)
        return data
    return build_split_file(cfg)
