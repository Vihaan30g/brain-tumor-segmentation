from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from segkit.config import load_config, resolve_path
from segkit.datasets.preprocessing import preprocess_pair, read_image, read_mask
from segkit.datasets.splits import ensure_split_file, list_image_names, validate_split_inputs


def main():
    parser = argparse.ArgumentParser(description="Sanity-check the dataset folder, masks and split.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "transunet_dataset1.yaml"))
    parser.add_argument("--max-files", type=int, default=0, help="Only inspect the first N files (0 = all).")
    args = parser.parse_args()
    cfg = load_config(args.config)

    root = resolve_path(cfg.dataset.root)
    images_dir = root / cfg.dataset.images_dir
    masks_dir = root / cfg.dataset.masks_dir
    print(f"Dataset root : {root}")
    validate_split_inputs(images_dir, masks_dir)
    names = list_image_names(images_dir)
    if args.max_files > 0:
        names = names[: args.max_files]
    print(f"Found {len(names)} image/mask pairs")

    shapes, raw_mask_values, empty, pos_frac, sizes_mismatch = set(), set(), 0, [], 0
    for name in names:
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(str(masks_dir / name), cv2.IMREAD_UNCHANGED)
        shapes.add(tuple(img.shape))
        if img.shape[:2] != msk.shape[:2]:
            sizes_mismatch += 1
        raw_mask_values.update(np.unique(msk).tolist()[:6])
        m = read_mask(masks_dir / name, int(cfg.dataset.mask_threshold))
        if m.sum() == 0:
            empty += 1
        pos_frac.append(float(m.mean()))
    print(f"Image shapes        : {sorted(shapes)[:5]}{' ...' if len(shapes) > 5 else ''}")
    print(f"Raw mask values     : {sorted(raw_mask_values)[:10]}")
    print(f"Image/mask size mismatch: {sizes_mismatch}")
    print(f"Empty masks         : {empty}")
    print(f"Tumor area fraction : mean={np.mean(pos_frac):.4f} min={np.min(pos_frac):.4f} max={np.max(pos_frac):.4f}")

    image, mask = preprocess_pair(str(images_dir / names[0]), str(masks_dir / names[0]), cfg)
    print(f"Preprocessed sample : image {image.shape} {image.dtype} range [{image.min()}, {image.max()}], mask sum={int(mask.sum())}")

    split = ensure_split_file(cfg)
    print(f"Split OK: {len(split['train'])} train, {len(split['val'])} val, {len(split['test'])} test")
    print("NOTE: the split is random at image level; if several slices come from the same patient, "
          "val/test scores can be optimistic.")


if __name__ == "__main__":
    main()
