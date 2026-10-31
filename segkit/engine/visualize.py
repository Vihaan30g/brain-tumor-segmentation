from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.utils.amp import autocast_ctx

_GREEN = np.array([0, 255, 0], dtype=np.float32)    # BGR
_RED = np.array([0, 0, 255], dtype=np.float32)
_BLUE = np.array([255, 0, 0], dtype=np.float32)
_STRIP = 24


def select_vis_indices(num_total, num_samples, seed, save_path=None):
    """Pick a fixed set of validation samples (reloaded on resume, so the same ones are shown every epoch)."""
    if save_path is not None and Path(save_path).exists():
        try:
            with Path(save_path).open("r", encoding="utf-8") as fh:
                saved = json.load(fh)["indices"]
            if saved and max(saved) < num_total:
                return [int(i) for i in saved]
        except (OSError, ValueError, KeyError):
            pass
    num_samples = max(1, min(int(num_samples), int(num_total)))
    rng = np.random.RandomState(int(seed))
    indices = sorted(int(i) for i in rng.choice(num_total, size=num_samples, replace=False))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(save_path).open("w", encoding="utf-8") as fh:
            json.dump({"indices": indices}, fh)
    return indices


def _blend(base_bgr, mask, color, alpha=0.4):
    out = base_bgr.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def _error_map(base_bgr, pred, gt):
    out = base_bgr.astype(np.float32).copy()
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    for m, color in ((tp, _GREEN), (fp, _RED), (fn, _BLUE)):
        out[m] = 0.35 * out[m] + 0.65 * color
    return out.clip(0, 255).astype(np.uint8)


def _label(panel, text):
    strip = np.full((_STRIP, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(strip, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def render_panel(image_norm, gt_mask, pred_mask, mean, std, title):
    """4-panel BGR figure: image | ground truth (green) | prediction (red) | error map (TP green, FP red, FN blue)."""
    gray = np.clip(image_norm * std + mean, 0.0, 1.0)
    gray = (gray * 255.0).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gt = gt_mask > 0.5
    pred = pred_mask > 0.5
    panels = [
        _label(base, title),
        _label(_blend(base, gt, _GREEN), "ground truth"),
        _label(_blend(base, pred, _RED), "prediction"),
        _label(_error_map(base, pred, gt), "TP green / FP red / FN blue"),
    ]
    return np.hstack(panels)


@torch.no_grad()
def save_epoch_samples(model, dataset, indices, device, epoch, samples_dir, cfg, amp=False, threshold=0.5):
    """Run inference on the fixed validation samples and save one PNG per sample.

    Returns (list of saved paths, vertically stacked RGB grid for TensorBoard).
    """
    was_training = model.training
    model.eval()
    samples = [dataset[i] for i in indices]
    images = torch.stack([s["image"] for s in samples]).to(device)
    with autocast_ctx(device, amp):
        logits = model(images)
    probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
    if was_training:
        model.train()

    mean, std = float(cfg.preprocessing.mean), float(cfg.preprocessing.std)
    out_dir = Path(samples_dir) / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, rows = [], []
    for k, (sample, prob) in enumerate(zip(samples, probs)):
        image = sample["image"].numpy()[0]
        gt = sample["mask"].numpy()[0] > 0.5
        pred = prob > threshold
        inter = float(np.logical_and(pred, gt).sum())
        denom = float(pred.sum() + gt.sum())
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        stem = Path(str(sample["name"])).stem
        panel = render_panel(image, gt, pred, mean, std, f"{stem} dice {dice:.3f}")
        path = out_dir / f"sample_{k}_{stem}_dice{dice:.3f}.png"
        cv2.imwrite(str(path), panel)
        paths.append(path)
        rows.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    return paths, np.vstack(rows)
