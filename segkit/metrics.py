from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dice_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    denom = np.sum(pred) + np.sum(target)
    return (2.0 * intersection + eps) / (denom + eps)


def iou_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - inter
    return (inter + eps) / (union + eps)


def precision_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1.0 - target))
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tp = np.sum(pred * target)
    fn = np.sum((1.0 - pred) * target)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tn = np.sum((1.0 - pred) * (1.0 - target))
    fp = np.sum(pred * (1.0 - target))
    return (tn + eps) / (tn + fp + eps)


# ---------------------------------------------------------------- from counts
def metrics_from_counts(tp, fp, fn, tn, eps=1e-6):
    """Vectorised metrics from confusion counts (works on scalars or arrays).

    Same formulas as the *_score functions above.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    tn = np.asarray(tn, dtype=np.float64)
    return {
        "dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
    }


# ---------------------------------------------------------------------- HD95
_STRUCT = ndi.generate_binary_structure(2, 1)


def _surface(mask):
    eroded = ndi.binary_erosion(mask, structure=_STRUCT, border_value=0)
    return mask & ~eroded


def hd95_score(pred, target, max_dist=None):
    """95th percentile symmetric Hausdorff distance (in pixels) for 2D binary masks.

    Conventions: both empty -> 0.0; exactly one empty -> `max_dist`
    (default: the image diagonal).
    """
    pred = np.squeeze(np.asarray(pred)).astype(bool)
    target = np.squeeze(np.asarray(target)).astype(bool)
    if max_dist is None:
        max_dist = float(np.hypot(*pred.shape[-2:]))
    p_any, t_any = bool(pred.any()), bool(target.any())
    if not p_any and not t_any:
        return 0.0
    if not p_any or not t_any:
        return float(max_dist)
    pred_surf = _surface(pred)
    targ_surf = _surface(target)
    dist_to_target = ndi.distance_transform_edt(~targ_surf)
    dist_to_pred = ndi.distance_transform_edt(~pred_surf)
    d_pt = dist_to_target[pred_surf]
    d_tp = dist_to_pred[targ_surf]
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))
