from __future__ import annotations

import numpy as np
import torch

from segkit.metrics import hd95_score, metrics_from_counts
from segkit.utils.amp import autocast_ctx


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    threshold=0.5,
    criterion=None,
    amp=False,
    compute_hd95=False,
    return_details=False,
):
    """Evaluate on a dataloader.

    Returns a dict with
      dice / iou / precision / recall / specificity : pooled over all pixels of all slices
      *_slice                                      : mean of the per-slice values
      hd95                                         : mean per-slice HD95 in pixels (if compute_hd95)
      loss                                         : mean loss (if criterion given)
    With return_details=True also returns per-slice arrays and file names.
    """
    was_training = model.training
    model.eval()

    counts = []
    names = []
    hd95_values = []
    loss_sum = 0.0
    n_seen = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits = model(images)
        logits = logits.float()
        if criterion is not None:
            loss_sum += float(criterion(logits, masks)) * images.shape[0]
        n_seen += images.shape[0]

        pred = torch.sigmoid(logits) > threshold
        gt = masks > 0.5
        tp = (pred & gt).flatten(1).sum(1)
        fp = (pred & ~gt).flatten(1).sum(1)
        fn = (~pred & gt).flatten(1).sum(1)
        tn = (~pred & ~gt).flatten(1).sum(1)
        counts.append(torch.stack([tp, fp, fn, tn], dim=1).cpu().numpy().astype(np.float64))
        if compute_hd95:
            pred_np = pred[:, 0].cpu().numpy()
            gt_np = gt[:, 0].cpu().numpy()
            hd95_values.extend(hd95_score(pred_np[i], gt_np[i]) for i in range(pred_np.shape[0]))
        batch_names = batch.get("name") if isinstance(batch, dict) else None
        if batch_names is not None:
            names.extend(list(batch_names))

    if was_training:
        model.train()

    if not counts:
        raise RuntimeError("Evaluation dataloader is empty.")
    counts = np.concatenate(counts, axis=0)  # (N, 4): tp, fp, fn, tn
    pooled = metrics_from_counts(*counts.sum(axis=0))
    per_slice = metrics_from_counts(counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3])

    metrics = {k: float(v) for k, v in pooled.items()}
    for k, v in per_slice.items():
        metrics[f"{k}_slice"] = float(np.mean(v))
    if compute_hd95:
        metrics["hd95"] = float(np.mean(hd95_values))
    if criterion is not None:
        metrics["loss"] = loss_sum / max(1, n_seen)
    metrics["num_slices"] = int(counts.shape[0])

    if return_details:
        details = {"names": names, **{k: np.asarray(v) for k, v in per_slice.items()}}
        if compute_hd95:
            details["hd95"] = np.asarray(hd95_values)
        return metrics, details
    return metrics
