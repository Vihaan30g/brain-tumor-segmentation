from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from segkit.config import load_config, resolve_path
from segkit.datasets import build_dataloaders
from segkit.engine.checkpoint import load_model_weights
from segkit.engine.evaluate import evaluate_model
from segkit.engine.visualize import render_panel
from segkit.models import build_model
from segkit.utils.amp import amp_enabled, autocast_ctx
from segkit.utils.logging import RunPaths, find_latest_run


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TransUNet checkpoint on the held-out split.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "transunet_dataset1.yaml"))
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory to evaluate (uses its checkpoints/best.pt). Default: the latest run.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint file (overrides --run-dir).")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--num-vis", type=int, default=10, help="Number of overlay images to save.")
    parser.add_argument("--no-hd95", action="store_true", help="Skip HD95 (faster).")
    return parser.parse_args()


def resolve_checkpoint(args, cfg):
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser()
        return ckpt, (ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
    else:
        run_dir = find_latest_run(resolve_path(cfg.experiment.output_root), cfg.experiment.name)
        if run_dir is None:
            raise FileNotFoundError("No run with checkpoints found. Train first or pass --checkpoint / --run-dir.")
    ckpt = RunPaths(run_dir).best
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} does not exist.")
    return ckpt, run_dir


@torch.no_grad()
def save_overlays(model, dataset, indices, device, out_dir, cfg, amp, threshold):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    mean, std = float(cfg.preprocessing.mean), float(cfg.preprocessing.std)
    for i in indices:
        sample = dataset[i]
        with autocast_ctx(device, amp):
            logits = model(sample["image"][None].to(device))
        pred = (torch.sigmoid(logits.float())[0, 0] > threshold).cpu().numpy()
        gt = sample["mask"].numpy()[0] > 0.5
        dice = (2.0 * np.logical_and(pred, gt).sum() + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        stem = Path(sample["name"]).stem
        panel = render_panel(sample["image"].numpy()[0], gt, pred, mean, std, f"{stem} dice {dice:.3f}")
        cv2.imwrite(str(out_dir / f"{stem}_dice{dice:.3f}.png"), panel)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path, run_dir = resolve_checkpoint(args, cfg)

    loaders = build_dataloaders(cfg, splits=(args.split,))
    loader = loaders[args.split]
    model = build_model(cfg, pretrained=False)  # weights come from the checkpoint; no downloads
    ckpt = load_model_weights(ckpt_path, model)
    model.to(device)
    amp = amp_enabled(cfg, device)
    threshold = float(cfg.train.val_threshold)
    print(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')}, saved val metrics: "
          f"dice={ckpt.get('metrics', {}).get('dice', float('nan')):.4f})")

    metrics, details = evaluate_model(model, loader, device, threshold=threshold, amp=amp,
                                      compute_hd95=not args.no_hd95, return_details=True)

    out_dir = Path(run_dir) / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"), "split": args.split, "threshold": threshold, **metrics}
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    keys = [k for k in ("dice", "iou", "precision", "recall", "hd95") if k in details]
    with (out_dir / "per_slice.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name"] + keys)
        for i, name in enumerate(details["names"]):
            writer.writerow([name] + [f"{details[k][i]:.6f}" for k in keys])

    if args.num_vis > 0:
        n = len(loader.dataset)
        idx = sorted(int(i) for i in np.random.RandomState(int(cfg.experiment.seed)).choice(n, size=min(args.num_vis, n), replace=False))
        save_overlays(model, loader.dataset, idx, device, out_dir / "overlays", cfg, amp, threshold)

    print(f"\n{args.split} results ({metrics['num_slices']} slices)")
    print("  pooled      : " + "  ".join(f"{k}={metrics[k]:.4f}" for k in ("dice", "iou", "precision", "recall", "specificity")))
    print("  per-slice   : " + "  ".join(f"{k}={metrics[k + '_slice']:.4f}" for k in ("dice", "iou", "precision", "recall")))
    if "hd95" in metrics:
        print(f"  HD95 (mean per slice): {metrics['hd95']:.2f} px")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
