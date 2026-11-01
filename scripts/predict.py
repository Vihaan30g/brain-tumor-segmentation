from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segkit.config import load_config, resolve_path
from segkit.datasets.preprocessing import preprocess_image, read_image
from segkit.engine.checkpoint import load_model_weights
from segkit.models import build_model
from segkit.utils.amp import amp_enabled, autocast_ctx
from segkit.utils.logging import RunPaths, find_latest_run


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a tumor mask for one MRI PNG image.")
    parser.add_argument("--image", required=True, help="Image path, or a filename inside dataset.images_dir.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "transunet_dataset1.yaml"))
    parser.add_argument("--run-dir", help="Run directory; defaults to the latest run with checkpoints.")
    parser.add_argument("--checkpoint", help="Checkpoint file; overrides --run-dir.")
    parser.add_argument("--output-dir", help="Directory for the predicted mask and visualization.")
    parser.add_argument("--threshold", type=float, help="Probability threshold; defaults to train.val_threshold.")
    return parser.parse_args()


def resolve_image_path(image_arg, cfg):
    requested = Path(image_arg).expanduser()
    dataset_images = resolve_path(cfg.dataset.root) / cfg.dataset.images_dir
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend((ROOT / requested, dataset_images / requested))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {image_arg}")


def resolve_model_checkpoint(args, cfg):
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint.resolve(), checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = Path.cwd() / run_dir
        checkpoint = RunPaths(run_dir).best
    else:
        run_dir = find_latest_run(resolve_path(cfg.experiment.output_root), cfg.experiment.name)
        if run_dir is None:
            raise FileNotFoundError("No trained run found. Pass --checkpoint or --run-dir.")
        checkpoint = RunPaths(run_dir).best

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    return checkpoint.resolve(), run_dir.resolve()


def save_prediction_visualization(original_image, prediction, output_path):
    gray = cv2.resize(
        original_image,
        (prediction.shape[1], prediction.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = prediction > 0
    overlay = base.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * red[mask]).astype(np.uint8)

    panels = []
    for panel, label in ((base, "Input"), (cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR), "Predicted mask"), (overlay, "Overlay")):
        labeled = panel.copy()
        cv2.putText(labeled, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(labeled)
    if not cv2.imwrite(str(output_path), np.hstack(panels)):
        raise OSError(f"Could not save prediction visualization: {output_path}")


@torch.inference_mode()
def predict(image_path, checkpoint_path, output_dir, cfg, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_u8, original_shape = preprocess_image(image_path, cfg)
    image = image_u8.astype(np.float32) / 255.0
    image = (image - float(cfg.preprocessing.mean)) / float(cfg.preprocessing.std)
    tensor = torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)

    model = build_model(cfg, pretrained=False, resnet_pretrained=False)
    checkpoint = load_model_weights(checkpoint_path, model)
    model.to(device).eval()
    with autocast_ctx(device, amp_enabled(cfg, device)):
        logits = model(tensor)
    probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    mask = (probability >= threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask_path = output_dir / f"{stem}_mask.png"
    visualization_path = output_dir / f"{stem}_prediction.png"
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Could not save predicted mask: {mask_path}")
    original_image = read_image(image_path)
    save_prediction_visualization(original_image, mask, visualization_path)

    print(f"Checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"Device: {device}; threshold: {threshold:.3f}")
    print(f"Predicted foreground: {int(np.count_nonzero(mask))} pixels")
    print(f"Mask saved to: {mask_path}")
    print(f"Visualization saved to: {visualization_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    image_path = resolve_image_path(args.image, cfg)
    checkpoint_path, run_dir = resolve_model_checkpoint(args, cfg)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir / "predictions"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    threshold = float(cfg.train.val_threshold if args.threshold is None else args.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}.")
    predict(image_path, checkpoint_path, output_dir, cfg, threshold)


if __name__ == "__main__":
    main()
