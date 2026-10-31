from __future__ import annotations

import os
import sys
from importlib import metadata


def _configure_cuda_allocator():
    """Less GPU memory fragmentation (must be set before torch initialises CUDA)."""
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ or sys.platform.startswith("win"):
        return
    try:
        major, minor = (int(x) for x in metadata.version("torch").split("+")[0].split(".")[:2])
    except Exception:
        return
    if (major, minor) >= (2, 1):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


_configure_cuda_allocator()

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segkit.config import load_config
from segkit.datasets import build_dataloaders
from segkit.engine.trainer import train_model
from segkit.models import build_model
from segkit.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TransUNet brain tumor segmenter.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "transunet_dataset1.yaml"),
                        help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default="auto",
                        help="'auto' (default): continue the latest unfinished run if there is one, otherwise start a new run. "
                             "Or give a run directory / checkpoint path to resume that run.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing runs and start a brand-new run directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.experiment.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is not available, training on CPU will be extremely slow.")

    loaders = build_dataloaders(cfg, splits=("train", "val"))
    model = build_model(cfg)
    model.to(device)
    result = train_model(cfg, model, loaders["train"], loaders["val"], device, resume=args.resume, fresh=args.fresh)
    if result.get("interrupted"):
        sys.exit(130)


if __name__ == "__main__":
    main()
