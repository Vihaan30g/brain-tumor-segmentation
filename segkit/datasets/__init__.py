from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from segkit.datasets.augmentations import worker_init_fn
from segkit.datasets.brain_png import BrainPNGDataset
from segkit.datasets.splits import ensure_split_file
from segkit.registry import Registry

DATASET_REGISTRY = Registry()
DATASET_REGISTRY.register("brain_png", BrainPNGDataset)


def get_dataset_class(name):
    return DATASET_REGISTRY.get(name)


def eval_batch_size(cfg):
    value = cfg.train.get("val_batch_size", None)
    return int(value) if value else int(cfg.train.batch_size)


def build_dataloaders(cfg, splits=("train", "val", "test")):
    """Build the requested loaders only (the dataset of each split is cached in RAM at creation)."""
    split = ensure_split_file(cfg)
    dataset_cls = get_dataset_class(cfg.dataset.name)
    num_workers = int(cfg.train.num_workers)
    pin = torch.cuda.is_available()
    loaders = {}

    if "train" in splits:
        train_ds = dataset_cls(cfg, sample_names=split["train"], split_name="train")
        generator = torch.Generator()
        generator.manual_seed(int(cfg.experiment.seed))
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
            drop_last=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )

    for name in ("val", "test"):
        if name in splits:
            ds = dataset_cls(cfg, sample_names=split[name], split_name=name)
            loaders[name] = DataLoader(
                ds,
                batch_size=eval_batch_size(cfg),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
                persistent_workers=num_workers > 0,
                drop_last=False,
                worker_init_fn=worker_init_fn,
            )
    return loaders
