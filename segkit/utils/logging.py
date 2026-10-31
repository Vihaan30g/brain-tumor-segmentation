from __future__ import annotations

import csv
import datetime
import logging
import re
import sys
from pathlib import Path

_STAMP_RE = re.compile(r"^\d{8}_\d{6}$")

CSV_FIELDS = [
    "epoch", "global_step", "lr_encoder", "lr_decoder", "train_loss",
    "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_specificity",
    "val_dice_slice", "val_iou_slice", "val_hd95",
    "epoch_time_sec", "gpu_peak_mem_gb", "is_best",
]


class RunPaths:
    """All files of one run live under runs/<experiment>/<timestamp>/."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.tb_dir = self.run_dir / "tensorboard"
        self.samples_dir = self.run_dir / "samples"
        self.logs_dir = self.run_dir / "logs"
        self.metrics_csv = self.run_dir / "metrics.csv"
        self.vis_indices = self.run_dir / "vis_indices.json"

    @property
    def last(self):
        return self.ckpt_dir / "last.pt"

    @property
    def last_prev(self):
        return self.ckpt_dir / "last_prev.pt"

    @property
    def last_tmp(self):
        return self.ckpt_dir / "last.pt.tmp"

    @property
    def best(self):
        return self.ckpt_dir / "best.pt"

    def make(self):
        for d in (self.run_dir, self.ckpt_dir, self.tb_dir, self.samples_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def setup_run_dir(output_root, experiment_name):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / experiment_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run(output_root, experiment_name, require_checkpoint=True):
    """Newest runs/<experiment>/<timestamp> directory (optionally one that has a checkpoint)."""
    base = Path(output_root) / experiment_name
    if not base.is_dir():
        return None
    candidates = sorted(p for p in base.iterdir() if p.is_dir() and _STAMP_RE.match(p.name))
    for run_dir in reversed(candidates):
        paths = RunPaths(run_dir)
        if not require_checkpoint:
            return run_dir
        if paths.last.exists() or paths.last_prev.exists() or paths.last_tmp.exists() or paths.best.exists():
            return run_dir
    return None


def get_logger(log_file=None, name="segkit"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(fh)
    return logger


def write_metrics_csv(rows, path):
    """(Re)write metrics.csv from the in-memory history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_FIELDS})
    tmp.replace(path)
