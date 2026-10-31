"""Small compatibility helpers for automatic mixed precision (no deprecated APIs)."""
from __future__ import annotations

import torch


def amp_enabled(cfg, device) -> bool:
    return bool(cfg.train.amp) and device.type == "cuda"


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")
