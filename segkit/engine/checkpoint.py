"""Crash-safe checkpointing.

Files in <run>/checkpoints/:
  last.pt       full training state after the most recently completed epoch
  last_prev.pt  the one before (fallback if last.pt is ever corrupted)
  best.pt       weights of the best validation epoch (never overwritten by a worse one)
Every write goes to a temp file first, is fsync'ed, and only then renamed into place.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def _torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(str(path), map_location="cpu")


def _fsync_dir(directory):
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    return tmp


def model_is_finite(model):
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    except Exception as exc:  # RNG restore is best effort
        print(f"[checkpoint] could not restore RNG state: {exc}")


def save_last(paths, state):
    """Write last.pt atomically, keeping the previous one as last_prev.pt."""
    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_save(state, paths.last)
    if paths.last.exists():
        os.replace(paths.last, paths.last_prev)
    os.replace(tmp, paths.last)
    _fsync_dir(paths.ckpt_dir)
    return paths.last


def save_best(paths, model, epoch, metrics, cfg_plain):
    payload = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "config": cfg_plain,
    }
    tmp = _atomic_save(payload, paths.best)
    os.replace(tmp, paths.best)
    _fsync_dir(paths.ckpt_dir)
    return paths.best


def load_last(paths):
    """Return (state, path_used) from the newest readable checkpoint, or (None, None).

    Order: last.pt.tmp (only exists if the process died between writing and renaming; it is
    complete if it loads), last.pt, last_prev.pt.
    """
    errors = []
    for candidate in (paths.last_tmp, paths.last, paths.last_prev):
        if not candidate.exists():
            continue
        try:
            state = _torch_load(candidate)
            if isinstance(state, dict) and "model_state" in state and "optimizer" in state and "epoch" in state:
                return state, candidate
            errors.append(f"{candidate.name}: not a training checkpoint")
        except Exception as exc:
            errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
    if errors:
        print("[checkpoint] could not load any resumable checkpoint:\n  " + "\n  ".join(errors))
    return None, None


def promote_tmp(paths):
    """If a complete last.pt.tmp was the newest state, make it the official last.pt."""
    if paths.last_tmp.exists():
        if paths.last.exists():
            os.replace(paths.last, paths.last_prev)
        os.replace(paths.last_tmp, paths.last)
        _fsync_dir(paths.ckpt_dir)


def cleanup_tmp(paths):
    for tmp in (paths.last_tmp, paths.best.with_name(paths.best.name + ".tmp")):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def load_model_weights(path, model, strict=True):
    """Load model weights from best.pt / last.pt. Returns the checkpoint dict."""
    ckpt = _torch_load(path)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt
