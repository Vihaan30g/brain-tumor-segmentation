from __future__ import annotations

import json
import math
import signal
import time
from pathlib import Path

import torch

from segkit.config import resolve_path, save_config_copy, to_plain
from segkit.engine.checkpoint import (
    capture_rng_state,
    cleanup_tmp,
    load_last,
    promote_tmp,
    model_is_finite,
    restore_rng_state,
    save_best,
    save_last,
)
from segkit.engine.evaluate import evaluate_model
from segkit.engine.visualize import save_epoch_samples, select_vis_indices
from segkit.losses import BCEWithDiceLoss
from segkit.utils.amp import amp_enabled, autocast_ctx, make_grad_scaler
from segkit.utils.logging import RunPaths, find_latest_run, get_logger, setup_run_dir, write_metrics_csv
from segkit.utils.seed import set_seed

MAX_CONSECUTIVE_NONFINITE = 50


# --------------------------------------------------------------------------- optimiser / lr
def is_encoder_param(name):
    return name.startswith("vit.") or name.startswith("resnet.")


def build_optimizer(model, cfg):
    """AdamW with two groups: pretrained backbones (`vit.*`, `resnet.*`) and everything else."""
    lr_enc = float(cfg.train.lr_encoder)
    lr_dec = float(cfg.train.lr_decoder)
    wd = float(cfg.train.weight_decay)
    enc, dec = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (enc if is_encoder_param(name) else dec).append(parameter)
    groups = []
    if enc:
        groups.append({"params": enc, "lr": lr_enc, "base_lr": lr_enc, "weight_decay": wd, "group_name": "encoder"})
    if dec:
        groups.append({"params": dec, "lr": lr_dec, "base_lr": lr_dec, "weight_decay": wd, "group_name": "decoder"})
    return torch.optim.AdamW(groups, betas=tuple(cfg.train.betas), weight_decay=wd)


def compute_lr(base_lr, step, warmup_steps, total_steps, min_lr):
    """Linear warm-up, then cosine decay to min_lr. Pure function of the global step (resume-safe)."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def group_lr(optimizer, name):
    for g in optimizer.param_groups:
        if g.get("group_name") == name:
            return g["lr"]
    return None


# --------------------------------------------------------------------------- tensorboard
class _NullWriter:
    def __getattr__(self, _):
        return lambda *a, **k: None


def make_writer(tb_dir, purge_step, logger):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # tensorboard not installed
        logger.info(f"[warn] TensorBoard unavailable ({exc}); continuing without it.")
        return _NullWriter()
    kwargs = {"log_dir": str(tb_dir), "flush_secs": 30}
    if purge_step is not None:
        kwargs["purge_step"] = int(purge_step)
    return SummaryWriter(**kwargs)


# --------------------------------------------------------------------------- run directory / resume
def _paths_from_resume_arg(resume):
    p = Path(resume).expanduser()
    if p.is_dir():
        return RunPaths(p if (p / "checkpoints").is_dir() else p.parent)
    if p.is_file():
        return RunPaths(p.parent.parent if p.parent.name == "checkpoints" else p.parent)
    raise FileNotFoundError(f"--resume path not found: {resume}")


def _prepare_run(cfg, resume, fresh):
    """Returns (RunPaths, resume_state or None, path the state was read from)."""
    output_root = resolve_path(cfg.experiment.output_root)
    name = cfg.experiment.name

    if fresh:
        return RunPaths(setup_run_dir(output_root, name)).make(), None, None

    if resume not in (None, "auto"):
        paths = _paths_from_resume_arg(resume)
        state, src = load_last(paths)
        if state is None:
            raise RuntimeError(f"No readable checkpoint found for --resume {resume} (looked in {paths.ckpt_dir}).")
        return paths.make(), state, src

    latest = find_latest_run(output_root, name)
    if latest is not None:
        paths = RunPaths(latest)
        state, src = load_last(paths)
        if state is None:
            raise RuntimeError(
                f"Found previous run {latest} but none of its checkpoints could be read.\n"
                f"Fix/remove it, or start a new run with:  python scripts/train.py --fresh"
            )
        if state.get("finished", False):
            print(f"[resume] latest run {latest.name} already finished ({state.get('stop_reason')}); starting a new run.")
            return RunPaths(setup_run_dir(output_root, name)).make(), None, None
        return paths.make(), state, src
    return RunPaths(setup_run_dir(output_root, name)).make(), None, None


def _config_warnings(cfg, state):
    old = state.get("config")
    if not old:
        return []
    new = to_plain(cfg)
    out = []
    for section, keys in (("model", None), ("loss", None), ("train", ("batch_size", "lr_encoder", "lr_decoder", "weight_decay", "warmup_epochs", "min_lr", "amp", "grad_clip_norm")), ("dataset", ("img_size", "split_file", "split_ratio")), ("preprocessing", None)):
        a, b = old.get(section, {}), new.get(section, {})
        ks = keys if keys is not None else sorted(set(a) | set(b))
        for k in ks:
            if a.get(k) != b.get(k):
                out.append(f"{section}.{k}: {a.get(k)!r} (checkpoint) -> {b.get(k)!r} (now)")
    return out


class _Stop(KeyboardInterrupt):
    pass


def _raise_on_sigterm(signum, frame):
    raise _Stop()


# --------------------------------------------------------------------------- training
def train_model(cfg, model, train_loader, val_loader, device, resume="auto", fresh=False):
    seed = int(cfg.experiment.seed)
    set_seed(seed)

    paths, state, state_src = _prepare_run(cfg, resume, fresh)
    logger = get_logger(paths.logs_dir / "train.log")
    resumed = state is not None
    if resumed:
        if state_src == paths.last_tmp:
            promote_tmp(paths)
        cleanup_tmp(paths)

    total_epochs = int(cfg.train.epochs)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Training loader is empty (dataset smaller than batch_size with drop_last=True?).")
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(cfg.train.warmup_epochs) * steps_per_epoch
    min_lr = float(cfg.train.min_lr)
    use_amp = amp_enabled(cfg, device)
    grad_clip = float(cfg.train.grad_clip_norm)
    threshold = float(cfg.train.val_threshold)
    log_every = max(1, int(cfg.train.log_every_n_iters))
    hd95_every = int(cfg.train.get("hd95_every_n_epochs", 0) or 0)
    patience = int(cfg.train.early_stopping_patience)
    viz_every = int(cfg.viz.get("every_n_epochs", 1) or 1)

    optimizer = build_optimizer(model, cfg)
    scaler = make_grad_scaler(use_amp)
    criterion = BCEWithDiceLoss(
        bce_weight=float(cfg.loss.bce_weight),
        dice_weight=float(cfg.loss.dice_weight),
        dice_smooth=float(cfg.loss.dice_smooth),
    )

    start_epoch, global_step = 0, 0
    best_score, best_epoch, patience_counter = -1.0, 0, 0
    history = []
    if resumed:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        if use_amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        global_step = int(state.get("global_step", start_epoch * steps_per_epoch))
        best_score = float(state.get("best_score", -1.0))
        best_epoch = int(state.get("best_epoch", 0))
        patience_counter = int(state.get("patience_counter", 0))
        history = list(state.get("history", []))
        restore_rng_state(state.get("rng"))
        # honour the *current* config for learning rates / weight decay (the optimizer state keeps the momentum buffers)
        for g in optimizer.param_groups:
            g["base_lr"] = float(cfg.train.lr_encoder) if g.get("group_name") == "encoder" else float(cfg.train.lr_decoder)
            g["weight_decay"] = float(cfg.train.weight_decay)
        for line in _config_warnings(cfg, state):
            logger.info(f"[warn] config changed since checkpoint: {line}")
        del state
        save_config_copy(cfg, paths.run_dir, f"config_resumed_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
    else:
        save_config_copy(cfg, paths.run_dir)

    writer = make_writer(paths.tb_dir, global_step + 1 if resumed else None, logger)
    if not resumed:
        import yaml
        writer.add_text("config", "<pre>" + yaml.safe_dump(to_plain(cfg), sort_keys=False) + "</pre>", 0)

    # fixed validation samples for the per-epoch visual results
    vis_indices = []
    if val_loader is not None and int(cfg.viz.num_samples) > 0:
        vis_indices = select_vis_indices(len(val_loader.dataset), cfg.viz.num_samples, seed, paths.vis_indices)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("=" * 78)
    logger.info(f"Run dir        : {paths.run_dir}")
    logger.info(f"Device         : {device} | AMP: {use_amp} | params: {n_params / 1e6:.1f}M")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        logger.info(f"GPU            : {torch.cuda.get_device_name(device)} ({total:.1f} GB)")
    logger.info(f"Train / val    : {len(train_loader.dataset)} / {len(val_loader.dataset) if val_loader is not None else 0} samples")
    logger.info(f"Iters/epoch    : {steps_per_epoch} | epochs: {total_epochs} | warmup: {cfg.train.warmup_epochs} epochs")
    if resumed:
        logger.info(f"RESUMING from  : {state_src.name} -> continuing at epoch {start_epoch + 1}/{total_epochs} "
                    f"(best val_dice so far {best_score:.4f} @ epoch {best_epoch})")
    else:
        logger.info("Starting a NEW run")
    logger.info(f"TensorBoard    : tensorboard --logdir {resolve_path(cfg.experiment.output_root)} --port 6006")
    logger.info("=" * 78)

    old_sigterm = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    interrupted = False
    stop_reason = None
    epoch_times = []
    nonfinite_streak = 0
    completed_epochs = start_epoch

    try:
        if start_epoch >= total_epochs:
            logger.info("All epochs are already completed in this run; nothing to do.")
            stop_reason = "completed"

        for epoch in range(start_epoch, total_epochs):
            t_epoch = time.time()
            model.train()
            if train_loader.generator is not None:
                train_loader.generator.manual_seed(seed + epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            loss_sum = torch.zeros((), device=device)
            loss_count = 0
            win_sum = torch.zeros((), device=device)
            win_count = 0
            t_win = time.time()
            seen_in_win = 0

            for iter_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)

                for g in optimizer.param_groups:
                    g["lr"] = compute_lr(g["base_lr"], global_step, warmup_steps, total_steps, min_lr)

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(device, use_amp):
                    logits = model(images)
                loss = criterion(logits, targets)

                if not torch.isfinite(loss):
                    nonfinite_streak += 1
                    logger.info(f"[warn] non-finite loss at epoch {epoch + 1} iter {iter_idx + 1}; batch skipped "
                                f"({nonfinite_streak} in a row)")
                    if nonfinite_streak >= MAX_CONSECUTIVE_NONFINITE:
                        raise RuntimeError("Loss is NaN/Inf for too many consecutive batches. Training aborted; "
                                           "last.pt still holds the last good epoch (resume with lower LR).")
                    continue
                nonfinite_streak = 0

                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                detached = loss.detach()
                loss_sum += detached
                win_sum += detached
                loss_count += 1
                win_count += 1
                seen_in_win += images.shape[0]

                if (iter_idx + 1) % log_every == 0:
                    avg = float(win_sum) / max(1, win_count)
                    speed = seen_in_win / max(1e-6, time.time() - t_win)
                    mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
                    logger.info(
                        f"epoch={epoch + 1}/{total_epochs} iter={iter_idx + 1}/{steps_per_epoch} loss={avg:.4f} "
                        f"lr_enc={group_lr(optimizer, 'encoder'):.2e} lr_dec={group_lr(optimizer, 'decoder'):.2e} "
                        f"{speed:.1f} img/s gpu_peak={mem:.1f}GB"
                    )
                    writer.add_scalar("train/loss_iter", avg, global_step)
                    win_sum.zero_()
                    win_count = 0
                    seen_in_win = 0
                    t_win = time.time()

            train_loss = float(loss_sum) / max(1, loss_count)
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0

            # ------------------------------------------------------------- validation
            metrics = {}
            if val_loader is not None:
                do_hd95 = hd95_every > 0 and ((epoch + 1) % hd95_every == 0)
                metrics = evaluate_model(model, val_loader, device, threshold=threshold, criterion=criterion,
                                         amp=use_amp, compute_hd95=do_hd95)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            val_dice = metrics.get("dice")
            is_best = val_dice is not None and val_dice > best_score
            if val_dice is not None:
                if is_best:
                    best_score, best_epoch, patience_counter = val_dice, epoch + 1, 0
                else:
                    patience_counter += 1

            epoch_time = time.time() - t_epoch
            epoch_times.append(epoch_time)
            completed_epochs = epoch + 1

            row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "lr_encoder": group_lr(optimizer, "encoder"),
                "lr_decoder": group_lr(optimizer, "decoder"),
                "train_loss": train_loss,
                "val_loss": metrics.get("loss"),
                "val_dice": metrics.get("dice"),
                "val_iou": metrics.get("iou"),
                "val_precision": metrics.get("precision"),
                "val_recall": metrics.get("recall"),
                "val_specificity": metrics.get("specificity"),
                "val_dice_slice": metrics.get("dice_slice"),
                "val_iou_slice": metrics.get("iou_slice"),
                "val_hd95": metrics.get("hd95"),
                "epoch_time_sec": round(epoch_time, 1),
                "gpu_peak_mem_gb": round(peak_gb, 2),
                "is_best": int(is_best),
            }
            history.append(row)
            write_metrics_csv(history, paths.metrics_csv)

            e = epoch + 1
            writer.add_scalar("epoch/train_loss", train_loss, e)
            writer.add_scalar("epoch/lr_encoder", row["lr_encoder"], e)
            writer.add_scalar("epoch/lr_decoder", row["lr_decoder"], e)
            writer.add_scalar("epoch/time_sec", epoch_time, e)
            writer.add_scalar("epoch/gpu_peak_mem_gb", peak_gb, e)
            for key in ("loss", "dice", "iou", "precision", "recall", "specificity", "dice_slice", "iou_slice", "hd95"):
                if metrics.get(key) is not None:
                    writer.add_scalar(f"val/{key}", metrics[key], e)

            # ------------------------------------------------------------- visual results
            if vis_indices and (e % viz_every == 0):
                try:
                    _, grid = save_epoch_samples(model, val_loader.dataset, vis_indices, device, e,
                                                 paths.samples_dir, cfg, amp=use_amp, threshold=threshold)
                    writer.add_image("val/samples", grid, e, dataformats="HWC")
                except Exception as exc:  # never lose a training run because of a plotting problem
                    logger.info(f"[warn] could not save epoch samples: {type(exc).__name__}: {exc}")
            writer.flush()

            # ------------------------------------------------------------- stop decision + checkpoints
            early_stop = patience > 0 and patience_counter >= patience
            finished = early_stop or (e == total_epochs)
            if early_stop:
                stop_reason = f"early stopping (no val_dice improvement for {patience} epochs)"
            elif e == total_epochs:
                stop_reason = "completed all epochs"

            if not model_is_finite(model):
                raise RuntimeError("Model weights contain NaN/Inf; refusing to overwrite checkpoints. "
                                   "Resume from last.pt / last_prev.pt.")
            saved = []
            if is_best:
                save_best(paths, model, e, metrics, to_plain(cfg))
                saved.append("best.pt")
            save_last(paths, {
                "version": 1,
                "epoch": e,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else {},
                "best_score": best_score,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": history,
                "finished": finished,
                "stop_reason": stop_reason,
                "rng": capture_rng_state(),
                "config": to_plain(cfg),
            })
            saved.append("last.pt")

            remaining = (total_epochs - e) * (sum(epoch_times[-5:]) / len(epoch_times[-5:]))
            hd = f" hd95={metrics['hd95']:.2f}px" if metrics.get("hd95") is not None else ""
            logger.info(
                f"epoch={e}/{total_epochs} train_loss={train_loss:.4f} val_loss={metrics.get('loss', float('nan')):.4f} "
                f"val_dice={metrics.get('dice', float('nan')):.4f} val_iou={metrics.get('iou', float('nan')):.4f} "
                f"val_dice_slice={metrics.get('dice_slice', float('nan')):.4f}{hd} | "
                f"{epoch_time:.0f}s ETA {remaining / 60:.0f}min | saved: {', '.join(saved)}"
                + (f" | NEW BEST (epoch {best_epoch})" if is_best else f" | best {best_score:.4f} @ {best_epoch}")
            )
            if finished:
                logger.info(f"Stopping: {stop_reason}")
                break

    except KeyboardInterrupt:
        interrupted = True
        logger.info(
            f"\nInterrupted. Completed epochs are safe in {paths.ckpt_dir}.\n"
            f"Run the same command again to resume from epoch {completed_epochs + 1} "
            f"(the unfinished epoch is simply repeated)."
        )
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        writer.flush()
        writer.close()

    summary = {
        "run_dir": str(paths.run_dir),
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "stop_reason": stop_reason,
        "interrupted": interrupted,
    }
    with (paths.run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if not interrupted:
        logger.info(f"Best val_dice {best_score:.4f} at epoch {best_epoch}. Best weights: {paths.best}")
        logger.info(f"Evaluate on the test split with: python scripts/evaluate.py --run-dir {paths.run_dir}")
    return {**summary, "best_checkpoint": str(paths.best)}
