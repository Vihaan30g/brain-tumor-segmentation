from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segkit.config import load_config
from segkit.losses import BCEWithDiceLoss
from segkit.models import build_model
from segkit.utils.amp import amp_enabled, autocast_ctx, make_grad_scaler


def main():
    parser = argparse.ArgumentParser(description="Build the model, check shapes and (optionally) peak GPU memory of one train step.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "transunet_dataset1.yaml"))
    parser.add_argument("--train-step", action="store_true",
                        help="Run one forward/backward/optimizer step at the configured batch size and report peak GPU memory.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download pretrained weights.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {"pretrained": False} if args.no_pretrained else {}
    model = build_model(cfg, **overrides)
    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("vit.", "resnet.")))
    print(f"Parameters: total {n_total / 1e6:.1f}M (backbones {n_enc / 1e6:.1f}M, rest {(n_total - n_enc) / 1e6:.1f}M)")

    size = int(cfg.dataset.img_size)
    with torch.no_grad():
        y = model.eval()(torch.randn(1, 1, size, size))
    print(f"Output shape (1,1,{size},{size}) -> {tuple(y.shape)}")

    if args.train_step:
        if not torch.cuda.is_available():
            print("CUDA not available; skipping --train-step.")
            return
        device = torch.device("cuda")
        bs = int(cfg.train.batch_size)
        model.to(device).train()
        use_amp = amp_enabled(cfg, device)
        scaler = make_grad_scaler(use_amp)
        crit = BCEWithDiceLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, 1, size, size, device=device)
        t = (torch.rand(bs, 1, size, size, device=device) > 0.9).float()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp):
            out = model(x)
        loss = crit(out, t)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Train step OK at batch {bs} (AMP={use_amp}): peak {peak:.2f} GB of {total:.2f} GB")


if __name__ == "__main__":
    main()
