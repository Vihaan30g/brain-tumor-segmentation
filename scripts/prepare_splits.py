from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segkit.config import load_config
from segkit.datasets.splits import build_split_file, ensure_split_file, split_file_path


def main():
    parser = argparse.ArgumentParser(description="Create the train/val/test split file (never overwritten unless --force).")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "transunet_dataset1.yaml"))
    parser.add_argument("--force", action="store_true",
                        help="Regenerate the split even if it exists. Do NOT do this after training has started.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = split_file_path(cfg)
    if args.force or not path.exists():
        split = build_split_file(cfg)
        print(f"Split file written: {path}")
    else:
        split = ensure_split_file(cfg)
        print(f"Split file already exists (kept): {path}")
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
