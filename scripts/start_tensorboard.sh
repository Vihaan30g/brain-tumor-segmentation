#!/usr/bin/env bash
# Live training curves + per-epoch validation images:  http://localhost:6006
cd "$(dirname "$0")/.." || exit 1
exec tensorboard --logdir runs --port "${1:-6006}" --reload_interval 15
