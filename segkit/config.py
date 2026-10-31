from __future__ import annotations

from pathlib import Path

import yaml

# Repository root (the folder that contains `segkit/`, `configs/`, `scripts/`).
REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SECTIONS = (
    "experiment", "dataset", "preprocessing", "augmentation",
    "model", "loss", "train", "viz",
)


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _convert_nested(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _convert_nested(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_convert_nested(v) for v in obj]
    return obj


def resolve_path(path) -> Path:
    """Resolve a config path. Relative paths are relative to the repo root (not the CWD)."""
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path):
    config_path = Path(path).expanduser()
    if not config_path.exists() and not config_path.is_absolute():
        alt = REPO_ROOT / config_path
        if alt.exists():
            config_path = alt
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _convert_nested(raw or {})
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"Config {config_path} is missing sections: {missing}")
    return cfg


def to_plain(value):
    """Convert AttrDict trees into plain dict/list (for yaml / checkpoints)."""
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


# Backwards compatible alias.
_to_plain_data = to_plain


def save_config_copy(cfg, output_dir, filename="config.yaml"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(cfg), fh, sort_keys=False)
    return path
