# Data folder

The dataset itself is **not** stored in git. Put it here:

```
data/
  dataset1/
    images/   1.png, 2.png, ...
    masks/    1.png, 2.png, ...      (same file names as in images/)
  splits/
    dataset1_seed42.json             (created automatically on first run; this one IS tracked in git)
```

The location is set by `dataset.root` in `configs/transunet_dataset1.yaml` (relative paths are
relative to the repository root).
