# Brain tumor MRI segmentation with a TransUNet-style model

PyTorch pipeline for binary brain tumor segmentation on 2D MRI slices (PNG images + PNG masks).
Model: timm ViT-R50 hybrid + ResNet-50 skip encoder + U-Net style decoder. Everything (paths,
model, optimiser, augmentation, ...) is configured in `configs/transunet_dataset1.yaml`.

## Setup

```bash
source ~/ml_env/bin/activate
pip install -r requirements.txt
```

Put the dataset in `data/dataset1/{images,masks}` (see `data/README.md`), then, from the repo root:

```bash
python scripts/check_data.py            # sanity-check files, masks and split
python scripts/check_model.py --train-step   # optional: model shapes + peak GPU memory of one step
```

## Train

```bash
python scripts/train.py
```

* The **same command resumes** an interrupted run (Ctrl-C, crash, power cut). It continues from the
  last completed epoch of the newest unfinished run in `runs/transunet_dataset1/<timestamp>/`.
* `python scripts/train.py --fresh` starts a new run; `--resume <run_dir or last.pt>` resumes a specific run.
* An epoch that was interrupted half-way is simply repeated.

### Live monitoring (second terminal)

```bash
tensorboard --logdir runs --port 6006    # then open http://localhost:6006
```

TensorBoard shows train loss (per iteration and epoch), validation loss / Dice / IoU / precision /
recall / HD95, learning rates, GPU memory, and the **validation sample images of every epoch**
(Images tab, tag `val/samples`).

### What is written to `runs/<experiment>/<timestamp>/`

| path | content |
|---|---|
| `checkpoints/last.pt` | full state after the latest completed epoch (model, optimizer, scaler, RNG, history) |
| `checkpoints/last_prev.pt` | the epoch before (automatic fallback if `last.pt` is damaged) |
| `checkpoints/best.pt` | weights of the best validation-Dice epoch |
| `samples/epoch_NNN/*.png` | the same 5 validation slices after every epoch: image, ground truth, prediction, error map |
| `metrics.csv` | one row per epoch (losses, metrics, lr, time, GPU memory) |
| `logs/train.log` | everything printed to the terminal |
| `tensorboard/` | TensorBoard event files |
| `config.yaml`, `summary.json` | config copy and final summary |

All checkpoint files are written atomically (temp file, fsync, rename), so a power cut during saving cannot
destroy the previous checkpoint.

## Evaluate on the held-out test split

```bash
python scripts/evaluate.py                       # latest run, checkpoints/best.pt
python scripts/evaluate.py --run-dir runs/transunet_dataset1/<timestamp>
python scripts/evaluate.py --checkpoint path/to/best.pt --split val
```

Writes `eval_test/metrics.json`, `per_slice.csv` and overlay images next to the run.

## Predict a mask for one image

Pass an image filename from `data/dataset1/images/` or a path to another grayscale PNG:

```bash
python scripts/predict.py --image 123.png
python scripts/predict.py --image /path/to/brain_mri.png --checkpoint runs/transunet_dataset1/<timestamp>/checkpoints/best.pt
```

By default, the script uses `checkpoints/best.pt` from the latest run and saves a binary
`*_mask.png` plus an input/mask/overlay comparison `*_prediction.png` in that run's `predictions/`
folder. Use `--output-dir` to choose another folder or `--threshold` to override the configured
prediction threshold.

## Notes

* Metrics: `dice/iou/precision/recall/specificity` are pooled over all validation pixels (this is what picks
  `best.pt`); `*_slice` are means of the per-slice values; `hd95` is the mean per-slice 95th-percentile
  Hausdorff distance in pixels (`train.hd95_every_n_epochs`).
* The train/val/test split is random at image level and is stored in `data/splits/*.json` (created once,
  never overwritten). If several slices come from the same patient, val/test scores are optimistic.
* GPU memory is tight on 8 GB cards: if you see out-of-memory errors lower `train.batch_size`
  (and optionally `train.val_batch_size`) in the config; run `python scripts/check_model.py --train-step` to measure.
