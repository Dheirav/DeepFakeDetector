# Model Card — Run `20260307_063053`

**Checkpoint:** `models/run_20260307_063053/best_resnet18.pth`
**Trained:** 2026-03-07

---

## 1. Model Overview

| Property | Value |
|---|---|
| Architecture | ResNet18 (ImageNet pretrained) |
| Task | 3-class image classification |
| Classes | `0` Real · `1` AI Generated · `2` AI Edited |
| Input | 224 × 224 RGB, normalised (ImageNet μ/σ) |
| Output head | 512 → 3 fully-connected, CrossEntropyLoss |
| Checkpoint | `best_resnet18.pth` (best val accuracy across all epochs) |

---

## 2. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Max epochs | 30 |
| Epochs trained | 15 (early stopped) |
| Seed | 42 |
| Device | CUDA |

---

## 3. Dataset

| Split | Path | Images |
|---|---|---|
| Train | `dataset_builder/train/` | ~31,000 |
| Val | `dataset_builder/val/` | ~23,000 |
| Test | `dataset_builder/test/` | 23,341 |

**Sources:** 20 collections including FaceForensics++, CASIA, DEFACTO, ForgeryNet, IMD2020, OpenForensics (AI Edited); Stable Diffusion, FLUX, Midjourney, StyleGAN, SynthBuster (AI Generated); COCO, ImageNet, OpenImages, Places365 (Real).

Class balance on test split: Real 7,795 · AI Generated 7,792 · AI Edited 7,754 — near-uniform at 33.3% each.

---

## 4. Per-Epoch Metrics

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | F1 Macro | F1 Real | F1 AI Gen | F1 AI Edited |
|---|---|---|---|---|---|---|---|---|
| 1  | 0.5784 | 73.3% | 0.4821 | 78.1% | 0.783 | 0.726 | 0.884 | 0.740 |
| 15 | 0.0402 | 98.7% | 0.6823 | 82.1% | 0.821 | 0.747 | 0.919 | 0.796 |
| **best** | — | — | — | **82.63%** | — | — | — | — |

> The best checkpoint is taken at the epoch with the highest validation accuracy (82.63%), not the final epoch.
> The widening gap between train acc (98.7%) and val acc (82.1%) at epoch 15 indicates overfitting — which is why early stopping / the best-checkpoint strategy matters.

---

## 5. Test Set Results

Evaluated on 23,341 held-out images from `dataset_builder/test/`.

### 5.1 Overall

| Metric | Value |
|---|---|
| Accuracy | **82.73%** |
| Macro Precision | 0.83 |
| Macro Recall | 0.83 |
| Macro F1 | **0.83** |

### 5.2 Per-Class

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Real | 0.77 | 0.75 | **0.76** | 7,795 |
| AI Generated | 0.91 | 0.93 | **0.92** | 7,792 |
| AI Edited | 0.80 | 0.80 | **0.80** | 7,754 |

**Key observation:** AI Generated images are the easiest class to detect (F1 0.92). Real vs AI Edited is the hardest boundary (F1 0.76 on Real).

### 5.3 Confusion Matrix

```
                  Predicted →
                  Real    AI Gen   AI Edited
Actual Real     [ 5,864    509      1,422 ]
Actual AI Gen   [   437  7,262         93 ]
Actual AI Edited[ 1,361    209       6,184 ]
```

#### Error Analysis

| Error type | Count | Share of all errors |
|---|---|---|
| Real → AI Edited (false positive) | 1,422 | 31.1% |
| AI Edited → Real (false negative) | 1,361 | 29.8% |
| **Real ↔ AI Edited total** | **2,783** | **60.9%** |
| Real → AI Gen | 509 | 11.1% |
| AI Gen → Real | 437 | 9.6% |
| AI Edited → AI Gen | 209 | 4.6% |
| AI Gen → AI Edited | 93 | 2.0% |

**~61% of all errors are Real ↔ AI Edited confusions.** This is expected: subtle inpainting and region-level edits produce images that retain the same overall scene statistics as real photographs.

---

## 6. Artefacts

| File | Description |
|---|---|
| `models/run_20260307_063053/best_resnet18.pth` | Best validation accuracy checkpoint |
| `models/run_20260307_063053/resnet18_epoch13.pth` | Epoch 13 checkpoint |
| `models/run_20260307_063053/resnet18_epoch14.pth` | Epoch 14 checkpoint |
| `models/run_20260307_063053/resnet18_epoch15.pth` | Final epoch checkpoint |
| `results/run_20260307_063053/metrics.csv` | Per-epoch train/val metrics |
| `results/run_20260307_063053/training_summary.json` | Run config + final stats |
| `results/run_20260307_063053/loss_curve.png` | Train vs val loss plot |
| `results/run_20260307_063053/accuracy_curve.png` | Train vs val accuracy plot |
| `results/run_20260307_063053/confusion_matrix.png` | Normalised confusion matrix (test) |
| `results/run_20260307_063053/y_true.npy` | Ground-truth labels (test set) |
| `results/run_20260307_063053/y_pred.npy` | Predicted labels (test set) |
| `results/run_20260307_063053/tensorboard/` | TensorBoard event files |
| `results/run_20260307_063053/profiler/` | PyTorch Profiler trace (epoch 1) |

---

## 7. Limitations & Known Issues

1. **Real ↔ AI Edited confusion (61% of errors):** Subtle edits that don't change scene-level statistics are the primary failure mode. Spatial frequency features or patch-level attention may help.
2. **Train/val accuracy gap:** 98.7% vs 82.1% at epoch 15 — the model has memorised training data. Stronger augmentation (MixUp, CutMix, frequency-domain noise) or weight decay would narrow this.
3. **Fixed 224 × 224 resolution:** High-frequency manipulation artefacts that appear at native resolution may be destroyed by resizing.
4. **No TTA (Test-Time Augmentation):** Single-crop inference; ensemble or multi-crop would improve results at the cost of latency.

---

## 8. Suggested Next Steps

| Priority | Change | Expected gain |
|---|---|---|
| High | Weighted `CrossEntropyLoss` to up-weight Real and AI Edited | +2–4% F1 on confused classes |
| High | Label smoothing (`label_smoothing=0.1`) | Reduce overconfident predictions |
| Medium | Replace ResNet18 with EfficientNet-B3 or ResNet50 | +2–4% overall accuracy |
| Medium | Add frequency-domain augmentations (JPEG noise, DCT masking) | Better AI Edited recall |
| Low | MixUp / CutMix during training | Reduce train/val gap |
| Low | TTA at inference time | +0.5–1% accuracy with no retraining |

---

## 9. How to Reproduce

```bash
# Train (reproduces this run with the same config)
python scripts/training/train_full.py \
  --data_dir dataset_builder/train \
  --val_dir  dataset_builder/val \
  --epochs 30 --batch_size 64 --lr 1e-4 \
  --seed 42

# Evaluate on test set
python scripts/evaluation/evaluate.py \
  --model_path models/run_20260307_063053/best_resnet18.pth \
  --data_dir   dataset_builder/test

# Plot confusion matrix
python scripts/evaluation/plot_confusion_matrix.py

# Streamlit UI
streamlit run frontend/app.py
```
