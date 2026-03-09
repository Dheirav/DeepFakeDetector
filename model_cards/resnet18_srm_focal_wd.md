# Model Card — resnet18_srm_focal_wd

## Overview

| Field | Value |
|---|---|
| **Run name** | `resnet18_srm_focal_wd` |
| **Checkpoint** | `models/resnet18_srm_focal_wd/best_resnet18.pth` |
| **Architecture** | ResNet-18 (ImageNet pretrained) + SRM high-pass filter |
| **Task** | 3-class image classification: Real / AI Generated / AI Edited |
| **Training script** | `scripts/training/train_full.py` |
| **Status** | Current best model |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (max / trained) | 30 / 15 (early stop) |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=2) |
| Weight decay | **1e-4** |
| Loss function | **WeightedFocalLoss** (class weights [1.5, 1.0, 1.5], γ=2) |
| SRM layer | **Yes** (3 RGB + 3 SRM residual channels → 6-channel input) |
| Label smoothing | **0.1** |
| torch.compile | Yes |
| Input size | 224 × 224 |
| Augmentation | Standard (flip, colour jitter, normalize) |
| Seed | 42 |

---

## Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 Macro | F1 Real | F1 AI-Gen | F1 AI-Edit |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4675 | 72.0% | 0.4161 | 77.8% | 0.776 | 0.693 | 0.877 | 0.757 |
| 5 | 0.2298 | 89.0% | 0.4098 | 80.7% | 0.806 | 0.723 | 0.906 | 0.790 |
| 10 | 0.0964 | 97.7% | 0.4329 | 82.8% | 0.828 | 0.764 | 0.921 | 0.799 |
| **15 (best)** | **0.0724** | **98.9%** | **0.4407** | **82.6%** | **0.827** | **0.761** | **0.920** | **0.798** |

> Early stopping triggered at epoch 15 (patience=5). Best checkpoint saved at epoch 10.

**Best val accuracy:** 82.80%  
**Val loss stability:** Plateaued at ~0.43 across epochs 8–15 (vs CE baseline which diverged to 0.68) — weight decay and label smoothing successfully reduced overfitting.

---

## Test Set Evaluation

Evaluated on `dataset_builder/test/` (23,341 samples).

**Overall accuracy: 82.94%**

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Real | 0.756 | 0.775 | 0.765 | 7,795 |
| AI Generated | 0.915 | 0.928 | 0.921 | 7,792 |
| AI Edited | 0.818 | 0.786 | 0.801 | 7,754 |
| **Macro avg** | **0.830** | **0.829** | **0.829** | 23,341 |
| Weighted avg | 0.830 | 0.829 | 0.829 | 23,341 |

### Confusion Matrix

```
               Pred Real   Pred AI-Gen   Pred AI-Edit
Actual Real       6040         503           1252
Actual AI-Gen      460        7227            105
Actual AI-Edit    1493         169           6092
```

### Key observations
- **AI Generated** is the easiest class (F1 0.921) — the model confidently separates synthetic images
- **Real** is the hardest class — 1,252 real images leaked into AI-Edited (11.5% of Real support)
- **AI Edited** is mid-difficulty — 1,493 AI-edited images predicted as Real (the toughest boundary)

---

## Comparison vs Previous Runs

| Run | Test Acc | F1 Macro | F1 Real | Val Loss (final) |
|---|---|---|---|---|
| resnet18_ce_baseline | — | — | — | 0.682 (diverging) |
| resnet18_srm_focal_5ep | 80.70% | 0.808 | 0.753 | 0.391 (not converged) |
| **resnet18_srm_focal_wd** | **82.94%** | **0.829** | **0.765** | **0.441 (stable)** |

---

## Known Limitations

- Train acc 98.9% vs test acc 82.9% — ~16pt overfitting gap remains despite weight decay
- Real↔AI-Edited boundary accounts for the majority of errors
- ResNet-18 backbone is relatively small; a stronger backbone (EfficientNet-B3, ConvNeXt-Tiny) would likely push past the current ceiling
- No test-time augmentation (TTA) applied

## Suggested Next Steps

1. Increase Real class weight to 2.0 to directly penalise Real misses
2. Add dropout (0.3–0.5) before FC head to reduce overfitting gap
3. Switch to CosineAnnealingLR schedule
4. Upgrade backbone to EfficientNet-B3 or ConvNeXt-Tiny
