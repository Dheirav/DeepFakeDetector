# Model Card — resnet18_dropout_cosine

## Overview

| Field | Value |
|---|---|
| **Run name** | `resnet18_dropout_cosine` |
| **Checkpoint** | `models/resnet18_dropout_cosine/best_model.pth` |
| **Architecture** | ResNet-18 (ImageNet pretrained) + SRM high-pass filter |
| **Task** | 3-class image classification: Real / AI Generated / AI Edited |
| **Training script** | `scripts/training/train_full.py` |
| **Status** | Superseded by dropout05_plateau and ConvNeXt |
| **Storage (models/)** | 172 MB |
| **Storage (results/)** | ~540 KB |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 15 (early stop, patience=10) |
| Best epoch | 10 |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | **CosineAnnealingWarmRestarts (T_0=10)** |
| Weight decay | 1e-4 |
| Loss function | WeightedFocalLoss (γ=2, label smoothing=0.1) |
| Class weights | [1.5, 1.0, 1.5] (default) |
| Dropout (FC head) | **0.4** |
| SRM layer | Yes (6-channel: 3 RGB + 3 SRM residual) |
| torch.compile | Yes |
| Input size | 224 × 224 |

---

## Training Results

| Epoch | Train Acc | Val Acc | Val F1 Macro |
|---|---|---|---|
| 1 | 69.33% | 78.11% | 0.782 |
| 10 (best) | — | **82.94%** | **0.830** |
| 15 (last) | 95.31% | 82.67% | 0.828 |

**Train/val gap at last epoch: 13.5 pts** — overfitting persists  
**Note:** Cosine restart fired at epoch 11 (T_0=10 too short), causing a dip to 79.3% val acc before recovering

---

## Test Set Evaluation

**Overall test accuracy: 83.12%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real | 0.752 | 0.789 | 0.770 |
| AI Generated | 0.936 | 0.913 | 0.924 |
| AI Edited | 0.812 | 0.791 | 0.802 |
| **Macro avg** | **0.833** | **0.831** | **0.832** |

### Confusion Matrix
```
               Pred Real   Pred AI-Gen   Pred AI-Edit
Actual Real       6151         345           1299
Actual AI-Gen      555        7116            121
Actual AI-Edit    1473         146           6135
```

---

## Notes

- First run with dropout — established that 0.4 dropout alone doesn't close the overfitting gap
- Cosine `T_0=10` was too short — fixed to `T_0=20` in subsequent runs
- Performance essentially identical to `resnet18_srm_focal_wd` — confirms ResNet-18 ceiling
