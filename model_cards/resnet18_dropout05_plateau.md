# Model Card — resnet18_dropout05_plateau

## Overview

| Field | Value |
|---|---|
| **Run name** | `resnet18_dropout05_plateau` |
| **Checkpoint** | `models/resnet18_dropout05_plateau/best_model.pth` |
| **Architecture** | ResNet-18 (ImageNet pretrained) + SRM high-pass filter |
| **Task** | 3-class image classification: Real / AI Generated / AI Edited |
| **Training script** | `scripts/training/train_full.py` |
| **Status** | Superseded by ConvNeXt-Tiny |
| **Storage (models/)** | 172 MB |
| **Storage (results/)** | ~540 KB |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 24 (early stop, patience=10) |
| Best epoch | 14 |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | **ReduceLROnPlateau (factor=0.5, patience=2)** |
| Weight decay | 1e-4 |
| Loss function | WeightedFocalLoss (γ=2, label smoothing=0.1) |
| Class weights | [1.5, 1.0, 1.5] (default) |
| Dropout (FC head) | **0.5** |
| SRM layer | Yes (6-channel: 3 RGB + 3 SRM residual) |
| torch.compile | Yes |
| Input size | 224 × 224 |

---

## Training Results

| Epoch | Train Acc | Val Acc | Val F1 Macro |
|---|---|---|---|
| 1 | 67.85% | 77.76% | 0.779 |
| 14 (best) | — | **82.97%** | **0.830** |
| 24 (last) | 98.84% | 82.65% | 0.827 |

**Train/val gap at last epoch: 16.2 pts** — gap actually widened vs dropout=0.4  
**Key finding:** Increasing dropout from 0.4 → 0.5 did not help; confirmed ResNet-18 architecture is the bottleneck

---

## Test Set Evaluation

**Overall test accuracy: 83.27%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real | 0.772 | 0.761 | 0.766 |
| AI Generated | 0.929 | 0.920 | 0.925 |
| AI Edited | 0.798 | 0.817 | 0.807 |
| **Macro avg** | **0.833** | **0.833** | **0.833** |

### Confusion Matrix
```
               Pred Real   Pred AI-Gen   Pred AI-Edit
Actual Real       5933         386           1476
Actual AI-Gen      492        7172            128
Actual AI-Edit    1264         159           6331
```

---

## Notes

- AI-Edit→Real errors improved to 1,264 (best among ResNet-18 runs) but Real→AI-Edit got worse (1,476)
- Boundary confusion shifted direction rather than decreasing — confirms the issue is representational, not regularisation
- Final ResNet-18 experiment before moving to ConvNeXt-Tiny
- All ResNet-18 runs plateaued at 83.1–83.3% test accuracy regardless of configuration
