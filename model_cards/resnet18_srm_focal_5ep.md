# Model Card — resnet18_srm_focal_5ep

## Overview

| Field | Value |
|---|---|
| **Run name** | `resnet18_srm_focal_5ep` |
| **Checkpoint** | `models/resnet18_srm_focal_5ep/best_baseline_resnet18.pth` |
| **Architecture** | ResNet-18 (ImageNet pretrained) + SRM high-pass filter |
| **Task** | 3-class image classification: Real / AI Generated / AI Edited |
| **Training script** | `scripts/training/train_baseline.py` |
| **Status** | Partial run — only 5 epochs, not converged |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (max / trained) | 5 / 5 (not converged) |
| Batch size | 32 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=2) |
| Weight decay | — (none) |
| Loss function | WeightedFocalLoss (class weights [1.5, 1.0, 1.5], γ=2) |
| SRM layer | **Yes** (3 RGB + 3 SRM residual channels → 6-channel input) |
| Label smoothing | 0.1 |
| Input size | 224 × 224 |
| Augmentation | Standard (flip, colour jitter, normalize) |
| Seed | 42 |

---

## Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.4914 | 70.4% | 0.4112 | 77.0% |
| 2 | 0.3848 | 78.0% | 0.3856 | 78.2% |
| 3 | 0.3342 | 81.7% | 0.3710 | 78.9% |
| 4 | 0.2972 | 84.4% | 0.3912 | 79.8% |
| **5** | **0.2607** | **86.9%** | **0.3908** | **80.1%** |

> Val accuracy was still climbing at epoch 5 — training ended early due to epoch cap, not convergence.

**Best val accuracy:** 80.07% *(not converged)*

---

## Test Set Evaluation

Evaluated on `dataset_builder/test/` (23,341 samples).

**Overall accuracy: 80.70%**

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Real | 0.700 | 0.814 | 0.753 | 7,795 |
| AI Generated | 0.928 | 0.886 | 0.906 | 7,792 |
| AI Edited | 0.817 | 0.721 | 0.766 | 7,754 |
| **Macro avg** | **0.815** | **0.807** | **0.808** | 23,341 |

### Confusion Matrix

```
               Pred Real   Pred AI-Gen   Pred AI-Edit
Actual Real       6345         386           1064
Actual AI-Gen      703        6901            188
Actual AI-Edit    2012         151           5591
```

---

## Limitations & Notes

- Only 5 epochs — val accuracy was clearly still improving. This is a proof-of-concept run only.
- Largest error source: 2,012 AI-Edited images misclassified as Real (worse than the converged run).
- No weight decay — overfitting likely to worsen with more epochs without regularisation.
- Demonstrates that SRM + focal loss clearly beats the plain CE baseline even without full convergence.
