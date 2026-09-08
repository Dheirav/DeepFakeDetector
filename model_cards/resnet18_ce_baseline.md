# Model Card — resnet18_ce_baseline

## Overview

| Field | Value |
|---|---|
| **Run name** | `resnet18_ce_baseline` |
| **Checkpoint** | `models/resnet18_ce_baseline/best_resnet18.pth` |
| **Architecture** | ResNet-18 (ImageNet pretrained) |
| **Task** | 3-class image classification: Real / AI Generated / AI Edited |
| **Training script** | `scripts/training/train_full.py` |
| **Status** | Superseded — use `resnet18_srm_focal_wd` instead |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (max / trained) | 30 / 15 (early stop) |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=2) |
| Weight decay | — (none) |
| Loss function | CrossEntropyLoss (standard, unweighted) |
| SRM layer | No |
| Label smoothing | No |
| Input size | 224 × 224 |
| Augmentation | Standard (flip, colour jitter, normalize) |
| Seed | 42 |

---

## Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 Macro |
|---|---|---|---|---|---|
| 1 | 0.5784 | 73.3% | 0.4821 | 78.1% | 0.783 |
| 5 | 0.2568 | 89.4% | 0.5125 | 80.7% | 0.810 |
| 10 | 0.0659 | 97.8% | 0.6103 | 82.6% | 0.826 |
| 15 (last) | 0.0402 | 98.7% | 0.6823 | **82.13%** | **0.821** |

> *Corrected 2026-09-08: epoch 15 was labelled "(best)" and its val accuracy given as
> 82.6%. `results/01/metrics.csv` records epoch 15 at **0.8213**. The best epoch was
> **10** at 0.8263 — which is the 82.63% quoted below, correctly valued but attributed
> to the wrong epoch.*

> Early stopping triggered at epoch 15 (patience=5).

**Best val accuracy:** 82.63%  
**Overfitting gap:** Train 98.7% vs Val 82.6% — val loss continuously rising from epoch 5, indicating significant overfitting with no regularisation.

---

## Test Set Evaluation

**Corrected 2026-09-08 — it was evaluated.** `results/01/y_true.npy` and `y_pred.npy`
exist and give **82.73%** test accuracy on the 23,341-image held-out split. This is the
number headlined in the project README, and all nine cells of that README table reproduce
from these two files to 4 d.p. See [`../LIMITATIONS.md`](../LIMITATIONS.md) for why it is
inflated and what it does not measure.

---

## Per-class Validation Metrics (final epoch)

| Class | F1 |
|---|---|
| Real | 0.747 |
| AI Generated | 0.919 |
| AI Edited | 0.796 |

---

## Limitations & Notes

- No weight decay or label smoothing — val loss diverges after epoch 5 while train loss keeps dropping.
- Standard CE loss gives equal weight to all classes — the harder Real/AI-Edited boundary is not explicitly penalised.
- No SRM layer — high-frequency manipulation artefacts are not explicitly extracted.
- Serves as the baseline reference. All subsequent experiments improve on this.
