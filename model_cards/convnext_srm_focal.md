# Model Card — convnext_srm_focal

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_srm_focal` |
| **Checkpoint** | `models/convnext_srm_focal/best_model.pth` |
| **Architecture** | ConvNeXt-Tiny (ImageNet pretrained) + SRM high-pass filter |
| **Task** | 3-class image classification: Real / AI Generated / AI Edited |
| **Training script** | `scripts/training/train_full.py --backbone convnext_tiny` |
| **Status** | ⭐ **Current Best** |
| **Storage (models/)** | 172 MB |
| **Storage (results/)** | ~540 KB |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 27 (early stop, patience=10) |
| Best epoch | 17 |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | CosineAnnealingWarmRestarts (T_0=20, T_mult=2) |
| Weight decay | 1e-4 |
| Loss function | WeightedFocalLoss (γ=2, label smoothing=0.1) |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout (FC head) | 0.4 |
| SRM layer | Yes — adapted `model.features[0][0]` to 6-channel input |
| torch.compile | Yes |
| Input size | 224 × 224 |
| Backbone params | ~28M (vs 11M for ResNet-18) |

---

## Architecture Details

ConvNeXt-Tiny uses an inverted bottleneck design with depthwise convolutions and layer normalisation.  
Compared to ResNet-18 it has 2.5× more parameters and substantially richer feature extraction.

**FC head modification:**
```python
model.classifier[2] = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(768, 3))
```

**SRM adaptation:**
```python
model.features[0][0] = adapt_conv1_for_srm(model.features[0][0])
# Original: Conv2d(3, 96, kernel=4, stride=4) → New: Conv2d(6, 96, kernel=4, stride=4)
```

---

## Training Results

| Epoch | Train Acc | Val Acc | Val F1 Macro |
|---|---|---|---|
| 1 | 72.63% | 81.18% | 0.812 |
| 17 (best) | — | **86.71%** | **0.866** |
| 27 (last) | 99.38% | 86.34% | 0.862 |

**Train/val gap at last epoch: 12.7 pts** — lowest gap of all runs

---

## Test Set Evaluation

**Overall test accuracy: 86.85%** — +3.54 pts over best ResNet-18

| Class | Precision | Recall | F1 | Δ vs best ResNet-18 |
|---|---|---|---|---|
| Real | 0.816 | 0.806 | 0.811 | +0.036 |
| AI Generated | 0.960 | 0.957 | 0.958 | +0.034 |
| AI Edited | 0.837 | 0.833 | 0.835 | +0.033 |
| **Macro avg** | **0.871** | **0.865** | **0.868** | **+0.034** |

### Confusion Matrix
```
               Pred Real   Pred AI-Gen   Pred AI-Edit
Actual Real       6206         296           1293
Actual AI-Gen      188        7521             83
Actual AI-Edit    1124          86           6544
```

### Error Reduction vs Best ResNet-18 (sweep_w150_100_150)

| Error Type | ResNet-18 | ConvNeXt | Δ |
|---|---|---|---|
| Real → AI-Edit | 1,299 | 1,293 | -6 |
| Real → AI-Gen | 345 | 296 | **-49** |
| AI-Gen → Real | 555 | 188 | **-367** |
| AI-Gen → AI-Edit | 121 | 83 | **-38** |
| AI-Edit → Real | 1,473 | 1,124 | **-349** |
| AI-Edit → AI-Gen | 146 | 86 | **-60** |

---

## Notes

- ConvNeXt's modern architecture (LayerNorm, depthwise-separable convs) provides a substantial jump — all error types improved, most significantly AI-Gen→Real (-367) and AI-Edit→Real (-349)
- Training gap (12.7 pts) is lowest of all runs, suggesting ConvNeXt generalises better inherently
- Cosine schedule with T_0=20 trained smoothly through 27 epochs
- The primary remaining weakness is AI-Edit→Real confusion (1,124 errors, recall 0.833); cascade specialist model is the next recommended step
