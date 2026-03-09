# Model Card — convnext_augv3_light

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_augv3_light` |
| **Checkpoint** | `models/convnext_augv3_light/best_model.pth` |
| **Architecture** | ConvNeXt-Tiny + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | ⭐ **Current Best** |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 29 / 50 |
| Best epoch | 19 |
| Backbone | convnext_tiny |
| Loss | weighted_focal |
| Focal gamma | 3.0 |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout | 0.4 |
| LR schedule | cosine |
| Augmentation | **light** |
| SRM | enabled |
| Seed | 42 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | **88.91%** |
| Test acc | **88.99%** |
| Macro F1 | **0.8895** |
| F1 Real | 0.8419 |
| F1 AI Generated | 0.9735 |
| F1 AI Edited | 0.8531 |

### Confusion Matrix

```
[[6540,  195, 1060],
 [ 108, 7646,   38],
 [1093,   76, 6585]]
```

### Boundary Focus (Real ↔ AI Edited)

- Real → AI Edited: 1,060
- AI Edited → Real: 1,093
- Total boundary errors: **2,153**

---

## Delta vs previous best (`convnext_gamma3`)

- Test accuracy: **+2.03 pts** (86.96 → 88.99)
- Macro F1: **+0.0201** (0.8694 → 0.8895)
- Boundary errors: **-243** (2,396 → 2,153)

---

## Notes

- Light augmentation with `gamma=3.0` gives the strongest generalization so far.
- This run is the right baseline for seed-stability verification and deployment candidate selection.
