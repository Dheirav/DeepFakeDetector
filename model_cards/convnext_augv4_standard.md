# Model Card — convnext_augv4_standard

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_augv4_standard` |
| **Checkpoint** | `models/convnext_augv4_standard/best_model.pth` |
| **Architecture** | ConvNeXt-Tiny + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | Superseded by `convnext_augv3_light` |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 30 / 50 |
| Best epoch | 20 |
| Backbone | convnext_tiny |
| Loss | weighted_focal |
| Focal gamma | 3.0 |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout | 0.4 |
| LR schedule | cosine |
| Augmentation | **standard** |
| SRM | enabled |
| Seed | 20 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | 86.62% |
| Test acc | 86.82% |
| Macro F1 | 0.8681 |
| F1 Real | 0.8124 |
| F1 AI Generated | 0.9592 |
| F1 AI Edited | 0.8327 |

### Confusion Matrix

```
[[6312,  232, 1251],
 [ 244, 7479,   69],
 [1189,   92, 6473]]
```

### Boundary Focus (Real ↔ AI Edited)

- Real → AI Edited: 1,251
- AI Edited → Real: 1,189
- Total boundary errors: **2,440**

---

## Notes

- Standard augmentation at seed 20 tracks close to prior ConvNeXt runs but does not beat `convnext_gamma3`.
- Useful as a stability reference point against `convnext_augv3_light`.
