# Model Card — convnext_gamma3

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_gamma3` |
| **Checkpoint** | `models/convnext_gamma3/best_model.pth` |
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
| Focal gamma | **3.0** |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout | 0.4 |
| LR schedule | cosine |
| Augmentation | default/legacy (`augment` not set in config) |
| SRM | enabled |
| Seed | 42 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | 86.78% |
| Test acc | 86.96% |
| Macro F1 | 0.8694 |
| F1 Real | 0.8128 |
| F1 AI Generated | 0.9586 |
| F1 AI Edited | 0.8369 |

### Confusion Matrix

```
[[6278,  251, 1266],
 [ 245, 7485,   62],
 [1130,   89, 6535]]
```

### Boundary Focus (Real ↔ AI Edited)

- Real → AI Edited: 1,266
- AI Edited → Real: 1,130
- Total boundary errors: **2,396**

---

## Notes

- `gamma=3.0` gave a small but consistent lift over `convnext_srm_focal`.
- Improvement magnitude is modest; single-model tuning appears near ceiling.
