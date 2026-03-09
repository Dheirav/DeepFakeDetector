# Model Card — convnext_augv2

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_augv2` |
| **Checkpoint** | `models/convnext_augv2/best_model.pth` |
| **Architecture** | ConvNeXt-Tiny + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | Rejected (strong augmentation degraded performance) |

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
| Augmentation | **strong** |
| SRM | enabled |
| Seed | 42 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | 83.87% |
| Test acc | 83.76% |
| Macro F1 | 0.8380 |
| F1 Real | 0.7738 |
| F1 AI Generated | 0.9374 |
| F1 AI Edited | 0.8029 |

### Confusion Matrix

```
[[6102,  316, 1377],
 [ 431, 7250,  111],
 [1444,  111, 6199]]
```

### Boundary Focus (Real ↔ AI Edited)

- Real → AI Edited: 1,377
- AI Edited → Real: 1,444
- Total boundary errors: **2,821**

---

## Notes

- Strong augmentation (noise + blur + wider compression) over-regularized the model.
- This run confirms boundary artifacts are easy to destroy with overly aggressive transforms.
