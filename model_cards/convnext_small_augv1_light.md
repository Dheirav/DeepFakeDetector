# Model Card — convnext_small_augv1_light

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_small_augv1_light` |
| **Checkpoint** | `models/convnext_small_augv1_light/best_model.pth` |
| **Architecture** | ConvNeXt-Small + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | Experimental |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 27 / 50 |
| Best val acc | 89.29% |
| Final train acc | 97.15% |
| Final val acc | 88.43% |
| Backbone | convnext_small |
| Loss | weighted_focal |
| Focal gamma | 3.0 |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout | 0.4 |
| LR schedule | cosine |
| Augmentation | light |
| SRM | enabled |
| Seed | 42 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | 89.29% |
| Final val acc | 88.43% |
| Final train acc | 97.15% |

---

## Notes

- Model trained with light augmentation and SRM enabled.
- For more details, see training logs in `results/convnext_small_augv1_light/`.
