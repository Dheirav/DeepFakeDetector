# Model Card — convnext_augv4_light_seeded_20

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_augv4_light_seeded_20` |
| **Checkpoint** | `models/convnext_augv4_light_seeded_20/best_model.pth` |
| **Architecture** | ConvNeXt-Tiny + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | Experimental (seed=20) |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 28 / 50 |
| Best val acc | 88.99% |
| Final train acc | 98.19% |
| Final val acc | 87.63% |
| Backbone | convnext_tiny |
| Loss | weighted_focal |
| Focal gamma | 3.0 |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout | 0.4 |
| LR schedule | cosine |
| Augmentation | light |
| SRM | enabled |
| Seed | 20 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | 88.99% |
| Final val acc | 87.63% |
| Final train acc | 98.19% |

---

## Notes

- Model trained with light augmentation and SRM enabled.
- For more details, see training logs in `results/convnext_augv4_light_seeded_20/`.
