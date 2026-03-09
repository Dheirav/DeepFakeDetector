# Model Card — convnext_augv4_light_seeded_11

## Overview

| Field | Value |
|---|---|
| **Run name** | `convnext_augv4_light_seeded_11` |
| **Checkpoint** | `models/convnext_augv4_light_seeded_11/best_model.pth` |
| **Architecture** | ConvNeXt-Tiny + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | Experimental (seed=11) |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 30 / 50 |
| Best val acc | 89.00% |
| Final train acc | 98.31% |
| Final val acc | 87.34% |
| Backbone | convnext_tiny |
| Loss | weighted_focal |
| Focal gamma | 3.0 |
| Class weights | [1.5, 1.0, 1.5] |
| Dropout | 0.4 |
| LR schedule | cosine |
| Augmentation | light |
| SRM | enabled |
| Seed | 11 |

---

## Results

| Metric | Value |
|---|---|
| Best val acc | 89.00% |
| Final val acc | 87.34% |
| Final train acc | 98.31% |

---

## Notes

- Model trained with light augmentation and SRM enabled.
- For more details, see training logs in `results/convnext_augv4_light_seeded_11/`.
