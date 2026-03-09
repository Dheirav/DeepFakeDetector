# Model Card — resnet50_augv1_light

## Overview

| Field | Value |
|---|---|
| **Run name** | `resnet50_augv1_light` |
| **Checkpoint** | `models/resnet50_augv1_light/best_model.pth` |
| **Architecture** | ResNet-50 (ImageNet pretrained) + SRM |
| **Task** | 3-class: Real / AI Generated / AI Edited |
| **Status** | Experimental |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs (trained / max) | 26 / 50 |
| Best val acc | 87.14% |
| Final train acc | 98.48% |
| Final val acc | 85.58% |
| Backbone | resnet50 |
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
| Best val acc | 87.14% |
| Final val acc | 85.58% |
| Final train acc | 98.48% |

---

## Notes

- Model trained with light augmentation and SRM enabled.
- For more details, see training logs in `results/resnet50_augv1_light/`.
