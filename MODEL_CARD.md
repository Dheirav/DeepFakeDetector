# Model Cards

This file is the top-level index for all trained models in this project.
Detailed cards for each run live in the [`model_cards/`](model_cards/) directory.

---

## Models at a Glance

| Model | Architecture | Test Acc | F1 Macro | Status |
|---|---|---|---|---|
| [resnet18_ce_baseline](model_cards/resnet18_ce_baseline.md) | ResNet-18, CE loss | — | — | Baseline reference |
| [resnet18_srm_focal_5ep](model_cards/resnet18_srm_focal_5ep.md) | ResNet-18 + SRM, focal loss | 80.70% | 0.808 | Partial (5 ep, not converged) |
| [resnet18_srm_focal_wd](model_cards/resnet18_srm_focal_wd.md) | ResNet-18 + SRM, focal + WD | **82.94%** | **0.829** | **Current best** |

---

## Current Best — `resnet18_srm_focal_wd`

**Checkpoint:** `models/resnet18_srm_focal_wd/best_resnet18.pth`

Key config: SRM high-pass filter layer · WeightedFocalLoss (weights [1.5, 1.0, 1.5]) · label smoothing 0.1 · weight decay 1e-4 · 15 epochs (early stopped from 30)

### Test Set Results (23,341 samples)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Real | 0.756 | 0.775 | 0.765 | 7,795 |
| AI Generated | 0.915 | 0.928 | 0.921 | 7,792 |
| AI Edited | 0.818 | 0.786 | 0.801 | 7,754 |
| **Macro avg** | **0.830** | **0.829** | **0.829** | 23,341 |

### Confusion Matrix

```
               Pred Real   Pred AI-Gen   Pred AI-Edit
Actual Real       6040         503           1252
Actual AI-Gen      460        7227            105
Actual AI-Edit    1493         169           6092
```

Primary failure mode: Real ↔ AI-Edited boundary (~56% of all errors).

---

## How to Reproduce the Best Run

```bash
python3 scripts/training/train_full.py \
  --use_srm \
  --loss weighted_focal \
  --label_smoothing 0.1 \
  --weight_decay 1e-4 \
  --epochs 30 \
  --early_stop_patience 5 \
  --batch_size 64 \
  --run_name resnet18_srm_focal_wd
```

## How to Evaluate

```bash
python3 scripts/evaluation/evaluate.py \
  --model_path models/resnet18_srm_focal_wd/best_resnet18.pth \
  --data_dir dataset_builder/test
```
