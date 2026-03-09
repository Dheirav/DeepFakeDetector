# Model Cards

One card per trained checkpoint. Each card documents architecture, training config, val metrics, and test set evaluation results.

## Current Best

| Model | Backbone | Test Acc | F1 Macro | Status |
|---|---|---|---|---|
| [convnext_augv3_light](convnext_augv3_light.md) | ConvNeXt-Tiny | **88.99%** | **0.8895** | ⭐ **Current Best** |

---

## Production Models

| Model | Checkpoint | Test Acc | F1 Macro | Status |
|---|---|---|---|---|
| [resnet18_ce_baseline](resnet18_ce_baseline.md) | `models/resnet18_ce_baseline/best_resnet18.pth` | — | — | Baseline reference |
| [resnet18_srm_focal_5ep](resnet18_srm_focal_5ep.md) | `models/resnet18_srm_focal_5ep/best_baseline_resnet18.pth` | 80.70% | 0.808 | Partial run (5 ep) |
| [resnet18_srm_focal_wd](resnet18_srm_focal_wd.md) | `models/resnet18_srm_focal_wd/best_resnet18.pth` | 82.94% | 0.829 | Superseded |

---

## Architecture Experiments

Architecture and regularisation experiments run after the weight sweep, using best weights [1.5, 1.0, 1.5] throughout.

| Model | Backbone | Key Changes | Val Acc | Test Acc | F1 Macro | Status |
|---|---|---|---|---|---|---|
| [resnet18_dropout_cosine](resnet18_dropout_cosine.md) | ResNet-18 | dropout=0.4, cosine LR (T_0=20) | 82.94% | 83.12% | 0.832 | Superseded |
| [resnet18_dropout05_plateau](resnet18_dropout05_plateau.md) | ResNet-18 | dropout=0.5, ReduceLROnPlateau | 82.97% | 83.27% | 0.833 | Superseded |
| [convnext_srm_focal](convnext_srm_focal.md) | ConvNeXt-Tiny | 28M params, dropout=0.4, cosine LR, gamma=2 | 86.71% | 86.85% | 0.8680 | Superseded |
| [convnext_gamma3](convnext_gamma3.md) | ConvNeXt-Tiny | gamma=3.0 (other settings unchanged) | 86.78% | 86.96% | 0.8694 | Superseded |
| [convnext_augv2](convnext_augv2.md) | ConvNeXt-Tiny | gamma=3.0 + augment=strong | 83.87% | 83.76% | 0.8380 | Rejected |
| [convnext_augv4_standard](convnext_augv4_standard.md) | ConvNeXt-Tiny | gamma=3.0 + augment=standard (seed=20) | 86.62% | 86.82% | 0.8681 | Superseded |
| [convnext_augv3_light](convnext_augv3_light.md) | **ConvNeXt-Tiny** | gamma=3.0 + augment=light | **88.91%** | **88.99%** | **0.8895** | ⭐ **Current Best** |

### Key takeaways
- Dropout (0.4 or 0.5) and LR schedule changes gave no meaningful improvement on ResNet-18 — all runs plateaued at 83.1–83.3%
- The ~13–16 pt train/val gap persisted across all ResNet-18 experiments, confirming an architecture bottleneck
- Switching to ConvNeXt-Tiny resolved the ResNet ceiling; gamma=3.0 added a small incremental lift
- Augmentation intensity matters: `strong` degraded heavily, `standard` stayed flat, and `light` produced the best generalization
- `convnext_augv3_light` reduced Real↔AI-Edited boundary errors from 2,396 (`convnext_gamma3`) to 2,153 (-243)

### What to do next
- Run seed-stability checks for `convnext_augv3_light` (e.g., seeds 7, 21, 42) and compare mean/std of test accuracy and boundary errors
- If stability holds (>=88.5% across seeds), promote to production checkpoint
- If variance is high, proceed to the cascade specialist Real-vs-AI-Edited model

---

## Weight Sweep Models (ResNet-18 + SRM + WeightedFocalLoss, 10 ep budget)

All sweep models share the same architecture and hyperparameters; only `class_weights` differs.  
Ranked by validation accuracy. Test accuracy from `evaluate.py` on `dataset_builder/test/` (23,341 samples).

| Rank | Model | Weights [Real, AIGen, AIEdit] | Val Acc | Test Acc | F1 Real | F1 AI-Gen | F1 AI-Edit | Storage |
|---|---|---|---|---|---|---|---|---|
| 1 ★ | [sweep_w150_100_150](sweep_w150_100_150.md) | [1.5, 1.0, 1.5] | 82.94% | **83.31%** | 0.775 | 0.924 | 0.802 | 172 MB |
| 2 | [sweep_w300_100_150](sweep_w300_100_150.md) | [3.0, 1.0, 1.5] | 82.83% | 82.91% | **0.773** | 0.920 | 0.798 | 172 MB |
| 3 | [sweep_w200_100_200](sweep_w200_100_200.md) | [2.0, 1.0, 2.0] | 82.73% | 82.86% | 0.764 | 0.921 | 0.803 | 172 MB |
| 4 | [sweep_w250_100_150](sweep_w250_100_150.md) | [2.5, 1.0, 1.5] | 82.67% | 82.90% | 0.773 | 0.919 | 0.801 | 172 MB |
| 5 | [sweep_w150_100_200](sweep_w150_100_200.md) | [1.5, 1.0, 2.0] | 82.63% | 82.85% | 0.759 | 0.922 | **0.804** | 172 MB |
| 6 | [sweep_w200_100_150](sweep_w200_100_150.md) | [2.0, 1.0, 1.5] | 82.59% | 83.05% | 0.772 | **0.924** | 0.799 | 172 MB |
| 7 | [sweep_w200_080_150](sweep_w200_080_150.md) | [2.0, 0.8, 1.5] | 82.52% | 82.56% | 0.766 | 0.918 | 0.797 | 172 MB |
| 8 | [sweep_w200_080_200](sweep_w200_080_200.md) | [2.0, 0.8, 2.0] | 82.22% | 82.61% | 0.766 | 0.918 | 0.799 | 172 MB |

### Key takeaways
- The reference weights **[1.5, 1.0, 1.5]** remain the best overall config
- Boosting Real weight past 2.0 yields diminishing returns and slower convergence
- Reducing AI-Gen weight below 1.0 consistently hurts performance (ranks 7 & 8)
- The Real ↔ AI-Edit confusion boundary is resilient to weight tuning alone

### Storage summary (sweep models)

| Location | Per model | Total (8 models) |
|---|---|---|
| `models/sweep_*/` | 172 MB | **1.38 GB** |
| `results/sweep_*/` | ~538 KB | ~4.3 MB |
| **Combined** | **~172.5 MB** | **~1.38 GB** |

---

### Storage summary (all runs)

| Run | models/ | results/ |
|---|---|---|
| resnet18_dropout_cosine | 172 MB | ~540 KB |
| resnet18_dropout05_plateau | 172 MB | ~540 KB |
| convnext_srm_focal | 172 MB | ~540 KB |
| 8 sweep models | 1.38 GB | ~4.3 MB |
| 3 production baselines | ~516 MB | — |
| **Total** | **~2.4 GB** | **~6 MB** |
