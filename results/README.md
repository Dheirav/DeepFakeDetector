# Results — run index and corrections

Audited 2026-09-08. Read this before using any run directory.

## Run folder names are not reliable metadata

The folder naming scheme (`NN__backbone__aug__dropout__sched__loss__features`) was
applied by hand and drifted from the actual configuration. **The authoritative
record is each run's `training_summary.json`**, which is written directly from the
variables used to build the model. Where the two disagree, the JSON is correct —
and for the four runs below the weights confirm the JSON.

| Run folder | Name claims | Actually trained with | Confirmed by |
|---|---|---|---|
| `23__resnet50__light__…__srm-gem` | GeM pooling | **no GeM** — `attention_head: "none"` | no `avgpool.p` in weights; `set(23) − set(14) = {}` |
| `24__resnet50__light__…__srm-cbam` | CBAM attention | **no CBAM** — `attention_head: "none"` | no `channel_att`/`spatial_att` tensors |
| `25__convnext-small__light__…__srm-gem` | GeM pooling | **no GeM** | no `avgpool.p`; `set(25) − set(21) = {}` |
| `26__convnext-small__light__…__srm-cbam` | CBAM attention | **no CBAM** | 345 keys vs run 18's 348 |

GeM registers `p` as a parameter unconditionally (`attention_heads.py:17-18`), so
its presence *is* detectable in a state dict — runs 15 and 17 have it. Its absence
in 23 and 25 is proven from the weights, not merely inferred from the config.

Runs 21, 25 and 26 are **architecturally identical**. Runs 25 and 26 differ in no
recorded config field at all, which makes them an accidental replicate and the
source of this project's measured **0.15 pp noise floor**.

Also note: several early runs (03, 04, 05) have folder names ending `__none` and
`__ce` while their configs record `use_srm: true` and `loss_type: weighted_focal`.

## Corrupt prediction files — do not use

| Run | training log (`best_val_acc`) | saved `y_pred.npy` | cause |
|---|---|---|---|
| `20__…__fft` | 0.8837 | **0.3491** (chance) | `evaluate.py` cannot detect FFT (the layer has no parameters, so no key name matches). It rebuilt a bare ConvNeXt and loaded **zero** weights — `missing=344, unexpected=344`. The evaluated network was 100% randomly initialised. |
| `18__…__srm-cbam` | 0.8356 | **0.7094** | `evaluate.py:415` detects ConvNeXt-Small by the substring `features.5.20`; CBAM re-nests the features module so keys read `features.0.5.20.*` and the pattern misses. Tiny and Small have identical channel widths, so a truncated 9-block model loaded with **no size mismatch** — `missing=0, unexpected=162`. |

Both were hidden by `load_state_dict(..., strict=False)` at `evaluate.py:617`.
These files should be regenerated once the loader is fixed, or deleted.

## What each number means

- **`best_val_acc`** in `training_summary.json` is a **maximum over epochs** on the
  *model-selection* split. It is a biased estimator and is not a test result.
  Most headline numbers quoted elsewhere in this repo are this quantity.
- **`y_true.npy` / `y_pred.npy`** are the genuine held-out test split — 23,341
  images, class counts 7,795 / 7,792 / 7,754. (The split is 30% rather than the
  configured 15% because of a splitter bug; see `../LIMITATIONS.md` §4.)
- **No logits are saved**, only argmax labels. ROC-AUC is unrecoverable.
- Test accuracy runs ~0.13 pp above `best_val_acc` on average because training
  validates under AMP autocast while `evaluate.py` runs fp32. This is not evidence
  of contamination.

## Subdirectories

| Path | Contents | Status |
|---|---|---|
| `01`–`26` | per-run metrics, curves, predictions | current; see corrections above |
| `archived/` | 8 class-weight sweep runs | complete and verified; excluded from `compare_runs.py`, which only scans the top level |
| `comparison/01`–`08` | dashboard PNGs at earlier milestones | **superseded** by `comparison/09` |
| `comparison/09` | dashboard PNGs, 2026-03-11 | faithful to the artifacts — including run 20 at 34.91%, which was visible and never acted on. Labels run 25 "(best)"; run 25 contains no GeM. |
| `comparison_test/quick/` | 3 PNGs from a `--runs` subset smoke test | **scratch**, undated |
| `tensorboard/` | 2 stray event files, 2026-03-07 | not tied to any run |
| `ablation_study.md` | ablation summary | **corrected 2026-09-08** — see its banner |

Full evidence: [`../docs/REVIEW_2026-09-08.md`](../docs/REVIEW_2026-09-08.md).

## Rebuild runs (OpenSDI, verified-clean data)

Everything below is a frozen encoder plus a small trained head, on the OpenSDI
slice. Accuracy is 3-class balanced accuracy on the run's own test split; where
the split was 58% real (the "real tripled" runs) the balanced figure is
prior-corrected. IoU is mean mask IoU over the 1,350 `ai_edited` test images.
Each directory holds `training_summary.json` with per-epoch history, `y_true`,
`y_pred`, `probs.npy`, and `best_model.pth` (except `mask_head_clipL448`, whose
one-epoch checkpoint is deliberately not committed).

| Path | Encoder | Data | Accuracy | IoU | What it answered |
|---|---|---|---|---|---|
| `linear_probe/` | DINOv2 S/B/L, CLIP B/32 @224 | 3,600 | 0.6593 (S) | - | first honest number; encoder capacity is not the bottleneck (S vs L p = 1.000) |
| `mask_head/` | DINOv2 S @448 | 3,600 | 0.7250 | 0.235 | resolution plus a mask head recovers `ai_edited` |
| `mask_head_large/` | DINOv2 S @448 | 13,500 | 0.7331 | 0.383 | data volume: 3.75x more images, no change |
| `mask_head_weighted/` | DINOv2 S @448 | 13,500 | 0.7328 | 0.385 | class-weighted loss; the DINOv2 reference for paired tests |
| `mask_head_672/` | DINOv2 S @672 | 13,500 | 0.7286 | 0.381 | resolution beyond 448: nothing |
| `mask_head_morereal/` | DINOv2 S @448 | 21,175 | 0.7292 corr. | 0.277 | real tripled: shifts the prior, teaches nothing |
| `mask_head_clip/` | CLIP B/16 @224 | 21,175 | 0.7835 corr. | 0.176 | the encoder was the bottleneck: +10 points |
| `mask_head_clip448/` | CLIP B/16 @448, interpolated pos. emb. | 21,175 | 0.7969 corr. | 0.237 | features and grid stack |
| `mask_head_clipL448/` | CLIP L/14 @448 | 21,175 | partial | 0.254 (1 epoch) | stopped: 2+ hours per epoch under thermal throttling; see `PARTIAL.md` |
| **`mask_head_clip448_balanced/`** | CLIP B/16 @448 | 13,500 | **0.8040** | 0.272 | **final model**; paired with `mask_head_weighted` (p = 2.02e-16); `heldout_generators.json` is the cross-generator result; `decision_rule.json` is the abstain line the frontend reads; `degradation.json` is the compression grid |
| `mask_head_clip448_ft4/` | CLIP B/16 @448, last 4 blocks trained | 13,500 | 0.9015 | 0.401 | fine-tuning: +10 in-domain, held-out mean 0.723 to 0.657, flux `ai_generated` 0.647 to 0.240. Checkpoint (116 MB) not committed; see its README |
| `mask_head_clip448_smoothaug/` | CLIP B/16 @448, smooth-patch aug | 13,500 | 0.7911 | 0.265 | negative: augmenting real images with smoothed regions hurt in-domain (p = 0.025) and held-out (0.689) |

`heldout_generators.json` and `heldout_probs_*.npy` in a directory are the
leave-one-generator-out evaluation of that checkpoint on `data_sources/heldout`.
The version in `mask_head_clip448/` is confounded by that run's real-heavy prior
and should not be compared against `mask_head_weighted/` on raw recall; use
`mask_head_clip448_balanced/` for that comparison.
