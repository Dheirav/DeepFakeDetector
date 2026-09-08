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
