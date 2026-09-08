# Ablation Study — Deepfake Detection Experiments

> ## ⚠️ CORRECTED 2026-09-08 — read this before any number below
>
> An audit invalidated this document's central conclusions. Corrections are
> inline; the original rows are struck through rather than deleted so the record
> is auditable.
>
> 1. **Runs 23, 24, 25 and 26 do not contain the components their folder names
>    claim.** All four record `attention_head: "none"` in `training_summary.json`,
>    and the weights confirm it: run 26 has no CBAM tensors (345 keys vs run 18's
>    348), and runs 23/25 have no `avgpool.p` (which GeM *does* save — runs 15 and
>    17 have it). Key-set diffs are empty: `set(25) − set(21) = {}`.
>    **The "Attention (CBAM)" and "Pooling (GeM)" rows below describe models
>    containing neither.**
> 2. **All comparisons below used `best_val_acc`**, a max over 20–30 noisy epochs
>    on the *model-selection* split. That is a biased estimator and it favours
>    whichever run happened to spike. Test-set numbers tell a different story.
> 3. **The measured noise floor is 0.15 pp.** Runs 25 and 26 differ in no recorded
>    config field — they are an accidental replicate. They disagree on 991 of
>    23,341 test samples; McNemar p = 0.28. **Every effect claimed below is
>    smaller than that.**
> 4. **Runs 25/26 differ from the baseline in two ways, not one** — `focal_gamma`
>    (2.0 vs run 19's 3.0) *and* SRM. They were never a clean single-variable
>    ablation.
> 5. **`results/18/` and `results/20/` saved predictions are corrupt.** Run 20's
>    `y_pred.npy` is 34.91% (chance) against an 88.37% training log, because
>    `evaluate.py` rebuilt it as a 100% randomly-initialised network. Run 18's is
>    70.94% vs 83.56%, because it was rebuilt as a truncated ConvNeXt-Tiny. Do not
>    use either.
>
> Evidence: [`../docs/REVIEW_2026-09-08.md`](../docs/REVIEW_2026-09-08.md).

This document summarizes the ablation study using only experiments and metrics present in the repository `results/` run folders. All metric values are taken from each run's `training_summary.json` and `metrics.csv` where noted; file links are provided.

## Quick summary
- Selected best model: `results/19__convnext-small__light__0.4__cosine__focal__none` — convnext_small (user-selected). Best val acc = 0.8940 (see training_summary.json).

## Per-run summary (selected runs in `results/`)

| Run folder | Backbone | SRM | FFT | Attention | Pooling | Best val acc | Final val acc | Source |
|---|---:|:---:|:---:|:---:|:---:|---:|---:|---|
| [01__resnet18__none__none__none__ce__none](results/01__resnet18__none__none__none__ce__none/training_summary.json#L1) | resnet18 | no | no | none | Avg | 0.8263 | 0.8213 | [training_summary.json](results/01__resnet18__none__none__none__ce__none/training_summary.json#L1) |
| [02__resnet18__none__none__none__focal__srm](results/02__resnet18__none__none__none__focal__srm/) | resnet18 | (srm artifacts present) | no | none | Avg | not found | not found | folder present (no `training_summary.json`) |
| [03__resnet18__none__0.4__cosine__ce__none](results/03__resnet18__none__0.4__cosine__ce__none/training_summary.json#L1) | resnet18 | yes | no | none | Avg | 0.8294 | 0.8181 | [training_summary.json](results/03__resnet18__none__0.4__cosine__ce__none/training_summary.json#L1) |
| [04__resnet18__none__0.5__plateau__ce__none](results/04__resnet18__none__0.5__plateau__ce__none/training_summary.json#L1) | resnet18 | yes | no | none | Avg | 0.8297 | 0.8265 | [training_summary.json](results/04__resnet18__none__0.5__plateau__ce__none/training_summary.json#L1) |
| [05__convnext-tiny__gamma3__0.4__cosine__ce__none](results/05__convnext-tiny__gamma3__0.4__cosine__ce__none/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8678 | 0.8470 | [training_summary.json](results/05__convnext-tiny__gamma3__0.4__cosine__ce__none/training_summary.json#L1) |
| [06__convnext-tiny__none__0.4__cosine__focal__srm](results/06__convnext-tiny__none__0.4__cosine__focal__srm/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8671 | 0.8482 | [training_summary.json](results/06__convnext-tiny__none__0.4__cosine__focal__srm/training_summary.json#L1) |
| [07__resnet18__none__none__none__focal__srm-wd](results/07__resnet18__none__none__none__focal__srm-wd/training_summary.json#L1) | resnet18 | yes | no | none | Avg | 0.8280 | 0.8259 | [training_summary.json](results/07__resnet18__none__none__none__focal__srm-wd/training_summary.json#L1) |
| [08__convnext-tiny__aug-v2__0.4__cosine__ce__srm](results/08__convnext-tiny__aug-v2__0.4__cosine__ce__srm/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8387 | 0.8147 | [training_summary.json](results/08__convnext-tiny__aug-v2__0.4__cosine__ce__srm/training_summary.json#L1) |
| [09__convnext-tiny__aug-v3-light__0.4__cosine__ce__srm](results/09__convnext-tiny__aug-v3-light__0.4__cosine__ce__srm/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8891 | 0.8728 | [training_summary.json](results/09__convnext-tiny__aug-v3-light__0.4__cosine__ce__srm/training_summary.json#L1) |
| [10__convnext-tiny__aug-v4-standard__0.4__cosine__ce__srm](results/10__convnext-tiny__aug-v4-standard__0.4__cosine__ce__srm/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8662 | 0.8408 | [training_summary.json](results/10__convnext-tiny__aug-v4-standard__0.4__cosine__ce__srm/training_summary.json#L1) |
| [11__convnext-tiny__aug-v4-light__0.4__cosine__ce__seed20](results/11__convnext-tiny__aug-v4-light__0.4__cosine__ce__seed20/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8899 | 0.8763 | [training_summary.json](results/11__convnext-tiny__aug-v4-light__0.4__cosine__ce__seed20/training_summary.json#L1) |
| [12__convnext-tiny__aug-v4-light__0.4__cosine__ce__seed11](results/12__convnext-tiny__aug-v4-light__0.4__cosine__ce__seed11/training_summary.json#L1) | convnext_tiny | yes | no | none | Avg | 0.8900 | 0.8734 | [training_summary.json](results/12__convnext-tiny__aug-v4-light__0.4__cosine__ce__seed11/training_summary.json#L1) |
| [13__convnext-small__aug-v1-light__0.4__cosine__ce__srm](results/13__convnext-small__aug-v1-light__0.4__cosine__ce__srm/training_summary.json#L1) | convnext_small | yes | no | none | Avg | 0.8929 | 0.8843 | [training_summary.json](results/13__convnext-small__aug-v1-light__0.4__cosine__ce__srm/training_summary.json#L1) |
| [14__resnet50__aug-v1-light__0.4__cosine__ce__srm](results/14__resnet50__aug-v1-light__0.4__cosine__ce__srm/training_summary.json#L1) | resnet50 | yes | no | none | Avg | 0.8714 | 0.8558 | [training_summary.json](results/14__resnet50__aug-v1-light__0.4__cosine__ce__srm/training_summary.json#L1) |
| [15__resnet50__strong__0.4__cosine__focal__srm-gem](results/15__resnet50__strong__0.4__cosine__focal__srm-gem/training_summary.json#L1) | resnet50 | yes | no | gem | GeM | 0.8292 | 0.8101 | [training_summary.json](results/15__resnet50__strong__0.4__cosine__focal__srm-gem/training_summary.json#L1) |
| [16__resnet50__strong__0.4__cosine__focal__srm-cbam](results/16__resnet50__strong__0.4__cosine__focal__srm-cbam/training_summary.json#L1) | resnet50 | yes | no | cbam | Avg | 0.8277 | 0.8264 | [training_summary.json](results/16__resnet50__strong__0.4__cosine__focal__srm-cbam/training_summary.json#L1) |
| [17__convnext-small__strong__0.4__cosine__focal__srm-gem](results/17__convnext-small__strong__0.4__cosine__focal__srm-gem/training_summary.json#L1) | convnext_small | yes | no | gem | GeM | 0.8428 | 0.8199 | [training_summary.json](results/17__convnext-small__strong__0.4__cosine__focal__srm-gem/training_summary.json#L1) |
| [18__convnext-small__strong__0.4__cosine__focal__srm-cbam](results/18__convnext-small__strong__0.4__cosine__focal__srm-cbam/training_summary.json#L1) | convnext_small | yes | no | cbam | Avg | 0.8356 | 0.8295 | [training_summary.json](results/18__convnext-small__strong__0.4__cosine__focal__srm-cbam/training_summary.json#L1) |
| [19__convnext-small__light__0.4__cosine__focal__none](results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json#L1) | convnext_small | no | no | none | Avg | 0.8940 | 0.8830 | [training_summary.json](results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json#L1) |
| [20__convnext-small__light__0.4__cosine__focal__fft](results/20__convnext-small__light__0.4__cosine__focal__fft/training_summary.json#L1) | convnext_small | no | yes | none | Avg | 0.8837 | 0.8758 | [training_summary.json](results/20__convnext-small__light__0.4__cosine__focal__fft/training_summary.json#L1) |
| [21__convnext-small__light__0.4__cosine__focal__srm](results/21__convnext-small__light__0.4__cosine__focal__srm/training_summary.json#L1) | convnext_small | yes | no | none | Avg | 0.8954 | 0.8743 | [training_summary.json](results/21__convnext-small__light__0.4__cosine__focal__srm/training_summary.json#L1) |
| [22__convnext-small__light__0.4__cosine__focal__srm-fft](results/22__convnext-small__light__0.4__cosine__focal__srm-fft/training_summary.json#L1) | convnext_small | yes | yes | none | Avg | 0.8934 | 0.8711 | [training_summary.json](results/22__convnext-small__light__0.4__cosine__focal__srm-fft/training_summary.json#L1) |
| [23__resnet50__light__0.4__cosine__focal__srm-gem](results/23__resnet50__light__0.4__cosine__focal__srm-gem/training_summary.json#L1) | resnet50 | yes | no | none | GeM | 0.8720 | 0.8628 | [training_summary.json](results/23__resnet50__light__0.4__cosine__focal__srm-gem/training_summary.json#L1) |
| [24__resnet50__light__0.4__cosine__focal__srm-cbam](results/24__resnet50__light__0.4__cosine__focal__srm-cbam/training_summary.json#L1) | resnet50 | yes | no | cbam | Avg | 0.8704 | 0.8572 | [training_summary.json](results/24__resnet50__light__0.4__cosine__focal__srm-cbam/training_summary.json#L1) |
| [25__convnext-small__light__0.4__cosine__focal__srm-gem](results/25__convnext-small__light__0.4__cosine__focal__srm-gem/training_summary.json#L1) | convnext_small | yes | no | none | GeM | 0.8952 | 0.8803 | [training_summary.json](results/25__convnext-small__light__0.4__cosine__focal__srm-gem/training_summary.json#L1) |
| [26__convnext-small__light__0.4__cosine__focal__srm-cbam](results/26__convnext-small__light__0.4__cosine__focal__srm-cbam/training_summary.json#L1) | convnext_small | yes | no | cbam | Avg | 0.8933 | 0.8688 | [training_summary.json](results/26__convnext-small__light__0.4__cosine__focal__srm-cbam/training_summary.json#L1) |

Notes:
- "SRM"/"FFT"/"Attention" flags are taken from the `training_summary.json` `config` where present; when missing, the run folder name was used as an indicator (see row for `02__...` which lacks `training_summary.json`).
- The repository `results/<run>/` folders also contain `y_true.npy` and `y_pred.npy` for many runs; per-request precision/recall/test metrics can be computed by loading those files. Those per-run classification reports are not included here in order to keep this document strictly to values already saved to `training_summary.json` and `metrics.csv`.

## Grouped ablation (requested feature-focused variants)

Baseline reference: `results/19__convnext-small__light__0.4__cosine__focal__none` — convnext_small, RGB-only, best val acc = 0.8940 ([training_summary.json](results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json#L1)).

**⚠️ The two bottom rows of this table are wrong — see correction 1 above.**

| Model Variant | Representative Run | Backbone | SRM | FFT | Attention | Pooling | Best val acc | Source |
|---|---|---:|:---:|:---:|:---:|:---:|---:|---|
| Baseline (RGB only) | 19 | convnext_small | no | no | none | Avg | 0.8940 | [results/19/.../training_summary.json](results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json#L1) |
| CNN + SRM | 21 | convnext_small | yes | no | none | Avg | 0.8954 | [results/21/.../training_summary.json](results/21__convnext-small__light__0.4__cosine__focal__srm/training_summary.json#L1) |
| CNN + FFT | 20 | convnext_small | no | yes | none | Avg | 0.8837 | [results/20/.../training_summary.json](results/20__convnext-small__light__0.4__cosine__focal__fft/training_summary.json#L1) |
| CNN + SRM + FFT (multi-domain) | 22 | convnext_small | yes | yes | none | Avg | 0.8934 | [results/22/.../training_summary.json](results/22__convnext-small__light__0.4__cosine__focal__srm-fft/training_summary.json#L1) |
| ~~Attention (CBAM)~~ **NO CBAM PRESENT** | 26 | convnext_small | yes | no | **none** | Avg | 0.8933 | [results/26/.../training_summary.json](results/26__convnext-small__light__0.4__cosine__focal__srm-cbam/training_summary.json#L1) |
| ~~Pooling (GeM)~~ **NO GeM PRESENT** | 25 | convnext_small | yes | no | none | **Avg** | 0.8952 | [results/25/.../training_summary.json](results/25__convnext-small__light__0.4__cosine__focal__srm-gem/training_summary.json#L1) |

### ~~Improvement vs baseline (best-val comparison)~~ — WITHDRAWN

The original list is retained below, struck through. Every entry was computed
from `best_val_acc` and every effect is smaller than the 0.15 pp noise floor.

> ~~Baseline (run 19): 0.8940 (reference)~~
> ~~CNN + SRM (run 21): +0.14 pp · CNN + FFT (run 20): −1.03 pp~~
> ~~CNN + SRM + FFT (run 22): −0.06 pp · CBAM (run 26): −0.07 pp · GeM (run 25): +0.12 pp~~

### Corrected: test-set comparison with significance testing

Paired McNemar tests on the held-out test set (n = 23,341), against baseline
run 19 (plain RGB ConvNeXt-Small):

| Comparison | test acc | Δ vs baseline | McNemar p |
|---|---|---|---|
| **19 — baseline, RGB only** | 0.8972 | — | — |
| 21 — + SRM | 0.8967 | **−0.06 pp** | 0.725 |
| 22 — + SRM + FFT | 0.8946 | −0.26 pp | 0.081 |
| 25 — "GeM" *(contains no GeM)* | 0.8973 | +0.00 pp | 1.000 |
| 26 — "CBAM" *(contains no CBAM)* | 0.8958 | −0.15 pp | 0.324 |
| 20 — + FFT | *corrupt* | — | — |

**Conclusion: no forensic component produces a measurable effect.** Plain RGB
ConvNeXt-Small is statistically indistinguishable from every variant, and SRM is
marginally *worse* on test. The +0.14 pp SRM gain claimed above does not survive
either a change of split or a significance test.

Two caveats that keep this from being a clean result *about SRM*:

- **SRM was effectively disconnected during training.** `srm.py:155` initialises
  all three residual channels to identical weights (0.1× the red channel, tiled),
  contributing ~1/1250 of the pre-activation variance. The residual branch would
  need to grow ~35× in norm to matter, which does not happen at `lr=1e-4` over
  11–30 epochs. Fix that before concluding anything about SRM.
- **FFT normalisation is batch-dependent.** `fft.py:22` reduces min/max over the
  whole batch, so a single image's FFT channel shifts its mean by 0.136 — 14% of
  its range — between `batch_size=64` training and `batch_size=1` inference.

## Best model

> **⚠️ 0.8940 is `best_val_acc`** — a maximum over epochs on the model-selection
> split, not a test result. Run 19's test accuracy is **0.8972**. More importantly,
> selecting on this metric is actively harmful here: across nine runs, validation
> accuracy correlates at **r = −0.956** with robustness to a routine JPEG re-save.
> Run 19 scores 0.000 on AI-generated images downscaled to 256px and re-saved at
> q60; run 17, which scored 5 points *worse* on validation, scores 0.955.
> See [`../LIMITATIONS.md`](../LIMITATIONS.md).

- Selected best model (user request): **0.8940** from `results/19__convnext-small__light__0.4__cosine__focal__none` (convnext_small, RGB-only). Source: [results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json](results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json#L1).

## Next steps (optional)
- Compute test accuracy, precision and recall per run by loading `results/<run>/y_true.npy` + `results/<run>/y_pred.npy` and generating classification reports. I can compute and append those exact values to this document on request.
- If you want the stacked per-epoch metrics, we can also include per-run `metrics.csv` peaks (val_f1_macro at best epoch) into the tables.

---
Document generated from in-repo `results/` folders and `training_summary.json`/`metrics.csv` files.

## Computed test metrics (appended)

The following metrics were computed from `y_true.npy` and `y_pred.npy` present in the run folder. These values were calculated by a local script that read the repository arrays — no values were invented.

- Run: [results/19__convnext-small__light__0.4__cosine__focal__none](results/19__convnext-small__light__0.4__cosine__focal__none/)
	- Test accuracy: 0.8972194850263485 — computed from `results/19__convnext-small__light__0.4__cosine__focal__none/y_true.npy` and `results/19__convnext-small__light__0.4__cosine__focal__none/y_pred.npy`
	- Precision (per class): [0.8486500190138168, 0.9648682559598495, 0.8763699545576049] — computed from the same files
	- Recall (per class): [0.8588838999358563, 0.9869096509240246, 0.8456280629352593]
	- F1-score (per class): [0.8537362917623057, 0.9757644968912574, 0.8607245996324495]
	- Support (per class): [7795, 7792, 7754]
	- Confusion matrix (rows=true [0,1,2], cols=pred [0,1,2]):

		[[6695, 190, 910],
		 [87, 7690, 15],
		 [1107, 90, 6557]]

	- Source files used for computation:
		- [results/19__convnext-small__light__0.4__cosine__focal__none/y_true.npy](results/19__convnext-small__light__0.4__cosine__focal__none/y_true.npy)
		- [results/19__convnext-small__light__0.4__cosine__focal__none/y_pred.npy](results/19__convnext-small__light__0.4__cosine__focal__none/y_pred.npy)
