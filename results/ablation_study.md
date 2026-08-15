# Ablation Study — Deepfake Detection Experiments

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

| Model Variant | Representative Run | Backbone | SRM | FFT | Attention | Pooling | Best val acc | Source |
|---|---|---:|:---:|:---:|:---:|:---:|---:|---|
| Baseline (RGB only) | 19 | convnext_small | no | no | none | Avg | 0.8940 | [results/19/.../training_summary.json](results/19__convnext-small__light__0.4__cosine__focal__none/training_summary.json#L1) |
| CNN + SRM | 21 | convnext_small | yes | no | none | Avg | 0.8954 | [results/21/.../training_summary.json](results/21__convnext-small__light__0.4__cosine__focal__srm/training_summary.json#L1) |
| CNN + FFT | 20 | convnext_small | no | yes | none | Avg | 0.8837 | [results/20/.../training_summary.json](results/20__convnext-small__light__0.4__cosine__focal__fft/training_summary.json#L1) |
| CNN + SRM + FFT (multi-domain) | 22 | convnext_small | yes | yes | none | Avg | 0.8934 | [results/22/.../training_summary.json](results/22__convnext-small__light__0.4__cosine__focal__srm-fft/training_summary.json#L1) |
| Attention (CBAM) | 26 | convnext_small | yes | no | cbam | Avg | 0.8933 | [results/26/.../training_summary.json](results/26__convnext-small__light__0.4__cosine__focal__srm-cbam/training_summary.json#L1) |
| Pooling (GeM) | 25 | convnext_small | yes | no | none | GeM | 0.8952 | [results/25/.../training_summary.json](results/25__convnext-small__light__0.4__cosine__focal__srm-gem/training_summary.json#L1) |

### Improvement vs baseline (best-val comparison)
- Baseline (run 19): 0.8940 (reference)
- CNN + SRM (run 21): 0.8954 → +0.0014 (+0.14 percentage points)
- CNN + FFT (run 20): 0.8837 → −0.0103 (−1.03 points)
- CNN + SRM + FFT (run 22): 0.8934 → −0.0006 (−0.06 points)
- Attention (CBAM, run 26): 0.8933 → −0.0007 (−0.07 points)
- Pooling (GeM, run 25): 0.8952 → +0.0012 (+0.12 points)

## Best model
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
