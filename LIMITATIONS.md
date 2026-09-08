# Limitations

Last updated 2026-09-08, after a full audit of the code, the dataset build and
every reported number.

This document exists because the headline result in this repository — ~89% on a
held-out test set — **does not measure what it appears to measure**. Everything
below was verified by running something. Where a claim elsewhere in the repo
contradicts this file, this file is correct.

---

## 1. The model does not detect AI images. It recognises source corpora.

Every one of the 20 source datasets maps to exactly one class, and no source is
shared between classes. Source→class purity is **100.0%** across all 20. Corpus
identity is therefore a perfect substitute for the label, and it is a far easier
signal to learn than manipulation traces.

**Evidence 1 — the file header alone beats the network.** A lookup table trained
on `(file format, width, height)` from the training split and evaluated on the
test split, reading **no pixels at all**:

| Feature | 3-class test accuracy |
|---|---|
| majority-class baseline | 33.40% |
| container format alone | 59.11% |
| resolution alone | 79.88% |
| **format + resolution** | **87.40%** |
| format + resolution + bytes-per-pixel *(in-sample)* | 96.71% |
| the trained model | ~89% |

All 13,905 TIFF files in the dataset are `ai_edited` — 100% precision, 53.8% of
that class. `format == TIFF → ai_edited` classifies 17.9% of the dataset perfectly
without decoding an image.

**Evidence 2 — a routine JPEG re-save inverts the prediction.** 300 FLUX images,
all genuinely AI-generated, through run 19. Every transform below preserves
whether an image is AI-generated:

| transform | accuracy | prediction split |
|---|---|---|
| none | 0.993 | real 2, ai_gen 298, ai_edit 0 |
| resize to 512 | 0.997 | real 1, ai_gen 299 |
| resize to 300 | 0.997 | real 1, ai_gen 299 |
| JPEG q75 at native size | 0.957 | real 3, ai_gen 287, ai_edit 10 |
| centre-crop 70% | 0.983 | real 4, ai_gen 295 |
| **resize to 256 + JPEG q80** | **0.027** | **real 280**, ai_gen 8, ai_edit 12 |

Downscale and re-encode — what happens to any image on its way across the web —
and **280 of 300 AI-generated images become "real" at 87% mean confidence.**

The full decision surface, `P(correct)` over resolution × JPEG quality on the
same 200 images:

```
  size    no-jpeg     q95     q85     q75     q60
  1024      0.990   0.990   0.985   0.960   0.925
   768      0.995   0.985   0.965   0.950   0.880
   512      0.995   0.990   0.980   0.945   0.675
   384      0.995   0.985   0.935   0.650   0.395
   320      0.995   0.970   0.680   0.380   0.160
   256      0.990   0.675   0.080   0.005   0.000
```

The output is a smooth function of resolution and compression and is largely
independent of image content. Resolution alone is harmless (0.990–0.995 down the
no-JPEG column); it is the **JPEG artefact scale relative to image size** that
carries the signal — which is the encoding signature of the `real` corpora
(COCO, Places365, OpenImages are mid-resolution JPEGs) against the large, clean
files of the generated corpora (FLUX is 1024×1024 in 100% of sampled images).

**Evidence 3 — the `ai_edited` score is too high to be real.** The model reports
0.86 F1 on `ai_edited`. DEFACTO images average **1.7% tampered pixels**, and
published methods score **0.8–6.9%** on tampered-image detection at 224px while
scoring 83–94% on fully-synthetic images. A healthy `ai_edited` number at this
resolution is itself evidence of a shortcut.

## 2. Train/test contamination

Deduplication and cluster-splitting run **once per source**, so nothing is ever
compared across corpora. The pipeline's own leakage assertion cannot fail: every
one of the 20 runs produced only singleton clusters (`Cluster size P99: 1`) within
a single class.

| measure | count | % of test |
|---|---|---|
| test images byte-identical (sha256) to a train image | **1,012** | **4.34%** |
| test images pHash-identical to a train image | **1,656** | **7.09%** |
| near-duplicate pairs spanning splits | 27,712 | — |
| near-duplicate pairs spanning train↔test | 9,871 | — |

Additionally, **743 COCO images appear both as a `real` example and as the base
image of a DEFACTO `ai_edited` example** (DEFACTO filenames embed the COCO image
ID). 616 of those pairs cross a split boundary. Confirmed independently by pHash:
880 of 903 cross-class near-duplicate pairs are COCO↔DEFACTO.

## 3. Model selection optimised for the shortcut

Validation accuracy is **inversely** correlated with robustness. Across nine runs,
Pearson **r = −0.956** between `best_val_acc` and mean accuracy under degradation:

| run | augmentation | val acc | accuracy at 256px/q60 |
|---|---|---|---|
| 19 *(shipped)* | light | 0.894 | **0.000** |
| 21 | light | 0.895 | 0.000 |
| 10 | standard | 0.866 | 0.650 |
| 17 | strong | 0.843 | **0.955** |
| 18 | strong | 0.836 | 0.935 |

Strong augmentation destroys the corpus fingerprint, so the validation set — which
shares that fingerprint — punishes it. The selection procedure therefore rejected
every robust model. **Run 17 was already trained and was discarded for scoring 5
points lower on a metric measuring the wrong thing.**

## 4. The split ratios are wrong

Configured 70/15/15; actual **40.05/29.98/29.98** in all 20 source builds
(31,183 train / 23,341 val / 23,341 test). Cause: `splitter.py:87` normalises each
split's deviation by its own target, so the greedy equalises *relative* fill and
pins the small splits at 2× target. Every model here trained on 31k images rather
than the intended 54k.

## 5. No forensic component has a measurable effect

Paired McNemar on the test set (n = 23,341) against plain RGB ConvNeXt-Small:

| variant | Δ test acc | p |
|---|---|---|
| + SRM | −0.06 pp | 0.725 |
| + SRM + FFT | −0.26 pp | 0.081 |
| "GeM" *(contains no GeM)* | +0.00 pp | 1.000 |
| "CBAM" *(contains no CBAM)* | −0.15 pp | 0.324 |

Measured noise floor, from two runs with identical configs: **0.15 pp**
(McNemar p = 0.28, disagreeing on 991 of 23,341 samples).

Two implementation faults mean these are not yet clean results *about* SRM and FFT:
`srm.py:155` initialises all three residual channels identically at ~1/1250 of the
pre-activation variance, and `fft.py:22` normalises over the whole batch so
features depend on batch composition.

## 6. Four checkpoints do not contain the components their names claim

Runs 23, 24, 25 and 26 are named `srm-gem` / `srm-cbam`. All four record
`attention_head: "none"`, and the weights agree — no CBAM tensors, no `avgpool.p`.
Key-set diffs against the plain-SRM runs are empty. See `results/README.md`.

## 7. Reproducibility

- **Directory iteration order changes 65% of split assignments** at a fixed seed
  (`indexer.py:83` uses unsorted `os.walk`). The determinism claim holds only
  against a cached index, not against the data.
- **Only 1 of 27 checkpoints loads correctly** in both `evaluate.py` and the
  frontend today. `evaluate.py:461` raises `TypeError` on every SRM checkpoint.
- **`results/18/` and `results/20/` saved predictions are corrupt** — 70.94% and
  34.91% (chance) against training logs of 83.56% and 88.37%, because the
  evaluator rebuilt the wrong architecture and `strict=False` hid it.
- **No logits are saved anywhere**, only argmax labels. ROC-AUC and calibration
  are unrecoverable from the stored artifacts.
- **`requirements.txt` has no version pins.**
- The exported image directories are empty; only the per-source metadata survives.

## 8. Grad-CAM figures are colour-inverted

`frontend/gradcam.py:222` blends a BGR colormap into an RGB image. Measured:
importance 1.0 renders **blue**, importance 0.0 renders **red** — the inverse of
what `app.py:203` tells the reader. Every heatmap figure produced before this fix
is wrong. The target-layer selection is also incorrect (Pearson r = 0.598 against
the canonical CAM for ConvNeXt; for CBAM models it selects a 1-channel layer,
making the output class-independent).

---

## What this repository currently demonstrates

Not a working detector. It demonstrates, with measurements, **how a
standard-practice dataset build produces a detector that scores 89% while reading
file metadata** — and it contains the instruments that show it: a metadata-only
probe, a degradation curve, a selection-inversion analysis, and a leakage
quantification.

Work to correct it is tracked in [`docs/SALVAGE_PLAN.md`](docs/SALVAGE_PLAN.md).
Full evidence: [`docs/REVIEW_2026-09-08.md`](docs/REVIEW_2026-09-08.md),
[`docs/DATASET_BUILDER_AUDIT.md`](docs/DATASET_BUILDER_AUDIT.md),
[`docs/GENERALISATION_LITERATURE.md`](docs/GENERALISATION_LITERATURE.md).
