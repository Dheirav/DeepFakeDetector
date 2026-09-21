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
ID). 616 of those pairs cross a split boundary.

### 2.1 The full-pool scan (2026-09-08)

The figures above are measured on the 77,865 exported images. The surviving
per-source indexes carry pHash and sha256 for **359,253** images — every file the
indexer saw, not just what was sampled — so the cross-source comparison the
pipeline never performed can be run retrospectively. Over the full pool:

| | count |
|---|---|
| exact (sha256) cross-source duplicate images | 44,347 |
| near-duplicate pairs, pHash Hamming ≤ 3 | 74,840 |
| **of those, cross-class — one picture, two labels** | **23,588** |

| pair | count |
|---|---|
| `coco` [real] ↔ `defacto` [ai_edited] | 12,482 |
| `coco` [real] ↔ `defacto_inpainting` [ai_edited] | 11,076 |
| `openforensics` [ai_edited] ↔ `openimages` [real] | 24 |
| `ffhq` [real] ↔ `stylegan` [ai_generated] | 2 |

The FFHQ↔StyleGAN pair is expected — that GAN was trained on FFHQ. The COCO↔DEFACTO
pairs are the serious ones: they are the same photograph labelled `real` in one
corpus and `ai_edited` in another, on the exact class boundary carrying 69% of the
model's errors.

### 2.2 One source contributed no new images at all

`COCO_Test` is a duplicate of `COCO`. Its filenames and hashes are **100.0%**
contained in the COCO index (40,657 of 40,661). `download_coco_test.py` names
`test2017.zip`, but the indexed content is train2017. Dataset impact: **1,489 of
the 7,000 exported COCO_Test images are byte-identical to an exported COCO image,
and 987 of those straddle a split boundary.**

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

## 9. Limitations of the rebuilt model

Sections 1 to 8 describe the original fine-tuned model. The models trained on
the verified-clean OpenSDI slice are different objects with different problems,
and they should not be quoted without these. Where a number below is from the
linear probe rather than the final CLIP mask head, it says so. The degradation
grid was first run on the probe and has since been repeated on the mask head;
the mask-head figures are the ones quoted here, from
`results/mask_head_clip448_balanced/degradation.json`.

**It degrades under compression, and does not notice.** On 600 held-out images
the final CLIP mask head goes from 0.832 clean to 0.680 at 256px JPEG q80 and
0.605 at 224px q50, against a 0.338 majority baseline (balanced classes). That
is a 15 to 23 point loss, and well above chance, where the linear probe at
256/q80 sat 1.3 points above its baseline and the original ConvNeXt inverted to
0.027. So: better than both earlier models, and still not something to trust on
an image taken off the web. Mask IoU falls 0.284 to 0.223 across the same grid.
The classes trade places as quality drops: at 320/q85 real recall is 0.374
while edited holds 0.695; at 224/q50 real is back to 0.783 and edited has
fallen to 0.251. The model keeps answering; it just changes which class it
dumps the uncertain images into.

The part that matters for the abstain rule: **coverage does not move.** The
model answers 55% of clean images and 50% of 224/q50 images, while its accuracy
when it does answer falls from 0.958 to 0.705. Its confidence is not a function
of image quality, so the "cannot tell" verdict does not protect against a
degraded input. It would need an explicit quality check in front of it, which
it does not have. Full grid in `results/mask_head_clip448_balanced/degradation.json`.

**It does not transfer well to unfamiliar generators.** The final CLIP mask head,
trained on sd15, has `ai_generated` recall of 0.820 on sd2, 0.660 on sd3, 0.635 on
sdxl and 0.647 on flux, and on the three unseen generators it calls 17 to 28
percent of synthetic images real. The linear probe was worse (0.470 on flux,
close to a coin flip). Better is not good: it has learned what sd15 looks like
more than what generated images look like, and the decline with architectural
distance from sd15 is the same shape the OpenSDI paper reports for its own model.

**Localisation transfers far worse than classification.** Mask IoU on the sd15
control is 0.272 in-distribution and 0.251 on the held-out sd15 shard, then 0.222
on sd3 and 0.161 on sdxl and flux. That is a 36 percent fall against 10 percent
for `ai_edited` recall on the same images. The model still flags an edited image
from an unseen generator while largely losing track of where the edit is, and
the paper's own MaskCLIP shows the same asymmetry (76 against 26 percent).

**Localisation is weak even in-distribution.** 0.272 IoU against the paper's
0.671. Swapping DINOv2 for CLIP raised classification 7 points and lowered the
control IoU from 0.353 to 0.251, so the encoder that classifies best does not
localise best. Nothing tried so far moves both in the same direction.

**`ai_edited` was the weak class and is now merely the weakest.** 0.465 F1 on the
linear probe at 224px, because a local edit covering roughly 1.7 percent of the
frame does not survive that resize; 0.722 on the CLIP mask head at 448px. That
class is where all of the improvement in the project came from, and where the
remaining in-distribution errors concentrate.

**The 80.4% is in-distribution.** It is a stratified random split within sampled
sd15 shards, so it measures performance on a clean dataset rather than transfer.
The cross-generator figures above are the transfer numbers, and they are lower.

**The training set is 7 percent of the dataset's.** 13,500 images against
200,000 in the paper. Doubling the data during the DINOv2 experiments changed
nothing measurable, but that was tested on the encoder that turned out to be the
bottleneck, so the data-volume result may not carry over to CLIP.

**The encoder is frozen, and unfreezing it is a trade, not a fix.** Training the
last 4 of 12 CLIP blocks lifts the in-distribution number to 0.9015 and drops
`ai_generated` recall on flux from 0.647 to 0.240 and on sdxl from 0.635 to
0.328. The encoder learns the training generator's fingerprint. The frozen
model is the headline because the held-out number is the one that matters for
an image of unknown origin; the fine-tuned checkpoint is documented in
`results/mask_head_clip448_ft4/README.md` and not committed (116 MB).

**It calls smooth surfaces edited.** The first real photo tested, a conference
room with a glossy red tablecloth and a laptop lid, came back `ai_edited` at
0.82 with the mask on the tablecloth. Every inpainted region in the training
data is smooth and low-noise and the decoder uses that as its cue. Augmenting
real training images with smoothed regions to break the cue made every measured
number worse (0.8040 to 0.7911 in-distribution, held-out mean 0.723 to 0.689),
which says the frozen features carry no better cue to fall back on. Held-out
`real` recall is 0.93, so roughly one real photo in fourteen is affected.

**Its probabilities are not calibrated.** Softmax outputs cluster near 0 and 1
regardless of correctness. The UI and CLI therefore abstain below a top
probability of 0.90, chosen by measuring coverage against accuracy on the saved
test probabilities: 56% of in-distribution images answered, 94% right when
answered, and the confident-wrong rate on sd2 real photos falls from 6.8% to
0.5%. That is a decision rule around an uncalibrated model, not calibration.

**One real photo is not a false-positive rate.** The tablecloth case is a
worked example. A set of 30 or more photos provably outside every dataset is
needed to turn it into a number, and does not exist yet.

**The `ai_generated` cross-generator column has no control.** Both sd15 test shards
sampled happened to hold `ai_edited`, so that column cannot separate "unseen
generator" from "unseen images". The `ai_edited` comparison does have its control
and should be preferred where the distinction matters.

**Reproducibility caveat.** Shard selection was chosen by hand after probing the
layout, and OpenSDI's shards are class-ordered, so a different selection would
give a different class mix. The exact shards used are recorded in
`dataset_builder/tools/convert_opensdi.py`.

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
