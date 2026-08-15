# Backlog

Ordered by impact. The first item invalidates the headline number, so nothing
below it is worth doing until it is done.

---

## 1. The validation number does not measure generalisation

**The problem.** Every source dataset maps to exactly one class, and no source
is shared between classes:

| Class | Sources |
|---|---|
| `real` | COCO, COCO_Test, FFHQ, ImageNet, OpenImages, Places365 |
| `ai_generated` | FLUX, FLUX_TopUp, Midjourney_DALLE, Midjourney_TopUp, SD_TopUp, SD_TopUp2, StableDiffusion, StableDiffusion_TopUp, StyleGAN, Synthbuster |
| `ai_edited` | CASIA, DEFACTO, DEFACTO_Inpainting, FaceForensics, ForgeryNet, IMD2020, OpenForensics |

`create_dataloaders` splits with `train_test_split(range(len(dataset)),
random_state=42)` — a random split over the pooled files. Train and validation
therefore contain the same corpora.

Corpus identity is a perfect predictor of the label, and it is a far easier
signal to learn than manipulation traces: quantisation tables, resampling
history, colour profile, capture pipeline. A network reaches 89% by answering
*"which dataset is this from"* and never learns *"has this been manipulated"*.

This is why an arbitrary photograph is misclassified. It belongs to none of the
six real corpora, so there is no "this looks like COCO" evidence to call it
real; it is assigned to whichever corpus signature it superficially resembles.

**The number is real. It is not a generalisation estimate.**

### 1.1 Measure the truth: leave-one-source-out validation

Hold out whole *sources*, not random files — train without Synthbuster, then
evaluate only on Synthbuster; repeat per source. Report the spread, not just the
mean. Expect a large drop. That figure is the honest one and everything else is
guesswork until it exists.

### 1.2 Break the confound where the data allows

FaceForensics++ and ForgeryNet ship **paired** originals and manipulations.
Using their originals as the `real` class for their own fakes gives matched
provenance — same capture, same codec, same pipeline — so the only remaining
difference is the manipulation. Highest-value single change here.

### 1.3 Destroy corpus signatures with augmentation

Random JPEG re-encoding across a range of qualities, random resize and crop,
mild blur. A shortcut that cannot survive the augmentation forces the model onto
something else. `preprocessing.py` already has the structure for this.

### 1.4 Re-run the ablation under leave-one-source-out

The SRM and FFT variants operate on residual and frequency content rather than
semantics — exactly the features that should transfer across corpora. Under an
honest split they may beat the ConvNeXt that currently wins on the random split.

---

## 2. Delete the placeholder dataloader

`scripts/dataloader/dataset_loader.py` defines `default_transforms` as
`Resize + ToTensor` with **no** `Normalize`, and is commented "(Placeholder)".
Training does not use it — `train_full.py` goes through
`preprocessing.py`, which normalises — but inference in `frontend/inference.py`
does normalise. Anyone who wires the placeholder up gets a silent train/inference
mismatch. It is dead code; remove it.

---

## 3. Reconcile the reported accuracy

`README.md` headlines 82.73%, traced to `model_cards/sweep_w200_100_200.md`.
`results/ablation_study.md` names run 19 (`convnext_small`) as the selected best
at 89.40% validation accuracy. Both may be correct while measuring different
things — single-stage validation versus the end-to-end three-way cascade — but
the repository does not say which is which. State what each number measures, and
publish the cascade's end-to-end result now that the evaluation is committed.

---

## 4. No CI

There is no `.github/workflows`. At minimum: lint, import-check every script, and
run the evaluation scripts against a small fixture so a refactor cannot silently
break the pipeline.
