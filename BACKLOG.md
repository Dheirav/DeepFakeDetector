# Backlog

Ordered by impact. The first item invalidates the headline number, so nothing
below it is worth doing until it is done.

---

## 1. The validation number does not measure generalisation

**The problem.** Every source dataset maps to exactly one class, and no source
is shared between classes:

| Class | Sources |
|---|---|
| `real` | COCO, COCO_Test, FFHQ, OpenImages, Places365 |
| `ai_generated` | FLUX, FLUX_TopUp, Midjourney_DALLE, Midjourney_TopUp, SD_TopUp, SD_TopUp2, StableDiffusion, StyleGAN, Synthbuster |
| `ai_edited` | CASIA, DEFACTO, DEFACTO_Inpainting, FaceForensics, IMD2020, OpenForensics |

*Corrected 2026-09-08: earlier versions of this table listed **ImageNet** and
**ForgeryNet**. Config files exist for both, but no artifacts were ever built and
neither contributed a single image to the 77,865. ImageNet was additionally 4×
Lanczos-upscaled to lossless PNG before removal — see the audit.*

*Mechanism corrected 2026-09-08.* This paragraph previously blamed
`create_dataloaders` in `scripts/dataloader/dataset_loader.py`. That file is dead
code — it has never run (`torch` is not imported, so line 76 raises `NameError`),
as item 2 below says. Training goes through `get_data_loaders` in `train_full.py`,
which reads the pre-built `dataset_builder/val` directory.

**The conclusion stands regardless**, because the confound is a property of the
dataset, not the splitter: the builder's greedy assignment explicitly *balances*
source distribution across splits (`splitter.py:96-99`), so every corpus appears
in train, val and test by design. No split method can separate corpus identity
from the label when each corpus maps to exactly one class.

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

## 2. ~~Delete the placeholder dataloader~~ — DONE 2026-09-08

`scripts/dataloader/dataset_loader.py` defines `default_transforms` as
`Resize + ToTensor` with **no** `Normalize`, and is commented "(Placeholder)".
Training does not use it — `train_full.py` goes through
`preprocessing.py`, which normalises — but inference in `frontend/inference.py`
does normalise. Anyone who wires the placeholder up gets a silent train/inference
mismatch. It is dead code; remove it.

---

## 3. ~~Reconcile the reported accuracy~~ — RESOLVED 2026-09-08

The premise was wrong. There was no discrepancy to reconcile; the two numbers
were never measuring the same thing and neither traces to the other:

- **README's 82.73%** is run 01's genuine **test** accuracy. All nine cells of its
  per-class table reproduce from `results/01/y_pred.npy` to 4 d.p.
- **`sweep_w200_100_200.md`'s 82.73%** is that run's **`best_val_acc`**. Its test
  accuracy is 82.86%, correctly reported in the card. The coincidence is numerical.
- **89.40%** is run 19's `best_val_acc`; its test accuracy is 89.72%.
- The "single-stage vs end-to-end cascade" speculation was wrong on both sides.
  **No cascade result exists anywhere in the repository**, and one cannot currently
  be produced — `evaluate.py:461` raises `TypeError` on every SRM checkpoint.

What *is* still true and more serious: `best_val_acc` is a max over epochs on the
selection split and is used as a headline throughout. Every such number should be
relabelled or replaced with a test figure.

---

## 4. ~~No CI~~ — DONE 2026-09-08

`tests/` (33 tests, stdlib `unittest`, no dataset required, ~7s) and
`.github/workflows/tests.yml`. The workflow also import-checks every module and
runs `--help` on all 36 CLI scripts.

Every bug found in the 2026-09-08 audit now has a regression test. The suite
earned itself immediately: writing the FFT test surfaced a bug the audit had
missed — `torch.fft.fftshift` was called without `dim`, so it shifted the batch
axis too and sample *i* received sample *(i + N/2)*'s spectrum.

Run it with:

    venv-linux/bin/python -m unittest discover -s tests -t .
