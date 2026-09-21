# Handover

The live state of the project. Read this first; the README is the account and
may lag behind by a run or two, this file should not. Last updated 2026-09-22,
at commit `4ac63208`, with uncommitted doc corrections from 2026-09-22.

## What this is, in three sentences

A three-class detector (real / AI-generated / AI-edited) whose first version
scored 89% by recognising which dataset a file came from, not what was in the
image. It was rebuilt on a corpus where that shortcut measurably does not exist,
and the rebuilt model scores 0.8040 in-distribution and 0.723 mean recall on
generators it never saw. Everything in between is measured, including the
things that did not work, and the write-up is the deliverable.

## The state of things

**The model in the frontend** is `results/mask_head_clip448_balanced/best_model.pth`:
frozen CLIP ViT-B/16 at 448px with interpolated positional embeddings, a
four-block mask decoder, and a small classifier. Trained on 4,500 images per
class from OpenSDI's sd15 shards, re-encoded to 512px JPEG q90.

| number | value | where measured |
|---|---|---|
| 3-class accuracy, balanced test split (4,050) | 0.8040 | `training_summary.json` |
| per-class F1 real / gen / edit | 0.727 / 0.961 / 0.722 | same |
| mask IoU on edited (1,350) | 0.272 | same |
| mean held-out-generator recall, 9 rows | 0.723 | `heldout_generators.json` |
| abstain line (top prob) | 0.90: answers 56%, right 94% of those | `decision_rule.json` |
| compression: clean / 256px q80 / 224px q50 | 0.832 / 0.680 / 0.605 | `degradation.json` |

**Two other trained models exist and are not the headline:**

- `results/mask_head_clip448_ft4/`: last 4 encoder blocks trained. 0.9015
  in-distribution, held-out mean 0.657, flux generated recall 0.240. Learned the
  training generator's fingerprint. Its 116 MB checkpoint is not in git; a copy
  with sha256 is at `/mnt/c/d_drive/projects/deepfake_models/mask_head_clip448_ft4/`.
- `results/mask_head_clip448_smoothaug/`: smooth-patch augmentation on real
  training images. Worse on every measurement (0.7911, held-out 0.689). Kept as
  a negative result.

**The original models** (ConvNeXt, ResNet, in `models/`) still load and run, so
the frontend can compare them on the same upload. They are what `LIMITATIONS.md`
sections 1 to 8 are about. Do not quote their numbers as detection results.

## What is open

1. **A false-positive rate on real photos from outside any dataset.** The one
   photo tested (a conference room) came back `ai_edited` at 0.82 with the mask
   on a glossy tablecloth; under the abstain rule it is a "cannot tell". One
   photo is an anecdote. Thirty or more unedited photos in `my_photos/real/`
   and `venv-linux/bin/python scripts/inference/predict_mask_head.py my_photos/`
   gives the number. Nothing else on this list matters as much.
2. **The two first-person passages in the README** ("The rebuild" opening
   paragraph; "What I take from this") were drafted from the author's own
   account and should be read and edited by the author.
3. **The abstain rule does not detect degraded input.** Coverage stays at 50 to
   55 percent across the compression grid while accuracy-when-answered falls
   from 0.958 to 0.705. A quality check in front of the model would be the fix;
   none exists.
4. **Localisation.** IoU 0.272 against the paper's 0.671, and the DINOv2 head
   that classifies worse localises better (0.385). Not pursued; one paragraph in
   LIMITATIONS.

## What is deliberately not going to be done

Each of these was considered and either measured or reasoned out. Do not
restart them without a new reason.

- More encoder capacity (S vs L p = 1.000; ViT-L/14 at 2+ hours per epoch under
  thermal throttling, see `results/mask_head_clipL448/PARTIAL.md`).
- More epochs, more data at the same source, 672px, threshold tuning, class
  rebalancing: six levers, all flat, table in the README.
- Milder augmentation variants of the smooth-patch idea: the frozen features
  have no better cue to move to, so the mechanism is the problem, not the dose.
- Training on more generators or adding a second real-photo source: both would
  help and both reintroduce a one-source-one-class confound unless the new
  source appears in every class. Only worth it with the photo set from item 1
  to measure against.
- Uploading checkpoints to Hugging Face (account creation failed; the D-drive
  backup is the mirror).

## How to run things

```bash
# UI (verdict, cannot-tell, predicted mask, token Grad-CAM)
venv-linux/bin/python -m streamlit run frontend/app.py

# your own photos
venv-linux/bin/python scripts/inference/predict_mask_head.py my_photos/

# tests: 46 in the files, 44 run by default; RUN_SLOW=1 adds the two that load the model on CPU
venv-linux/bin/python -m unittest discover -s tests -t .

# retrain the headline model (about 90 minutes on the RTX 4060 laptop GPU)
venv-linux/bin/python scripts/training/train_mask_head.py \
    --encoder clip:ViT_B_16 --size 448 --data_dir data_sources/opensdi_large \
    --mask_dir data_sources/opensdi_large_masks --max-per-class 4500 \
    --epochs 10 --batch 8 --class-weights 1.5 1.0 1.0 --out results/<name>

# then the three evaluations, in this order
venv-linux/bin/python scripts/evaluation/mask_head_generalisation.py --checkpoint results/<name>/best_model.pth
venv-linux/bin/python scripts/evaluation/abstain_sweep.py --run results/<name> --choose 0.9
venv-linux/bin/python scripts/evaluation/mask_head_degradation.py --checkpoint results/<name>/best_model.pth
```

Progress readers: `tools/train-progress.sh <log> --watch` and
`tools/heldout-progress.sh <log> --watch`. Both derive the ETA from the measured
rate and say so when there is no rate yet.

## Things that will bite you

- **The GPU jobs must be started detached** (`setsid nohup ... &`). Started from
  a Claude Code shell, CUDA initialisation trips the harness's low-memory
  heuristic on this 8 GB box and the job is killed at zero progress. The kernel
  never OOMs; it is the harness.
- **Never compare two models on raw held-out recall unless they trained on the
  same class mix.** The first CLIP vs DINOv2 comparison was off by 13 to 29
  points from exactly this, and prior correction narrowed it without fixing it.
  `--max-per-class 4500` is how the balanced runs were made.
- **The test split is deterministic** (sorted listing, stratified 30 percent,
  seed 42) and every mask-head run shares it, which is what makes McNemar valid
  across runs. Changing `--max-per-class` or the data directory changes the
  split.
- **Every training image was re-encoded to 512px JPEG q90.** Inputs at inference
  should go through the same step (the CLI and UI do it by default). Raw phone
  JPEGs are a distribution shift and give different answers, sometimes better.
- **`pkill -f` on a pattern that appears in your own command line kills your own
  shell.** Use `pgrep` and check the process name first.
- **`*.log` is gitignored.** The logs that documents cite were force-added; do
  the same for any new one you cite.
- **No pip installs into `venv-linux/`.** `statsmodels` is not there; McNemar is
  done by hand with `scipy.stats.chi2` (see the commit messages for the formula).

## Where the data is

`data_sources/` (3.8 GB, not in git): `opensdi_large/` is the training corpus
(12,175 real / 4,500 / 4,500, of which the balanced runs use the first 4,500 of
each in sorted order), `opensdi_large_masks/` its masks, `heldout/{flux,sd15,
sd2,sd3,sdxl}` the cross-generator test sets at 400 per class where present,
`opensdi/` the original 3,600-image slice the probe used. Rebuilt with
`dataset_builder/tools/convert_opensdi.py`, which deletes shards after
conversion. The metadata probe (`scripts/data/metadata_confound.py`) must sit at
the majority baseline on any directory before it is trained on.

## Where to read

In order: `README.md` (the account), `LIMITATIONS.md` section 9 (what the
current model cannot do), `docs/BENCHMARK_COMPARISON.md` (against the paper),
`results/README.md` (every run and which numbers are corrupt). The original
system's manual is `docs/LEGACY_PIPELINE.md`; the audit that started the
rebuild is `docs/REVIEW_2026-09-08.md`; the plan and its outcome is
`docs/SALVAGE_PLAN.md`. Documents in `docs/` dated 2026-09-08 or earlier that
quote accuracies above 80% are describing the confounded models.
