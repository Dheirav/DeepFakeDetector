# Salvage plan — 2026-09-08

Written after a full audit (see `REVIEW_2026-09-08.md`,
`DATASET_BUILDER_AUDIT.md`, `GENERALISATION_LITERATURE.md`).

The premise: the project's failure is **fully diagnosed and measured**. That is
an asset, not a wound. Most theses that fail this way never find out why. The
work below turns the diagnosis into the contribution and then builds the
corrected version on top of it.

Every phase produces something standalone. If you run out of time at the end of
any phase, you still have a defensible deliverable.

---

## Status — 2026-09-08

**Phases 0, 1 and 2 are complete, and the code half of Phase 3 is done.** What
remains in Phase 3 is data acquisition, which needs network and disk rather than
code.

| Item | Status | Evidence |
|---|---|---|
| Split record under version control | done | 20 CSVs staged, 77,865 rows |
| Grad-CAM colour inversion | **fixed** | both render paths agree: importance 1.0 -> red |
| Grad-CAM target layer | **fixed** | no longer selects CBAM's 1-channel conv; class-0/1 CAM correlation -0.449 |
| One shared model builder | **done** | `scripts/modules/model_builder.py`; 21/21 checkpoints load `strict=True` |
| Config embedded in checkpoints | **done** | round-trip verified across 3 architectures |
| False claims removed from repo | done | see `LIMITATIONS.md`, `results/README.md` |
| README rebuilt around the findings | done | six measurements as its spine |
| Splitter cost function | **fixed** | 70/15/15 -> exactly 70.0/15.0/15.0 |
| Near-duplicate clustering | **fixed** | recall 0% -> 100% at Hamming <= 7 (`modules/clustering.py`) |
| Re-encode on export | **implemented** | `normalise_on_export` config key; format+resolution now carry 0 bits |
| Leave-one-source-out splits | **implemented** | `holdout_sources` config key; verified no held-out source leaks to train/val |
| `merge_exports.py` | **fixed** | master export index 0 -> 6,000 rows |
| Validator reject path | **fixed** | unreadable images dropped; `low_resolution` now scores 0.69, below the 0.7 gate |
| SRM initialisation | **fixed** | residual channels distinct, variance contribution 0.0008 -> 0.10 |
| Matched-pair index | **built** | `dataset_builder/pair_index.csv` -- **16,719 pairable manipulations from 13,110 distinct originals** |
| Re-download sources + masks | **not started** | needs network/disk; 13 of 20 have a URL or script in-repo |
| Retrain under the corrected build | **not started** | blocked on the above |

Smoke tests: 0 import failures and 0 `--help` failures across every module in
`scripts/` and `frontend/`, up from 1 and 2. Code net: **-1,095 / +685** lines.

---

## What you already have

| Asset | State |
|---|---|
| Split membership for all 77,865 images | **Complete** — 20 per-source `export_index.csv`, sha256 + split for every row, 100% populated |
| Source data | Partial — 1.7 GB `ai_generated` on disk; the rest deleted but **re-obtainable** (13 of 20 sources have a URL or script in-repo) |
| Trained checkpoints | 25 files; **21 load cleanly** under a config-driven loader |
| A robust model | **Run 17** — 0.955 where the shipped model scores 0.000 |
| Matched manipulation/original pairs | **~11,400 recoverable from filenames alone**; ~30-45k if sources are re-fetched |
| Segmentation masks | Never downloaded — available upstream for DEFACTO, CASIA, IMD2020, FF++ |
| Dataset builder | Good skeleton; four bugs, all localised |
| Frontend | Works; needs one colour fix and a repointed checkpoint |
| **Measurements of the failure** | **Done — see Phase 1** |

---

## Phase 0 — stop the bleeding (a few hours)

Nothing here needs data or training.

1. **Put the split record under version control.** `.gitignore:48` excludes
   `dataset_builder/output/artifacts/*/`; git tracks 2 files there and 0 under
   `models/`. Those 20 CSVs (31 MB) are the only surviving record of which image
   went where, and `train/val/test/` are empty. Losing this working copy loses
   the test set and every checkpoint. `git add -f` them.
2. **Fix the Grad-CAM colour inversion** — one line at `frontend/gradcam.py:222`.
   Every heatmap figure currently shows the least-important regions as red, and
   `app.py:203` tells the reader the opposite. Regenerate all figures afterwards.
3. **Replace three model builders with one.** `build_model_from_config()` reading
   `training_summary.json`, `strict=True`, config embedded in future checkpoints.
   ~180 lines replacing ~660. Proof-of-concept written and verified: loads 21/21
   with zero missing or unexpected keys.
4. **Repoint `frontend/config.py` at run 17.**

**Deliverable:** the repo becomes reproducible and the figures become truthful.

---

## Phase 1 — the instruments (1-2 days; most of it already exists)

This is the part that turns a failed model into a contribution. **None of it
needs new data or retraining.** Most of it was produced during the audit and
needs only formalising into figures and a written protocol.

| Instrument | Result | Status |
|---|---|---|
| **Metadata-only probe** — predict the class from file header alone, no pixels | **87.4%** three-class (train->test); 96.7% in-sample. Model gets 89% | measured |
| **Degradation curve** — accuracy over resolution x JPEG quality | 0.993 -> **0.027**; full 7x5 grid | measured |
| **Selection inversion** — val accuracy vs robustness across 9 runs | Pearson **r = -0.956** | measured |
| **Leakage quantification** — cross-source train/test contamination | **4.34%** byte-identical, **7.09%** pHash | measured |
| **Replicate noise floor** — two identical configs | **0.15 pp**, McNemar p=0.28 | measured |
| **Ablation under significance testing** | every forensic variant p > 0.07 vs plain RGB | measured |
| **Real-vs-real coherence probe** — can a model separate COCO from FFHQ, both labelled "real"? | not yet run | needs re-downloaded data |

The literature review found that **two of these have no published image-domain
instance**: the metadata-only leakage probe and the real-vs-real coherence probe.
Both are logistic regressions. That is a genuine, small, defensible novelty
claim — not "we built a detector", but "we built the instruments that show why
these detectors don't transfer, and here is what they measure on a corpus built
the standard way."

**Deliverable:** a complete methods-critique chapter with six original
measurements. This alone is a pass-grade thesis contribution, and it is ~90%
done.

---

## Phase 2 — fix the builder (3-5 days)

Four localised changes, sequenced deliberately.

1. **Splitter cost function** (`splitter.py:87,93,99`) — ~12 lines. It currently
   normalises each split's deviation by its own target, so every build is
   40/30/30 instead of 70/15/15. Fix first: it corrupts everything downstream.
2. **LSH banding**, extracted into a shared `modules/clustering.py` imported by
   both `deduplicator` and `splitter` — ~30 lines. Current near-duplicate recall
   is **0 of 1,287** measured pairs. Do this before going global, or the O(n^2)
   inner loop stops being harmless.
3. **Re-encode on export** (`exporter.py:26`) — ~25 lines. One resolution, one
   format, one JPEG quality, EXIF stripped. This is what kills the 87.4%
   metadata shortcut.
4. **Leave-one-source-out split mode** — drop the `source_cost` term
   (`splitter.py:96-99`), which is exactly what forces every source into all
   three splits; add a `holdout_sources` key. ~40 lines.

Also: one global config over all 20 sources instead of 22 separate runs, so
dedup and leakage checks finally see across corpora.

**Deliverable:** a builder whose audit report means something.

---

## Phase 3 — rebuild the data (1-2 weeks, mostly waiting on downloads)

1. **Re-download** the deleted sources. 13 of 20 have a URL or script in-repo;
   DEFACTO, CASIA, IMD2020, FFHQ, COCO and OpenImages need URLs sourced manually.
2. **Download the masks this time.** DEFACTO, CASIA, IMD2020 and FF++ all ship
   pixel-level ground truth. Your FF++ downloader already supports it
   (`TYPE = ['videos','masks','models']`) — you took videos only.
3. **Build matched pairs.** DEFACTO and DEFACTO_Inpainting filenames carry the
   COCO image ID; CASIA carries its own authentic IDs; IMD2020 has one
   manipulation per original. Use each manipulation's *own* original as its
   `real` counterpart. Same camera, codec, resolution — the only difference is
   the edit. **This is the change that kills the confound at the root**, because
   both halves of every pair have identical packaging.
4. **Normalise encoding** across everything that cannot be paired.

**Deliverable:** a dataset where the shortcut cannot exist by construction.

---

## Phase 4 — retrain honestly (3-5 days)

1. **Frozen-feature linear probe first.** CLIP or DINOv2 features + logistic
   regression. Minutes to train, and it doubles as a shortcut control: if the
   probe matches the fine-tuned CNN, the CNN learned nothing the features didn't
   already have.
2. **Crop, don't resize.** SAFE (KDD 2025) is a direct published fix for the
   resize artefact measured in Phase 1. Gragnaniello et al.: *"a 2x downsampling
   has catastrophic effects."*
3. **Fix the SRM initialisation** (`srm.py:155`) before re-running that ablation.
   All three residual channels currently get identical weights contributing
   ~1/1250 of the pre-activation variance — SRM was never really connected, so
   the null result isn't yet a result about SRM.
4. **Evaluate leave-one-source-out** and report the spread, not the mean.

**Expect 65-80%, and say so up front.** In-the-wild detectors sit at 0.55-0.70
AUC; UnivFD drops 0.94 -> 0.56 moving to real-world data. 89% is not on that
scale.

---

## Phase 5 — optional, if time allows

Add a **segmentation head for the `ai_edited` class at >=512px**. The literature
is unambiguous: DEFACTO images average 1.7% tampered pixels, and published
methods score 0.8-6.9% on tampered detection at 224px while scoring 83-94% on
fully-synthetic. **Your 0.86 F1 on `ai_edited` is not achievable by detecting
manipulations at that resolution** — its existence is itself evidence of the
shortcut, arrived at independently of the JPEG experiment.

Report IoU / pixel-F1 for that class, not accuracy.

---

## The thesis this becomes

> We built a three-class detector the standard way, reached 89%, and found it
> collapsed on unseen images. We then built instruments to measure why: a
> metadata-only probe that reaches 87.4% without reading a pixel, a degradation
> curve showing 99.3% -> 2.7% under a routine JPEG re-save, and a model-selection
> analysis showing validation accuracy is *inversely* correlated (r = -0.956)
> with robustness. We show the cause is a corpus-label confound baked into how
> these datasets are assembled, quantify the resulting train/test contamination
> at 4.34%, rebuild the corpus with matched manipulation/original pairs and
> normalised encoding, and report honest leave-one-source-out performance.

That is a better thesis than a working 89%, and it is defensible line by line,
because every number in it was measured rather than hoped for.

**Minimum viable version if time is short:** Phases 0, 1 and 2, plus an honest
limitations chapter. That is a complete, coherent piece of work using data
already on disk, and Phase 1 is nearly finished.
