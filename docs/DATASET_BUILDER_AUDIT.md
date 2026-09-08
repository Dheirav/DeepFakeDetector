# Dataset builder audit — 2026-09-08

Verified against the 20 per-source artifact sets in
`dataset_builder/output/artifacts/` (77,865 images). Every number below was
reproduced by running code, not read off a report.

---

## 1. Every split is 40/30/30, not 70/15/15

`modules/splitter.py:87,93,99` — the greedy assignment cost is a *relative*
error, `abs(count_after - target) / target`. While a split is under target this
equals `1 - fill_fraction`, so `argmin` picks **the split that is already
proportionally fullest**. A small split keeps winning until its relative
overshoot exceeds the large split's relative deficit — i.e. until it reaches
`2 x target`. With 0.70/0.15/0.15 that pins val and test at 0.30N each.

Measured across all 20 sources:

| | train | val | test |
|---|---|---|---|
| configured | 70.0% | 15.0% | 15.0% |
| **actual** | **40.0%** (31,183) | **30.0%** (23,341) | **30.0%** (23,341) |

Class-wise test counts are 7,795 / 7,792 / 7,754 — an exact match for the
`y_true.npy` bincounts in every `results/` run. **43% of the intended training
data was diverted into val and test.** Every model in this repo was trained on
31k images instead of 54k. No audit flagged it: `audit_dataset.py:130` only warns
when a split falls below 5%.

This is a dozen-line fix and it is the highest-value change in the pipeline.

## 2. Near-duplicate detection has ~0% recall

`deduplicator.py:132-148` and `splitter.py:16-53` are the same algorithm,
duplicated: bucket on the first 12 hex chars of a 16-hex-char (64-bit) pHash,
then union-find at Hamming <= 8 *within a bucket*. A pair is only ever compared
if 48 of 64 bits match exactly, so all differing bits must fall in the trailing
16.

Analytic and Monte-Carlo (200k pairs per distance):

| Hamming d | P(same bucket) | simulated recall |
|---|---|---|
| 1 | 2.50e-01 | 25.0% |
| 2 | 5.95e-02 | 5.96% |
| 4 | 2.86e-03 | 0.27% |
| 8 | 2.91e-06 | 0.0005% |

On real images (400 corpus images x 14 realistic transforms), recall on genuine
near-duplicates (d > 0) is **5.1%**; crop-by-5% recall is **0.0%**. On the
pipeline's own stored pHashes, across flux / stylegan / openimages / coco / ffhq:
**0 of 1,287 genuine near-duplicate pairs were caught.**

The Hamming-8 threshold is decorative. What actually runs is exact-pHash
matching. That is why COCO removed 48 near-dups out of 163,957 (0.03%) and
places365, openimages, flux, stylegan and defacto removed exactly zero.

**Downstream: 42,180 near-duplicate pairs survive into the final splits, 27,712
span splits, and 9,871 sit directly across train<->test.** All 20 audits report
zero leakage, because md5 and sha256 differ.

Fix: replace the single 12-char prefix with LSH banding — 16 bands of 4 bits,
candidate if any band matches, then verify true distance. ~30 lines. Extract it
into one shared `modules/clustering.py` that both callers import; keeping two
copies in sync is what produced this.

## 3. The determinism claim is false

No module uses the global seed. `grep` over `modules/*.py` finds zero uses of
module-level `random.*` or `np.random.*` — every stochastic step builds its own
local RNG from `config['random_seed']`. `set_global_seed` and its `_SEED_SET`
guard (`pipeline.py:58-66`) are decorative.

Filesystem ordering does leak in. `indexer.py:83` uses `os.walk` and nothing ever
sorts. That order propagates into bucket insertion order, cluster ordering,
`cluster_id` assignment, and the list that `rng.shuffle` permutes. Measured on
flux (3,000 rows, seed 42 fixed):

```
same input file, run twice             split differs:    0/3000
SAME IMAGES, rows in different order   split differs: 1960/3000  (65%)
                                       cluster_id differs: 3000/3000
```

**65% of images land in a different split purely from directory iteration
order.** Re-index on another machine, after a filesystem move, or after a `cp -r`
and you get a materially different dataset from identical code, config and seed.

## 4. The sampler does not sample

`sampler.py:48` creates an RNG and never uses it; `selected = []` on line 49 is
dead. `balance_sources` is deterministic top-k by `rank_rows` — quality, then
**resolution, then file size**.

So the pipeline systematically keeps the highest-resolution, largest-file images
from every source. That is precisely the axis that most strongly identifies a
corpus, and it sharpens the fingerprint the model then learns.

## 5. The audit certifies nothing relevant

`audit_dataset.py` reads one source's `export_index.csv` and `stat()`s the files.
The verdict (`:264-278`) is `FAIL` for exactly four conditions: zero rows;
missing > 1%; malformed > 1%; exact-hash leakage > 0. Everything else resolves to
`WARN`, which `strict_mode` ignores entirely — so `strict_mode: true` in all 22
configs buys almost nothing.

It structurally cannot detect any of the known problems:

- **Cross-source leakage** — it reads one source at a time. There is no
  cross-source pass anywhere.
- **Resolution/format fingerprint** — it stores width/height only to test `> 0`
  (`:243`). It never opens an image. No format, quantization-table or resampling
  check exists.
- **Class-source confound** — it computes class x split and source x split, but
  never **class x source**, the one cross-tab that would expose it.

`validate_no_leakage` (`splitter.py:120-130`) is tautological: clusters are
assigned to splits atomically at `:106`, so "no cluster spans two splits" can
never fire. All 20 runs logged `Cluster size P99: 1` — every image was its own
cluster — so the check was vacuous twice over.

The missing cross-tab, computed here: **source -> class purity is 100.0% for all
20 sources.** A lookup table on train metadata, evaluated on test, no pixels read:

```
majority-class baseline   33.40%
file format alone         59.11%
resolution alone          79.88%
format + resolution       87.40%
```

**87.4% three-class accuracy from two header fields**, against a model reporting
89%. All 13,905 TIFF images in the dataset are `ai_edited`; zero TIFF appears
anywhere else.

## 6. ImageNet was 4x Lanczos-upscaled, then removed

`dataset_builder/logs/upscale_20260302_110413.log` is the only upscale log:

```
Input dirs : train/real, val/real, test/real     Scale: 4x     Backend: realesrgan
ERROR: Real-ESRGAN requested but not installed. ... Falling back to Lanczos.
Total: 4000  Succeeded: 4000  Backend: lanczos  Format: PNG  Runtime: 30.39s
```

`realesrgan` and `basicsr` are not installed in `venv-linux`; 7.6 ms/image is
consistent with Lanczos and impossible for ESRGAN. So **4,000 ImageNet `real`
images were 4x PIL-Lanczos upscaled and saved as lossless PNG.** No other source
was touched. 4x Lanczos leaves the top ~3/4 of the frequency spectrum empty — a
CNN separates that from native imagery trivially, and every one of those images
is labelled `real`.

ImageNet does not appear in the 20 surviving artifact sets, so it is not in the
current 77,865. Nothing records when or why it was removed. **Any result computed
before 2026-03-02 should be treated as contaminated by this.**

Three further defects in the patching path: `upscale_images.py:366` splits
`quality_flag` on `","` while `validator.py:131` joins with `"|"`, so compound
flags are never cleared; `:365-370` sets `resolution_ok = True` without
recomputing `quality_score`, falsifying the audit trail; and `split_index.csv`
has no `export_path` column so `Patched 0 rows` — the two manifests permanently
disagree.

## 7. Other confirmed defects

- **`--append` is dead code.** `pipeline.py:259-273` is indented into
  `_merge_artifacts` instead of `run_pipeline`, so the merge never runs; if it
  did, `:262` is an unguarded self-recursive call with four undefined names.
  Live consequence: `output/artifacts/coco_test_tmp_1772822342/` is an orphaned
  append run holding 7,000 images that was never merged.
- **`merge_exports.py` emits zero export rows.** `:37` shares one `seen` set
  between the sampled pass (`:52`) and the export pass (`:63`); sampled rows
  insert every key first, so every export row is skipped as a duplicate. Verified
  on flux + places365: `master_sampled_index.csv` 6,000 rows,
  `master_export_index.csv` **0 rows**, 0 files copied. Also `:78` flattens
  `split/class/` out of the destination path.
- **The validator never drops anything.** `validator.py:130` flags `corrupt`,
  `:141` writes the row anyway. `index.csv` and `validated_index.csv` have
  identical row counts for all 20 sources. The only gate is `min_quality_score`,
  and `low_resolution` costs exactly 0.3 → score exactly 0.70, which passes
  `>= 0.7`. **The `min_width`/`min_height` rules are advisory.**
- **Stale-artifact stage skipping.** `pipeline.py:100-140` guards each stage with
  `if not <artifact>.exists()`. Change any config value and re-run into the same
  `artifacts_dir` and every stage silently reuses the old CSV. Anyone fixing the
  splitter and re-running will get the identical broken splits back.
- **Export root collision.** Every config's `export_root` is
  `.../dataset_builder`, so all 20 runs write into one shared tree while the
  root's `export_index.csv`, `checksums.csv` and `manifest.json` are overwritten
  by whichever run finished last. The per-source copies under
  `output/artifacts/<src>/` are the only complete record.
- Swallowed errors: `deduplicator.py:46-47` returns 999 on any hash exception;
  `:127-128` bare `continue` on pHash failure; `:125` file-descriptor leak;
  `exporter.py:28,166` logs and continues on copy failure.
- Statistics: `exporter.py:116` sorts `quality_score` as a **string**;
  `:113-116` the shuffle is a no-op (the sort key ends in the unique path);
  `audit_dataset.py:216` `'median'` is `sorted_q[n//2]`; `:69,71` use
  `str.startswith` for path containment.

---

## Verdict

The split and cluster layer needs rewriting, not patching. Three findings mean
the pipeline never did the job it reports doing: splits are 40/30/30; near-dup
recall is ~0% with 9,871 near-duplicate pairs across train<->test; and the audit
that certifies it reports PASS on all 20 runs while being structurally incapable
of detecting any of it.

**Worth keeping:** the stage decomposition, the artifact-per-stage CSV design,
`indexer`, `validator` (once it gets a reject path), the ranking/tie-breaker
logic, and the audit's reporting scaffolding. That is a real skeleton, and it is
why this was diagnosable at all.

**Must be rewritten:** `assign_clusters_to_splits` (wrong in kind, not in
tuning); the bucketing in both `deduplicator` and `splitter`, extracted into one
shared module so they cannot drift again; the audit's verdict logic and check
set; `merge_exports.py`.

Sequenced by value per unit of effort:

1. **Splitter cost function** — a dozen lines, and it has corrupted every dataset
   built so far.
2. **LSH banding in a shared clustering module** — ~30 lines, lifts near-dup
   recall from ~0% to near-complete. Do this *before* going global: the O(n^2)
   inner loop is only harmless today because the buckets are all singletons.
3. **Re-encode on export** — replace `exporter.py:26`'s `shutil.copy2` with
   resize-to-`target_resolution` + fixed-quality JPEG + EXIF strip, ~25 lines.
   Removes the format and resolution channels that alone give 87.4%.
4. **Leave-one-source-out splits** — drop the `source_cost` term
   (`splitter.py:96-99`), which is what currently forces every source into all
   three splits, and add a `holdout_sources` key. Under 40 lines, and it is the
   only split that yields a defensible generalisation number.
