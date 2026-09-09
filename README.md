# Multi-Level Deepfake Detection

A three-class image classifier — **Real / AI-Generated / AI-Edited** — built end to
end: a 20-source dataset builder, a training harness, evaluation, Grad-CAM
explainability, and a Streamlit UI.

It reaches ~89% on its own held-out test set. **That number is not a
generalisation estimate**, and most of this README is about how I established
that and what it actually measures.

I started this because generative models have made it genuinely hard to know
whether you can believe something you're looking at, and because a friend of mine
ran into real trouble from generative AI being used maliciously. I wanted
something that could tell the difference.

---

## ⚠️ Status

**The original models in this repository do not detect AI-generated content. They
recognise which source dataset an image came from.** That is established below
with measurements rather than asserted.

A rebuild is under way on a corpus verified to carry no such shortcut, and its
first results are in [The rebuild](#the-rebuild). The honest number is **65.8%**,
against the original 89%, and the gap between those two figures is the whole
point of the project.

- Full limitations, for both models: **[`LIMITATIONS.md`](LIMITATIONS.md)**
- Audit trail: [`docs/REVIEW_2026-09-08.md`](docs/REVIEW_2026-09-08.md) ·
  [`docs/DATASET_BUILDER_AUDIT.md`](docs/DATASET_BUILDER_AUDIT.md)
- Dataset survey and confound measurements: [`docs/DATASETS_2026.md`](docs/DATASETS_2026.md)
- Plan: [`docs/SALVAGE_PLAN.md`](docs/SALVAGE_PLAN.md)

---

## What this project found

All twenty source corpora map to exactly one class each — every "real" image comes
from a photo dataset, every "AI-generated" one from a generator dump, every
"edited" one from a forgery benchmark. Source→class purity is **100.0%**. That
makes corpus identity a perfect stand-in for the label, and it is far easier to
learn than a manipulation trace.

I did not set out to find this. I had finished training and was writing the
documentation, and I wanted to try the model on some of my own photographs before
calling it done. It got them wrong — confidently. Everything below is what I did
to work out why.

### 1. The file header beats the network

A lookup table on `(format, width, height)`, fit on train and scored on test,
**reading no pixels at all**:

| Feature | 3-class test accuracy |
|---|---|
| majority-class baseline | 33.40% |
| container format alone | 59.11% |
| resolution alone | 79.88% |
| **format + resolution** | **87.40%** |
| the trained model | ~89% |

All 13,905 TIFF files in the dataset are `ai_edited` — 100% precision, 53.8% of
that class, 17.9% of the dataset classified perfectly without decoding an image.

### 2. A routine JPEG re-save inverts the prediction

`P(correct)` over resolution × JPEG quality, on 200 images that are **all
AI-generated**. Content never changes; only the encoding does.

```
  size    no-jpeg     q95     q85     q75     q60
  1024      0.990   0.990   0.985   0.960   0.925
   768      0.995   0.985   0.965   0.950   0.880
   512      0.995   0.990   0.980   0.945   0.675
   384      0.995   0.985   0.935   0.650   0.395
   320      0.995   0.970   0.680   0.380   0.160
   256      0.990   0.675   0.080   0.005   0.000
```

Downscale to 256px and save at q80 — what happens to any image crossing the web —
and **280 of 300 AI-generated images are classified "real" at 87% confidence.**
Accuracy goes 0.993 → 0.027. Resolution alone is harmless (the no-JPEG column);
it is compression artefact scale relative to image size that carries the signal,
because that is the encoding signature separating the `real` corpora from the
generated ones.

### 3. Validation accuracy is inversely correlated with robustness

Across nine runs, Pearson **r = −0.956** between `best_val_acc` and accuracy under
degradation:

| run | augmentation | val acc | accuracy at 256px / q60 |
|---|---|---|---|
| 19 *(the one that shipped)* | light | 0.894 | **0.000** |
| 10 | standard | 0.866 | 0.650 |
| 17 *(rejected)* | strong | 0.843 | **0.955** |

Strong augmentation destroys the corpus fingerprint, so the validation set — which
shares that fingerprint — penalises it. **A robust model was already trained and
was discarded for scoring five points lower on a metric measuring the wrong
thing.**

### 4. The test set is contaminated

Deduplication and cluster-splitting run once per source, so nothing is ever
compared across corpora:

| measure | count | % of test |
|---|---|---|
| test images byte-identical to a train image | 1,012 | **4.34%** |
| test images pHash-identical to a train image | 1,656 | **7.09%** |

Separately, **743 COCO photos appear both as a `real` example and as the base
image of a DEFACTO `ai_edited` example** — DEFACTO filenames embed the COCO ID —
with 616 of those pairs crossing a split boundary.

### 5. No forensic component has a measurable effect

Paired McNemar on the test set (n = 23,341) against plain RGB ConvNeXt-Small:
**+SRM p = 0.725**, +SRM+FFT p = 0.081, "GeM" p = 1.000, "CBAM" p = 0.324. The
measured noise floor, from two runs with identical configs, is **0.15 pp** — larger
than every claimed effect. Two of those four checkpoints also turned out not to
contain the component their folder name claims.

### 6. The `ai_edited` score is too high to be genuine

The model reports 0.86 F1 on `ai_edited`. DEFACTO images average **1.7% tampered
pixels**, and published methods score **0.8–6.9%** on tampered-image detection at
224px while reaching 83–94% on fully-synthetic images. A healthy number here, at
this resolution, is itself evidence of a shortcut.

### What I take from it

The thing that found this was a few minutes of testing on photographs my pipeline
had never touched, and I did it last instead of first. Everything before that —
the ablations, the sweeps, the model cards — was measuring the same flaw more and
more precisely.

The fix is not a better network. Every corpus mapped to exactly one class, so a
shortcut was available and no architecture was going to refuse it. It has to be
fixed in the data: **matched pairs**, where each manipulated image's own original
is its `real` counterpart, so both sides share a camera, a codec and a resolution
and only the manipulation differs. About 11,400 such pairs are recoverable from
filenames already in this dataset — DEFACTO, CASIA and IMD2020 all encode their
source image's ID. That is what the rebuild is built around; see
[`docs/SALVAGE_PLAN.md`](docs/SALVAGE_PLAN.md).

---

## The rebuild

The findings above say what went wrong. This section says what happened when the
same task was attempted on a corpus where the shortcut does not exist.

<!-- ✍️  YOUR WORDS: a sentence or two on deciding to rebuild rather than patch. -->

### A dataset where the shortcut is measurably absent

The corpus is a slice of **OpenSDI** (`nebula/OpenSDI_train`), which supplies all
three classes from a single real-image pool with segmentation masks for the
edited class. Its binary label decomposes into the three needed here through the
`key` field, where `entire/` marks a fully synthetic image and `partial/` a
locally edited one.

OpenSDI was not taken on trust. Measured as shipped, on shards holding the same
photographs edited and unedited, the geometry is clean but **the JPEG
quantisation table separates the classes at 93.8%**, because editing requires
re-saving and the editor used 16 distinct quality settings against the originals'
3. That signal is "was this re-encoded by the editor", not a manipulation trace,
and it is intrinsic to any locally-manipulated dataset rather than a defect of
this one.

Re-encoding every image identically removes it. Measured on the converted output,
3600 images across three classes:

| feature | accuracy | over baseline |
|---|---|---|
| majority-class baseline | 49.6% | |
| container format | 49.6% | +0.0 |
| resolution, megapixels, aspect ratio | 49.6% | +0.0 |
| file size | 49.6% | +0.0 |
| JPEG quantisation table | 49.6% | +0.0 |

Every metadata feature sits exactly at the baseline, carrying **zero** information
about the class. The same probe scores 87.4% on this project's original corpus
and 93.0% on OpenSDI as shipped.

### The honest number

A linear probe on frozen DINOv2 features, because fine-tuning moves a network
toward whatever separates the training classes, and when a shortcut is available
that is what it moves toward. Kumar et al. (ICLR 2022) measured fine-tuning at
roughly +2 points in-domain and −7 out-of-domain against a probe on the same
features. The probe is also the control: a fine-tuned network that cannot beat it
learned nothing the frozen features did not already contain.

| | original model | linear probe |
|---|---|---|
| dataset | confounded | verified clean |
| accuracy | 0.890 | **0.658** |
| `real` F1 | 0.854 | 0.688 |
| `ai_generated` F1 | 0.976 | 0.775 |
| `ai_edited` F1 | **0.861** | **0.465** |

The `ai_edited` row is the one that matters. 0.86 is not achievable at this
resolution: manipulations of this kind average 1.7% tampered pixels, and
published image-level methods score 0.8 to 6.9 percent on tampered detection at
224px. The original 0.86 was therefore evidence of a shortcut rather than of
detection, while 0.465 on a corpus carrying no metadata signal is a believable
number for the same task. 65.8% overall also sits inside the 65 to 80 percent
band the literature reports for held-out-corpus evaluation.

### It still breaks under compression, differently

The same degradation grid, on 600 held-out images:

| condition | probe | original model |
|---|---|---|
| clean | 0.667 | 0.993 |
| 320px / q85 | 0.615 | 0.680 |
| 256px / q80 | **0.533** | **0.027** |

Against a 0.520 majority-class baseline, the probe at 256px q80 is **1.3 points
above guessing "real" for everything**. It does not survive compression.

What differs is the failure mode. The original model *inverted*, reaching 0.027
by calling 280 of 300 AI-generated images real at 87% mean confidence, which is
confidently and systematically wrong. The probe *decays toward chance*. Per-class
recall shows the mechanism: real recall falls from 0.731 to 0.397 while both fake
classes improve, so under compression the probe increasingly calls everything
fake. That is the mirror image of the original model's bias toward calling
everything real, which is the same weakness with the opposite sign.

### Transfer tracks architectural distance

OpenSDI trains on sd15 and ships a test set spanning five generators, making
leave-one-generator-out the protocol the dataset was built for. Per-class recall
on `ai_generated`:

| generator | recall | relationship to training generator |
|---|---|---|
| sd2 | 0.718 | same family |
| sd3 | 0.667 | same family |
| sdxl | 0.463 | same family, different scale |
| flux | 0.470 | different architecture |

On flux the probe calls 193 of 400 synthetic images real against 188 correct,
which is close to a coin flip on a generator it has not seen. This is the
field-wide result rather than a peculiarity of this project: detectors learn
generator-specific artefacts, not a general notion of "synthetic".

### Encoder capacity is not the bottleneck

| encoder | dim | accuracy |
|---|---|---|
| DINOv2 ViT-S/14 | 384 | 0.6583 |
| DINOv2 ViT-B/14 | 768 | 0.6296 |
| DINOv2 ViT-L/14 | 1024 | 0.6593 |

Paired McNemar: S against L gives **p = 1.000**, S against B p = 0.132, B against
L p = 0.119. All three are statistically indistinguishable, so ten times the
parameters bought a rounding error.

The per-class split says where the constraint actually is. Going from ViT-S to
ViT-L, `ai_generated` F1 improves from 0.775 to 0.802 while `ai_edited` moves from
0.465 to 0.453, which is no improvement. Whether a whole image is synthetic is a
global property that a richer representation can exploit, whereas a local edit
covering a small fraction of the frame is destroyed by the resize to 224px before
the encoder ever sees it. No encoder recovers information thrown away upstream of
it.

**So the lever for `ai_edited` is resolution and a mask head, not a bigger
backbone**, and the masks are already downloaded.

<!-- ✍️  YOUR WORDS: what you take from the rebuild. -->

### Reproducing this

```bash
# build the dataset (downloads shards, normalises, deletes them after)
venv-linux/bin/python dataset_builder/tools/convert_opensdi.py

# confirm the shortcut is absent before training on it
venv-linux/bin/python scripts/data/metadata_confound.py data_sources/opensdi

# train and evaluate
venv-linux/bin/python scripts/training/train_linear_probe.py
venv-linux/bin/python scripts/evaluation/degradation_test.py
venv-linux/bin/python scripts/evaluation/heldout_generator_test.py
```

---

## 🎯 Project Overview
- **Goal:** Detect and classify images as **Real**, **AI Generated**, or **AI Edited**
- **Approach:** Modular pipeline with dataset building, preprocessing, training, evaluation, and explainability

### Key Features

#### 🗄️ Dataset Builder
- Production-grade pipeline across 20 source collections (77,865 images, 0.52% max class imbalance)
- Perceptual-hash deduplication **within each source** (see the correction note below)
- Quality filtering by resolution, blur score, and format
- Cluster-based train/val/test splitting **within each source**
- Fixed seeds and per-source config files
- Audit reports with per-source statistics

> **Correction (2026-09-08).** Earlier versions of this list claimed dedup and
> leakage prevention *across* sources, full determinism, and 70/15/15 splits.
> An audit disproved all four. The pipeline runs once per source, so nothing is
> ever compared across corpora; near-duplicate recall is ~0%; splits are
> actually 40/30/30; and directory iteration order changes 65% of split
> assignments at a fixed seed. **4.34% of the test set is byte-identical to a
> training image.** See [`docs/DATASET_BUILDER_AUDIT.md`](docs/DATASET_BUILDER_AUDIT.md)
> and [`LIMITATIONS.md`](LIMITATIONS.md). These are being fixed; the claims were
> removed rather than left standing.
- `DeepfakeDataset` gracefully skips missing class folders with a warning instead of crashing

#### ⚡ Training — `train_full.py` & `train_baseline.py`
- **cuDNN auto-tuning** (`benchmark=True`) — eliminates ~3,600 redundant `cudaFuncGetAttributes` calls per step, delivering noticeably faster steps on fixed 224×224 inputs
- **AMP** (Automatic Mixed Precision, `float16` autocast + `GradScaler`) — ~1.5–2× faster conv/matmul on laptop tensor cores with half the memory bandwidth pressure
- **`torch.compile`** support (PyTorch ≥ 2.0) — fuses element-wise ops and removes redundant kernel launches; detected and enabled at runtime automatically
- **`persistent_workers=True` + `prefetch_factor=2`** — DataLoader workers survive between epochs (no respawn overhead) and pre-fetch 2 batches ahead so the GPU never idles waiting for data
- **Worker count 4 → 2** — prevents CPU thermal throttling on laptops where workers compete with the training process
- **`zero_grad(set_to_none=True)`** — frees gradient memory entirely instead of writing zeros
- **`non_blocking=True`** tensor transfers — CPU-to-GPU overlap with compute
- **`ReduceLROnPlateau` scheduler** — halves LR when val loss plateaus, stopping loss oscillation
- **Early stopping** (`--early_stop_patience`, default 5) — halts training when val acc stagnates
- **Bug fix:** validation split previously used `train_transform` (augmented); now correctly uses `val_transform`
- **Bug fix:** default `--data_dir` corrected to `dataset_builder/train` (actual export path)
- `pretrained=True` → `ResNet18_Weights.DEFAULT` (removes deprecation warning)
- PyTorch Profiler integration in epoch 1 to surface per-op CPU/CUDA bottlenecks

#### 📊 Evaluation
- `evaluate.py` — `--data_dir` now optional (defaults to `dataset_builder/test`); `classification_report` only reports classes actually present in the data (no crash on partial splits)
- `plot_confusion_matrix.py` — default paths fixed to be relative to the script file; dynamic n-class axis rendering so the plot works with 1, 2, or 3 classes

#### 🖥️ Streamlit UI — `frontend/app.py`
- **`@st.cache_resource` model loader** — model is loaded once per session and reused; no reload on every widget interaction
- **✂️ Interactive crop panel** — drag-to-crop before analysis using `streamlit-cropper`; supports Free / 1:1 / 4:3 / 16:9 / 3:4 aspect ratios
- **Grad-CAM tabbed panel** with three views: overlay, side-by-side comparison (original | raw heatmap | overlay), and raw grayscale activation map; each tab has a download button
- **All-class Grad-CAM expander** — renders heatmaps for all three classes side-by-side in one click
- **Per-class confidence progress bars** — visual breakdown of all three class probabilities
- `use_container_width` replaces deprecated `use_column_width` throughout
- **Bug fix:** double `.unsqueeze(0)` removed — `preprocess_image` already returns `[1,C,H,W]`

#### 🔍 Grad-CAM (`frontend/gradcam.py`, `frontend/inference.py`)
- `torch.compile` checkpoint compatibility — automatically strips the `_orig_mod.` key prefix that compiled models add, so compiled checkpoints load cleanly
- `strict=True` loading — weight mismatches now surface as a clear error instead of silently training from a partially-initialized model
- Auto-detects last Conv2d layer; supports manual `target_layer` override
- Hook cleanup (`cam.cleanup()`) prevents memory leaks across multiple calls
- `overlay_heatmap` supports OpenCV (fast, ~2–5 ms) or matplotlib (quality, ~10–20 ms) backends with auto-detection

---

## 📁 Directory Structure
```
deepfake-project/
│
├── README.md                     # This file
├── LIMITATIONS.md                # What the numbers do and do not measure — read this
├── BACKLOG.md                    # Ordered work list
├── requirements.txt              # Python dependencies (⚠️ unpinned)
│
├── docs/
│   ├── REVIEW_2026-09-08.md          # Full code + results audit
│   ├── DATASET_BUILDER_AUDIT.md      # Dataset pipeline audit
│   ├── GENERALISATION_LITERATURE.md  # What the field reports for cross-domain transfer
│   ├── SALVAGE_PLAN.md               # Phased plan to correct the project
│   └── DATASET.md                    # Dataset design specification
│
├── dataset_builder/              # Production dataset pipeline — also contains the built dataset
│   ├── main.py                   # Pipeline orchestrator
│   ├── pipeline.py               # Pipeline logic
│   ├── train/                    # Built dataset — train split (~31,146 images)
│   ├── val/                      # Built dataset — val split (~23,360 images)
│   ├── test/                     # Built dataset — test split (~23,359 images)
│   ├── config/                   # Per-source pipeline configs (20 sources)
│   ├── scripts/                  # Download scripts for each source
│   ├── modules/                  # Pipeline modules (indexer, validator, deduplicator, …)
│   └── output/                   # Pipeline artifacts and manifests
│
├── scripts/                      # Training and evaluation scripts
│   ├── preprocessing/
│   │   ├── preprocessing.py
│   │   └── visualize_augmentations.py
│   ├── dataloader/
│   │   ├── dataset.py
│   │   └── dataset_loader.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_full.py
│   │   └── train_config.yaml
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── evaluation_matrices.py
│   │   └── plot_confusion_matrix.py
│   └── data/
│       ├── clean_dataset.py
│       ├── split_data.py
│       └── dataset_stats.py
│
├── frontend/                     # Streamlit UI
│   ├── app.py                    # Main UI application
│   ├── config.py                 # UI configuration
│   ├── inference.py              # Inference utilities
│   └── gradcam.py                # Grad-CAM implementation
│
├── models/                       # Saved model checkpoints
├── logs/                         # Training logs
└── results/                      # Evaluation outputs and plots
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset

The dataset is fully constructed in `dataset_builder/train/`, `dataset_builder/val/`, and `dataset_builder/test/`.
See [DATASET.md](docs/DATASET.md) for the full breakdown (77,865 images, 0.52% max class imbalance) and
[dataset_builder/README.md](dataset_builder/README.md) for pipeline documentation.

> **Note:** Dataset images are excluded from git (see `.gitignore`). Model checkpoints and results are also local-only. Re-train using the commands below or download a checkpoint separately.

### 3. Train a Model

**Baseline training** (fast, no extras):
```bash
python scripts/training/train_baseline.py
# defaults: --data_dir dataset_builder/train  --epochs 5  --batch_size 32
```

**Full training** (AMP, TensorBoard, Grad-CAM profiling, early stopping):
```bash
python scripts/training/train_full.py
# defaults: --data_dir dataset_builder/train  --val_dir dataset_builder/val
# or use a config file:
python scripts/training/train_full.py --config scripts/training/train_config.yaml
```

Checkpoints are saved to `models/<run_id>/`, plots and metrics to `results/<run_id>/`.

### 4. Evaluate

```bash
python scripts/evaluation/evaluate.py \
  --model_path models/<run_id>/best_resnet18.pth
# --data_dir defaults to dataset_builder/test
```

Then plot the confusion matrix:
```bash
python scripts/evaluation/plot_confusion_matrix.py
# reads results/y_true.npy + results/y_pred.npy written by evaluate.py
```

### 5. Launch the Streamlit UI

```bash
streamlit run frontend/app.py
```

Workflow:
1. Upload any JPG / PNG / WEBP image
2. Click **🔎 Analyse** — classification + confidence bars appear
3. Explore the Grad-CAM panel (overlay / side-by-side / raw heatmap tabs)
4. Optionally enable **✂️ Crop** in the sidebar first to focus on a region

### 6. Grad-CAM from the command line

```bash
python demo_gradcam.py \
  --model models/<run_id>/best_resnet18.pth \
  --image path/to/image.jpg \
  --output_dir results/gradcam
```

---

## 📈 Baseline Results (ResNet18, 15 epochs, March 2026)

*This is the earliest run, kept for the record. See [What this project found](#what-this-project-found) for what these numbers measure.*

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real | 0.7653 | 0.7523 | 0.7588 |
| AI Generated | 0.9100 | 0.9320 | 0.9209 |
| AI Edited | 0.8032 | 0.7975 | 0.8004 |
| **Overall accuracy** | | | **82.73%** |

Evaluated on the held-out test set (23,341 images, balanced across classes).
The main confusion is Real ↔ AI Edited — 69% of all errors fall on that boundary.

These nine figures reproduce exactly from `results/01/y_true.npy` and `y_pred.npy`.
Three caveats, all measured:

- This is run 01, the **weakest** model in the repository. Later runs reach ~89.7%
  test accuracy; see `results/ablation_study.md`.
- **The number is inflated by leakage.** 4.34% of the test set is byte-identical
  to a training image and 7.09% is a perceptual-hash duplicate, because
  deduplication ran per-source and never across sources.
- **It does not measure generalisation.** Every source corpus maps to exactly one
  class, so corpus identity predicts the label perfectly. A lookup table on file
  format and resolution alone — reading no pixels — reaches 87.4% on this task.
  See [`LIMITATIONS.md`](LIMITATIONS.md).

---

## 📊 Dataset Builder Pipeline

The `dataset_builder/` module was used to construct the dataset from 20 source collections.

> **Note.** The exported image directories (`dataset_builder/train|val|test/`) are
> **empty in this repository** — the images are gitignored and the working copies
> were removed. What survives is the complete per-source metadata in
> `dataset_builder/output/artifacts/*/export_index.csv`: 77,865 rows with sha256,
> source path and split assignment for every image, so the exact dataset
> membership is reconstructible if the sources are re-downloaded.

### Built Dataset Stats
| Class | Count | Sources |
|---|---|---|
| Real | 26,000 | FFHQ, COCO, Open Images, COCO Test, Places365 |
| AI Generated | 26,000 | Synthbuster, SD 1.x, FLUX.1, StyleGAN, MJ/DALL·E + 4 top-up batches |
| AI Edited | 25,865 | DEFACTO, DEFACTO Inpainting, OpenForensics, FaceForensics++, CASIA, IMD2020 |
| **Total** | **77,865** | 20 artifact sources, 0.52% max imbalance |

### Pipeline Capabilities — audited 2026-09-08

| Claim | Status |
|---|---|
| Automated quotas per source | ✅ works |
| Quality filtering by resolution / blur | ⚠️ computed, but nothing is ever rejected — the validator flags and writes the row anyway |
| pHash deduplication | ❌ **~0% recall on real near-duplicates** (0 of 1,287 measured pairs); the 12-hex-char bucket makes it exact-match only |
| Cluster splitting prevents leakage | ❌ **per-source only** — 4.34% of test is byte-identical to train |
| Deterministic and reproducible | ❌ directory iteration order changes **65%** of split assignments at fixed seed |
| 70/15/15 split ratio | ❌ actually **40/30/30** — a relative-vs-absolute error in the greedy cost function |
| Audit reports | ⚠️ generated, but the verdict can only FAIL on four conditions; all 20 runs report PASS |

Full evidence and line numbers: [`docs/DATASET_BUILDER_AUDIT.md`](docs/DATASET_BUILDER_AUDIT.md).

### Pipeline Stages
1. **Indexing**: Scan all source directories and create a master index
2. **Validation**: Verify image integrity, resolution, and format
3. **Deduplication**: Remove duplicates using pHash similarity
4. **Quality Filtering**: Filter by resolution, blur score, and other metrics
5. **Sampling**: Select exact quotas per source and balance classes
6. **Cluster-Based Split**: Create train/val/test splits using similarity clustering
7. **Export**: Copy selected files to final dataset structure
8. **Audit**: Generate compliance reports and statistics

### Configuration
Each source has its own config in `dataset_builder/config/`. Example structure:

```yaml
random_seed: 42
artifacts_dir: output/artifacts
export_root: .   # exports directly into dataset_builder/

image_rules:
  min_width: 256
  min_height: 256

class_targets:
  real: 5000   # per-source quota

split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15
```

### Re-running the Pipeline (if needed)
```bash
cd dataset_builder
python main.py --config config/<source>_config.yaml [--dry-run] [--log-level INFO]
```

**Dry-run mode** simulates the pipeline without writing files.

---

## 🔧 Data Preparation (Legacy Scripts)

If you already have a small, organized dataset, you can use the legacy scripts in `scripts/data/`:

- **Clean corrupted images:**
  ```bash
  python scripts/data/clean_dataset.py --data_dir data
  ```

- **Split into train/val:**
  ```bash
  python scripts/data/split_data.py --data_dir data --test_size 0.2
  ```

- **View dataset statistics:**
  ```bash
  python scripts/data/dataset_stats.py --data_dir data
  ```

**Note:** For large-scale dataset construction from multiple sources, use the **dataset_builder pipeline** instead.

---

## 🎨 Preprocessing & Augmentation

The `scripts/preprocessing/preprocessing.py` module provides:
- Resize to 224×224
- RGB conversion
- Normalization (ImageNet stats)
- Augmentations: horizontal flip, rotation, brightness/contrast adjustment, JPEG compression simulation

**Usage:**
```python
from scripts.preprocessing.preprocessing import train_transform, val_transform

# For training
transformed = train_transform(image=image)["image"]

# For validation/testing
transformed = val_transform(image=image)["image"]
```

**Visualize augmentations:**
```bash
python scripts/preprocessing/visualize_augmentations.py --image_path data/real/sample.jpg
```

---

## 🧠 Model Training

### Label Mapping
- **Real:** 0
- **AI Generated:** 1
- **AI Edited:** 2

### Training Scripts

#### Baseline Training (`train_baseline.py`)
Fast, self-contained training run with all performance optimisations:
```bash
python scripts/training/train_baseline.py
# defaults: --data_dir dataset_builder/train  --epochs 5  --batch_size 32
```

**Features:**
- ResNet18 pretrained backbone (`ResNet18_Weights.DEFAULT`)
- AMP (float16 autocast + GradScaler)
- `torch.compile` (PyTorch ≥ 2.0, auto-detected)
- `ReduceLROnPlateau` LR scheduler
- Early stopping (`--early_stop_patience`)
- cuDNN auto-tuning, persistent DataLoader workers, prefetch
- Correct val transform (no augmentations on validation)
- Best model checkpointing, per-epoch console summary

#### Advanced Training (`train_full.py`)
Full-featured training with experiment tracking:
```bash
python scripts/training/train_full.py
# or with config:
python scripts/training/train_full.py --config scripts/training/train_config.yaml
```

**Features:**
- All baseline optimisations (AMP, cuDNN benchmark, compile, persistent workers)
- YAML config support
- TensorBoard logging (loss, accuracy, LR, GPU/CPU resource metrics)
- PyTorch Profiler on epoch 1 — surfaces CPU/CUDA bottlenecks automatically
- `ReduceLROnPlateau` scheduler + early stopping
- Per-epoch checkpoint saving + best model tracking
- F1 macro, per-class F1 logged every epoch
- Training/validation loss and accuracy curves saved as PNGs

**Monitor with TensorBoard:**
```bash
tensorboard --logdir results/tensorboard/
```

### Hardware-Specific Configurations

Adjust `batch_size` and `num_workers` based on your hardware:

| Hardware | Batch Size | Epochs | Workers | VRAM |
|----------|-----------|--------|---------|------|
| **Entry-level** (Integrated GPU, 8GB RAM) | 8-16 | 10-15 | 1 | <2GB |
| **Mid-range** (GTX 1650/3050, 16GB RAM) | 16-32 | 15-20 | 2 | 4GB |
| **High-end** (RTX 4060/4070, 16GB+ RAM) | 64 | 30+ | 2-4 | 8GB+ |

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Monitor CPU/RAM:**
```bash
htop
```

---

## 📈 Evaluation

### Compute Metrics
```bash
python scripts/evaluation/evaluate.py \
    --model_path models/best_resnet18.pth \
    --data_dir dataset_builder
```

**Metrics computed:**
- Accuracy (overall and per-class)
- Precision, Recall, F1-score
- Confusion matrix
- Classification report

### Visualize Confusion Matrix
```bash
python scripts/evaluation/plot_confusion_matrix.py \
    --y_true_path results/y_true.npy \
    --y_pred_path results/y_pred.npy
```

---

## 🔍 Explainability (Grad-CAM)

Generate Grad-CAM heatmaps to understand model decisions:

**Via Streamlit UI:**
```bash
streamlit run frontend/app.py
```
Upload an image and click "Analyze" to see prediction + heatmap overlay.

**Programmatic usage:**
```python
from frontend.gradcam import GradCAM, overlay_heatmap
from frontend.inference import load_model, preprocess_image
from PIL import Image

model = load_model("models/best_resnet18.pth")
cam = GradCAM(model)

image = Image.open("sample.jpg")
tensor = preprocess_image(image)
heatmap = cam(tensor, class_idx=1)
overlay = overlay_heatmap(image, heatmap, alpha=0.5)
overlay.save("heatmap_output.png")
```

---

## 🎨 Frontend (Streamlit UI)

Interactive web interface for inference and visualization:

```bash
streamlit run frontend/app.py
```

**Features:**
- Image upload (JPG, PNG, WEBP) with size validation
- **✂️ Interactive crop panel** — drag-to-crop before analysis (Free / 1:1 / 4:3 / 16:9 / 3:4 aspect ratios); toggle via sidebar
- Real-time inference with a large prediction badge (🟢 Real / 🔴 AI Generated / 🟠 AI Edited)
- Per-class confidence progress bars for all three classes
- **Grad-CAM tabbed panel:**
  - 🌡️ Overlay tab — heatmap blended onto the image + download button
  - 📊 Side-by-side comparison tab — original | raw heatmap | overlay in one image
  - 🗺️ Raw heatmap tab — grayscale activation map
- **All-class Grad-CAM expander** — renders heatmaps for all three classes side-by-side
- Sidebar controls: model checkpoint path, GPU toggle, target class, colormap (jet/viridis/hot/plasma), opacity slider
- Model cached with `@st.cache_resource` — loads once per session

**Configuration:**
Edit `frontend/config.py` to set default model path. The default points to the trained checkpoint: `models/run_20260307_063053/best_resnet18.pth`.

---

## 🔬 Experiment Tracking & Reproducibility

### Best Practices
- ✅ Use **config files** for all experiments (YAML)
- ✅ Set **random seeds** for reproducibility:
  ```python
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  ```
- ✅ Track experiments with **TensorBoard** or **MLflow**
- ✅ Version datasets and models
- ✅ Document hyperparameters in logs

### Logging
All scripts output logs to:
- Console (stdout)
- `logs/` directory
- TensorBoard (for training)
- `dataset_builder/output/pipeline.log` (for dataset construction)

---

## 🏗️ Extending the Project

### Adding New Models
1. Implement model in `scripts/training/`
2. Update `train_baseline.py` or `train_full.py`
3. Ensure label mapping: Real=0, AI Generated=1, AI Edited=2

### Adding New Datasets
1. Download source data into `data_sources/<class>/<SourceName>/`
2. Create a new config in `dataset_builder/config/<source>_config.yaml`
3. Run: `cd dataset_builder && python main.py --config config/<source>_config.yaml`
4. Verify output in `dataset_builder/train/`, `val/`, `test/`

**Important:** Always use a fresh `artifacts_dir` subdirectory per source to avoid double-counting during re-runs.

### Custom Augmentations
Edit `scripts/preprocessing/preprocessing.py` to add Albumentations transforms.

---

## 📚 Documentation

- [DATASET.md](docs/DATASET.md) — Dataset design specification and sampling strategy
- [dataset_builder/README.md](dataset_builder/README.md) — Complete pipeline documentation
- [scripts/data/README.md](scripts/data/README.md) — Legacy data utilities
- [scripts/dataloader/README.md](scripts/dataloader/README.md) — PyTorch dataset and dataloader
- [scripts/training/README.md](scripts/training/README.md) — Training documentation
- [scripts/evaluation/README.md](scripts/evaluation/README.md) — Evaluation metrics
- [scripts/preprocessing/README.md](scripts/preprocessing/README.md) — Preprocessing and augmentation

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `--batch_size` (try 16 from 32)
- Use `torch.cuda.empty_cache()` between runs
- Monitor with `nvidia-smi`

**2. Import Errors (ModuleNotFoundError)**
- Always run from the project root (`deepfake-project/`)
- Check that `frontend/__init__.py` exists
- Verify `sys.path` includes project root in scripts

**3. `FileNotFoundError` on dataset paths**
- The correct paths are `dataset_builder/train`, `dataset_builder/val`, `dataset_builder/test` — not `data/`
- All training/evaluation scripts now default to these paths automatically

**4. Port 8501 already in use (Streamlit)**
```bash
kill $(lsof -ti:8501)
```

**5. Model checkpoint fails to load**
- If you saved a model with `torch.compile` enabled, the state dict keys are prefixed with `_orig_mod.` — `inference.py` strips this automatically
- Ensure you pass the full path including the run subfolder: `models/run_<id>/best_resnet18.pth`

**6. Slow Training / CPU thermal throttling**
- `num_workers` is set to 2 by default for laptop use — don't increase above the number of physical cores
- `cudnn.benchmark=True` is set — first batch of epoch 1 is slower while cuDNN tunes; subsequent steps are fast
- `torch.compile` adds a one-time compilation cost on the first forward pass (~30–60 s) — normal behaviour

**7. Low Accuracy**
- Real ↔ AI Edited confusion accounts for 69% of errors in the baseline — use weighted loss (`CrossEntropyLoss(weight=...)`) to focus on that boundary
- Try a larger backbone (ResNet50, EfficientNet-B3) for +2–4% F1 on hard classes
- Add label smoothing: `CrossEntropyLoss(label_smoothing=0.1)`

---

## 🔗 References

- **Albumentations:** [https://albumentations.ai/](https://albumentations.ai/)
- **PyTorch:** [https://pytorch.org/](https://pytorch.org/)
- **Streamlit:** [https://streamlit.io/](https://streamlit.io/)
- **TensorBoard:** [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
- **Grad-CAM Paper:** [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
- **COCO Dataset:** [https://cocodataset.org/](https://cocodataset.org/)
- **FaceForensics++:** [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)

### On generalisation and shortcut learning

The findings above are an instance of a documented, field-wide problem. A reviewed
bibliography with reported figures is in
[`docs/GENERALISATION_LITERATURE.md`](docs/GENERALISATION_LITERATURE.md); the most
directly relevant:

- **GenImage** ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571)) — same-generator
  98.5–99.9% vs cross-generator ~60–70%.
- **Deepfake-Eval-2024** ([arXiv:2503.02857](https://arxiv.org/abs/2503.02857)) —
  detectors moving to in-the-wild data: UnivFD 0.94 → 0.56 AUC.
- **B-Free** ([arXiv:2412.17671](https://arxiv.org/abs/2412.17671)) — Figure 2 shows the
  same detector flipping its prediction depending on which corpus supplied the reals.
  This project's confound, published.
- **SAFE** ([arXiv:2408.06741](https://arxiv.org/abs/2408.06741), KDD 2025) — replace
  down-sampling with cropping; a direct fix for the resize artefact measured above.
- **Grommelt et al.** (ECCV 2024 Workshops) — debiasing JPEG quality and image size alone
  moves cross-generator accuracy 71.68% → 82.74%.

---

## 👥 Contributors

This project was developed collaboratively:
- **Data Collection & Organization:** Dataset sourcing and curation
- **Data Cleaning & Preprocessing:** Image validation and augmentation pipeline
- **Dataset Builder:** Production-grade pipeline architecture
- **Model Training:** Baseline and advanced training implementations
- **Evaluation & Explainability:** Metrics, visualization, and Grad-CAM

---

## 📝 License

See LICENSE file for details.
