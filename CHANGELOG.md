# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-03-07

### Training — performance & correctness (`scripts/training/`)

#### `train_full.py`
- **`cudnn.benchmark=True`, `deterministic=False`** — eliminated ~3,600 `cudaFuncGetAttributes` calls
  per profiler step caused by the old `deterministic=True` setting. On fixed 224×224 inputs cuDNN
  now auto-tunes once and reuses the fastest convolution kernel on every subsequent step.
- **`persistent_workers=True` + `prefetch_factor=2`** — DataLoader workers now survive between
  epochs instead of being respawned each time. Workers pre-fetch 2 batches ahead so the GPU never
  idles waiting for data.
- **Worker count reduced from 4 → 2** — on a laptop, 4 workers compete with the training process
  for CPU cores; 2 workers + prefetching is faster overall and runs cooler.
- **Default `--data_dir` changed** from `"data"` to `"dataset_builder/train"`, `--val_dir` to
  `"dataset_builder/val"` — matches the actual dataset export path so the script works with no
  arguments.

#### `train_baseline.py` — full rewrite
- All performance fixes from `train_full.py` applied (cuDNN benchmark, persistent workers,
  prefetch factor, worker count).
- **Bug fix:** validation split previously used `train_transform` (with random augmentations).
  Now wraps each subset with the correct transform (`val_transform` for val). This was causing
  noisy validation metrics.
- **AMP added** (`torch.cuda.amp.GradScaler` + `autocast`) — runs conv/matmul in float16,
  roughly 1.5–2× faster on laptop tensor cores with half the memory bandwidth pressure.
- **`optimizer.zero_grad(set_to_none=True)`** — frees gradient buffers entirely instead of
  writing zeros, cheaper on memory bandwidth.
- **`non_blocking=True`** on `.to(device)` — overlaps tensor transfers with CPU work.
- **`torch.compile(model)`** (PyTorch ≥ 2.0) — fuses element-wise ops and removes redundant
  kernel launches; runtime-detected so it degrades gracefully on older PyTorch.
- **`ReduceLROnPlateau` scheduler** added — halves LR when val loss stops improving, preventing
  loss oscillation and wasted epochs at too high a learning rate.
- **Early stopping** added (`--early_stop_patience`, default 5) — stops when val acc has not
  improved, saves laptop from running hot past the best checkpoint.
- **`pretrained=True` → `ResNet18_Weights.DEFAULT`** — removes deprecation warning.
- **Default `--data_dir`** changed to `"dataset_builder/train"`.

### Evaluation (`scripts/evaluation/`)

#### `evaluate.py`
- **`--data_dir` now optional** — defaults to `dataset_builder/test`.
- **`classification_report` fixed** — previously called with all 3 class names regardless of
  which classes are present in the data; now computes `present_labels` from `y_true` so
  evaluating a partially-populated split no longer raises a label mismatch error.

#### `plot_confusion_matrix.py`
- **Default paths fixed** — were bare relative strings resolved from `cwd`; now use
  `os.path.abspath` anchored to the script file, matching how `evaluate.py` handles paths.
- **Hardcoded `range(3)` replaced** — axes ticks and cell text now derive `n` from the union of
  labels present in `y_true`/`y_pred` so the plot works with 1 or 2 classes.

### Dataset (`scripts/dataloader/dataset.py`)

- **`DeepfakeDataset.__init__` no longer hard-crashes** on missing class folders — skips the
  folder with a warning and only raises `RuntimeError` if zero samples are found total.
  Needed because some dataset splits only contain a subset of classes.

### Frontend (`frontend/`)

#### `app.py` — full rewrite
- **`@st.cache_resource`** on model loader — model is loaded once per session and reused across
  all widget interactions. Previously the model reloaded on every button click.
- **Full workflow redesigned:**
  1. Sidebar: checkpoint path, GPU toggle, Grad-CAM target class, colormap, opacity.
  2. File uploader → immediate image preview.
  3. **✂️ Crop panel** (optional, sidebar toggle): interactive drag-to-crop using
     `streamlit-cropper` before analysis. Supports Free / 1:1 / 4:3 / 16:9 / 3:4 aspect ratios.
  4. **🔎 Analyse** button → classification + per-class confidence progress bars.
  5. **Grad-CAM tabs:** overlay, side-by-side comparison, raw heatmap — each with a download
     button.
  6. **Expandable all-class comparison** — renders heatmaps for all three classes side-by-side.
- **`use_column_width=True` → `use_container_width=True`** across all `st.image` calls —
  removes Streamlit deprecation warnings.
- **Double `.unsqueeze(0)` bug fixed** — `preprocess_image` already returns `[1,C,H,W]`;
  the old code added a second batch dimension causing a shape crash.

#### `inference.py`
- **`torch.compile` state dict prefix stripped** — compiled checkpoints prefix every key with
  `_orig_mod.`; the loader now detects and strips this automatically so compiled models load
  cleanly.
- **`strict=False` → `strict=True`** — silent weight mismatches now surface as a clear error
  instead of training from a partially-initialized model.
- **`pretrained=True` → `ResNet18_Weights.DEFAULT`** — removes deprecation warning.
- **`ResNet18_Weights` import added.**

#### `config.py`
- Default `MODEL_CHECKPOINT` updated to `models/run_20260307_063053/best_resnet18.pth` — the
  actual trained checkpoint path so the UI works out of the box without manual configuration.

#### `gradcam.py` (no code changes)
- `demo_gradcam.py` fixed: removed erroneous `.unsqueeze(0)` call on a tensor already batched
  by `preprocess_image`.

### Infrastructure

#### `.gitignore`
- Added: `models/`, `results/`, `dataset_builder/logs/`, `dataset_builder/train/`,
  `dataset_builder/val/`, `dataset_builder/test/`, `dataset_builder/data/`,
  `dataset_builder/checksums.csv`, `dataset_builder/export_index.csv`,
  `dataset_builder/manifest.json`, `dataset_builder/output/artifacts/*/`,
  `logs/`, `*.log`, `.hf_cache/`, standard OS/editor files.
- Kept tracked: `dataset_builder/config/` (pipeline configs), top-level artifact index CSVs.

#### `requirements.txt`
- `streamlit-cropper` should be added for the crop feature.

### New dependencies
```
streamlit
streamlit-cropper
```

---

## [v0.1.0] — 2026-02-13 (prior session)

- Initial dataset builder pipeline with 20 sources, ~77,865 images.
- ResNet18 baseline training script.
- Frontend Streamlit app scaffold.
- Grad-CAM implementation.
