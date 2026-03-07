# Multi-Level Deepfake Detection Project

A robust, research-grade pipeline for multi-class deepfake detection using deep learning. This project provides end-to-end tools for dataset construction, model training, evaluation, and explainability.

---

## 🎯 Project Overview
- **Goal:** Detect and classify images as **Real**, **AI Generated**, or **AI Edited**
- **Approach:** Modular pipeline with dataset building, preprocessing, training, evaluation, and explainability

### Key Features

#### 🗄️ Dataset Builder
- Production-grade pipeline across 20 source collections (77,865 images, 0.52% max class imbalance)
- Perceptual-hash deduplication to remove near-duplicates across sources
- Quality filtering by resolution, blur score, and format
- Cluster-based train/val/test splitting — prevents similar images leaking across splits
- Fully deterministic and reproducible (fixed seeds, locked configs)
- Audit reports with per-source statistics and compliance checks
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
├── DATASET.md                    # Dataset design specification
├── requirements.txt              # Python dependencies
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
See [DATASET.md](DATASET.md) for the full breakdown (77,865 images, 0.52% max class imbalance) and
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

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real | 0.7653 | 0.7523 | 0.7588 |
| AI Generated | 0.9100 | 0.9320 | 0.9209 |
| AI Edited | 0.8032 | 0.7975 | 0.8004 |
| **Overall accuracy** | | | **82.73%** |

Evaluated on the held-out test set (23,341 images, balanced across classes).
The main confusion is Real ↔ AI Edited — 69% of all errors fall on that boundary.

---

## 📊 Dataset Builder Pipeline

The `dataset_builder/` module was used to construct the dataset from 20 source collections.
**The build is complete** — exports are in `dataset_builder/train/`, `val/`, `test/`.

### Built Dataset Stats
| Class | Count | Sources |
|---|---|---|
| Real | 26,000 | FFHQ, COCO, Open Images, COCO Test, Places365 |
| AI Generated | 26,000 | Synthbuster, SD 1.x, FLUX.1, StyleGAN, MJ/DALL·E + 4 top-up batches |
| AI Edited | 25,865 | DEFACTO, DEFACTO Inpainting, OpenForensics, FaceForensics++, CASIA, IMD2020 |
| **Total** | **77,865** | 20 artifact sources, 0.52% max imbalance |

### Pipeline Capabilities
- ✅ **Automated sampling** with configurable quotas per source
- ✅ **Deduplication** using perceptual hashing (pHash) to remove near-duplicates
- ✅ **Quality filtering** based on resolution, blur, and metadata
- ✅ **Cluster-based splitting** prevents similar images from leaking across train/test
- ✅ **Deterministic and reproducible** with fixed random seeds
- ✅ **Audit reports** with comprehensive statistics and compliance checks

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

- [DATASET.md](DATASET.md) — Dataset design specification and sampling strategy
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
- **ImageNet:** [https://www.image-net.org/](https://www.image-net.org/)
- **FaceForensics++:** [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)

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

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Data Preparation](#data-preparation)
4. [Preprocessing](#preprocessing)
5. [Dataset Loading](#dataset-loading)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Explainability](#explainability)
9. [Experiment Tracking & Reproducibility](#experiment-tracking--reproducibility)
10. [Example Workflow](#example-workflow)
11. [Best Practices](#best-practices)
12. [Contributors & Roles](#contributors--roles)
13. [References](#references)

---

## Project Overview
- **Goal:** Detect and classify images as Real, AI Generated, or AI Edited.
- **Approach:** End-to-end pipeline with data cleaning, augmentation, PyTorch dataset, ResNet18 baseline, advanced training, evaluation, and explainability.
- **Research-Grade:** Modular, reproducible, and supports experiment tracking.

---

## Directory Structure
```
project-root/
│
├── data/
│   ├── real/
│   ├── ai_generated/
│   └── ai_edited/
│
├── models/                # Saved model checkpoints
├── results/               # Plots, logs, TensorBoard
├── scripts/
│   ├── data/              # Cleaning, splitting, stats
│   ├── preprocessing/     # Augmentations, normalization
│   ├── dataloader/        # Dataset, DataLoader
│   ├── training/          # Baseline & advanced training
│   ├── evaluation/        # Metrics, confusion matrix
│   └── explainability/    # Grad-CAM, heatmaps
│
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
```

---

## Data Preparation
- **Folders:**
  - `dataset_builder/train/`, `dataset_builder/val/`, `dataset_builder/test/`
- **Scripts:**
  - `scripts/data/clean_dataset.py`: Removes corrupted images.
  - `scripts/data/split_data.py`: Splits into train/val sets.
  - `scripts/data/dataset_stats.py`: Reports image counts per class.
- **Best Practices:**
  - Use diverse sources (COCO, ImageNet, GANs, FaceForensics++).
  - Document sources and quality in a dataset report.

---

## Preprocessing
- **Script:** `scripts/preprocessing/preprocessing.py`
- **Transforms:**
  - Resize to 224x224
  - Convert to RGB
  - Normalize pixel values
  - Augmentations: horizontal flip, rotation, brightness/contrast, compression
- **Library:** Albumentations
- **Usage:**
  - Import `train_transform` and `val_transform` in dataset or training scripts.

---

## Dataset Loading
- **Scripts:**
  - `scripts/dataloader/dataset.py`: Custom PyTorch `Dataset` with label mapping (real=0, ai_generated=1, ai_edited=2)
  - `scripts/dataloader/dataset_loader.py`: Train/val split, DataLoader creation, stats
- **Features:**
  - Batch loading, shuffling, reproducible splits
  - Dataset statistics reporting

---

## Model Training
- **Scripts:**
  - `scripts/training/train_baseline.py`: Minimal, research-grade baseline (ResNet18, validation, best model saving, CLI args, reproducibility)
  - `scripts/training/train_full.py`: Advanced (config-driven, TensorBoard, checkpoints, plots, learning rate scheduling, experiment tracking)
- **Features:**
  - Device selection (CPU/GPU)
  - Hyperparameter tuning (CLI/config)
  - Early stopping/checkpoints (in advanced script)
  - Logging: loss, accuracy, validation metrics
  - Reproducibility: random seed setting
- **Outputs:**
  - Best model: `models/best_resnet18.pth`
  - Checkpoints: `models/resnet18_epoch{N}.pth`
  - Plots: `results/loss_curve.png`, `results/accuracy_curve.png`
  - TensorBoard logs: `results/tensorboard/`

---

## Evaluation
- **Scripts:**
  - `scripts/evaluation/evaluate.py`: Accuracy, precision, recall, F1, confusion matrix
  - `scripts/evaluation/evaluation_matrices.py`: Additional metrics
  - `scripts/evaluation/plot_confusion.py`, `plot_confusion_matrix.py`: Visualization
- **Usage:**
  - Run after training to assess model performance
  - Save and analyze misclassified images for error analysis

---

## Explainability
- **Script:** `scripts/explainability/grad_cam.py`
- **Function:**
  - Generates Grad-CAM heatmaps for model interpretability
  - Visualizes model attention on input images
- **Usage:**
  - Run after training to generate heatmaps for selected images

---

## Experiment Tracking & Reproducibility
- **TensorBoard:** Integrated in advanced training for live metrics and comparison
- **Config Files:** YAML config for all experiment settings
- **Random Seeds:** Set for torch, numpy, random, cudnn
- **Best Practices:**
  - Log all hyperparameters and environment details
  - Use version control for code and configs

---

## Example Workflow
1. Clean and preprocess the dataset:
   ```bash
   python scripts/data/clean_dataset.py
   python scripts/data/split_data.py
   python scripts/data/dataset_stats.py
   ```
2. Train a model:
   ```bash
   python scripts/training/train_baseline.py --data_dir dataset_builder --epochs 5
   # or advanced
   python scripts/training/train_full.py --config scripts/training/train_config.yaml
   ```
3. Evaluate:
   ```bash
   python scripts/evaluation/evaluate.py --model_path models/best_resnet18.pth
   ```
4. Visualize explainability:
   ```bash
   python scripts/explainability/grad_cam.py --model_path models/best_resnet18.pth --image_path dataset_builder/test/real/example.jpg
   ```
5. Monitor with TensorBoard:
   ```bash
   tensorboard --logdir results/tensorboard/
   ```

---

## Best Practices
- Use config files for reproducible experiments
- Track all runs with TensorBoard or MLflow
- Save and document all model checkpoints and results
- Analyze misclassifications and feature embeddings
- Keep code modular and well-documented

---

## Hardware-Specific Training Configurations

To ensure stable training and avoid system crashes or overheating, use the following recommended configurations based on your laptop/PC specs. Adjust `batch_size` and `epochs` in `scripts/training/train_config.yaml` or via CLI as needed.

### 1. **Entry-Level Laptop (Integrated GPU or Low VRAM <2GB, 8GB RAM)**
- `batch_size: 8-16`
- `epochs: 10-15`
- `num_workers: 1`
- `pin_memory: False`
- Use `train_baseline.py` for best stability.

### 2. **Mid-Range Laptop (GTX 1650/3050, 4GB VRAM, 8-16GB RAM)**
- `batch_size: 16-32`
- `epochs: 15-20`
- `num_workers: 2`
- `pin_memory: True`
- Use `train_full.py` with moderate settings.

### 3. **High-End Laptop (RTX 4060/4070, 8GB+ VRAM, 16GB+ RAM)**
- `batch_size: 64`
- `epochs: 30`
- `num_workers: 2-4`
- `pin_memory: True`
- Enable mixed precision for faster training (ask for help if needed).

**Tip:** If you get CUDA out-of-memory errors, reduce `batch_size` and restart training. Monitor system temperature and usage with `nvidia-smi` and system tools.

---

## Monitoring GPU and CPU Usage During Training

To ensure your system is running efficiently and not overheating during training, monitor your hardware usage:

### GPU Monitoring
- **Command:**
  ```bash
  watch -n 1 nvidia-smi
  ```
- Shows GPU utilization, memory usage, temperature, and running processes.
- If GPU memory is nearly full or temperature is high (>80°C), reduce batch size or pause training.

### CPU & RAM Monitoring
- **Command:**
  ```bash
  htop
  ```
- Shows CPU core usage, RAM usage, and running processes in real time.
- Install with `sudo apt install htop` if not present.

**Tip:** Always monitor your system during the first few epochs of a new experiment, especially with new batch sizes or model changes.

---

## Contributors & Roles
- Data Collection: Person 1
- Data Cleaning/Preprocessing: Person 2
- Dataset Loader: Person 3
- Model Training: Person 4
- Evaluation/Explainability: Person 5

---

## References
- [Albumentations](https://albumentations.ai/)
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [COCO Dataset](https://cocodataset.org/)
- [ImageNet](https://www.image-net.org/)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)

---

# Deepfake Detection Project

This repository provides tools, scripts, and pipelines for building, training, and evaluating deepfake detection models.

## Project Structure
- `dataset_builder/`: Production-grade, deterministic dataset builder pipeline ([see detailed docs](dataset_builder/README.md))
- `models/`: Model architectures and training scripts
- `scripts/`: Data processing, evaluation, and utility scripts
- `data/`: Raw and processed data directories
- `results/`: Experiment outputs and results

## Dataset Builder Pipeline
The `dataset_builder` module provides a robust, auditable, and fully automated pipeline for constructing machine learning datasets for deepfake detection. It supports:
- Modular, deterministic, and config-driven stages
- Strong error handling and compliance validation
- Dry-run and strict mode for safe experimentation
- Structured logging and reporting

See [dataset_builder/README.md](dataset_builder/README.md) for full usage, configuration, and artifact details.

## Quick Start
1. Prepare your dataset and config YAML (see `dataset_builder/README.md`).
2. Run the dataset builder:
   ```bash
   cd dataset_builder
   python main.py --config path/to/config.yaml
   ```
3. Train and evaluate models using scripts in `models/` and `scripts/`.

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Documentation
- [Dataset Builder Pipeline](dataset_builder/README.md)
- [Project Documentation](PROJECT_DOCUMENTATION.md)

## License
See LICENSE file.
