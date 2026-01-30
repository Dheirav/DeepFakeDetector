# Multi-Level Deepfake Detection Project — Detailed Documentation

This repository implements a robust, research-grade pipeline for multi-class deepfake detection using deep learning. The project is modular, reproducible, and designed for extensibility and research.

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
  - `data/real/`, `data/ai_generated/`, `data/ai_edited/`
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
   python scripts/training/train_baseline.py --data_dir data --epochs 5
   # or advanced
   python scripts/training/train_full.py --config scripts/training/train_config.yaml
   ```
3. Evaluate:
   ```bash
   python scripts/evaluation/evaluate.py --model_path models/best_resnet18.pth
   ```
4. Visualize explainability:
   ```bash
   python scripts/explainability/grad_cam.py --model_path models/best_resnet18.pth --image_path data/real/example.jpg
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
