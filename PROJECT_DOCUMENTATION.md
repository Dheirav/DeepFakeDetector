# Deepfake Detection Project Documentation

## Project Overview
This project implements a multi-level deepfake detection pipeline using deep learning. The workflow covers dataset collection, cleaning, preprocessing, loading, model training, evaluation, and explainability.

---

## Directory Structure

- data/
  - ai_edited/
  - ai_generated/
  - real/
- models/
- results/
- scripts/
  - data/
    - clean_dataset.py
    - split_data.py
    - dataset_stats.py
  - preprocessing/
    - preprocessing.py
  - dataloader/
    - dataset.py
    - dataset_loader.py
  - training/
    - train_baseline.py
    - train_full.py
  - evaluation/
    - evaluate.py
    - evaluation_matrices.py
    - plot_confusion.py
    - plot_confusion_matrix.py
  - explainability/
    - grad_cam.py

---

## Module Descriptions

### 1. Data Collection & Organization
- **Folders:** `data/real/`, `data/ai_generated/`, `data/ai_edited/`
- **Scripts:** None (manual or external download)
- **Notes:**
  - Each folder contains 300–500 images per class.
  - Sources: COCO, ImageNet, GANs, diffusion models, FaceForensics++.

### 2. Data Cleaning
- **Script:** `scripts/data/clean_dataset.py`
- **Function:** Removes corrupted images from each class folder.

### 3. Dataset Statistics
- **Script:** `scripts/data/dataset_stats.py`
- **Function:** Counts and reports the number of images per class.

### 4. Data Splitting
- **Script:** `scripts/data/split_data.py`
- **Function:** Splits each class into training and validation sets.

### 5. Preprocessing
- **Script:** `scripts/preprocessing/preprocessing.py`
- **Function:**
  - Converts images to RGB
  - Resizes to 224x224
  - Normalizes pixel values
  - Applies augmentations: horizontal flip, rotation, brightness/contrast

### 6. Dataset Loader
- **Scripts:**
  - `scripts/dataloader/dataset.py`: Custom PyTorch `Dataset` for loading images and labels (real=0, ai_generated=1, ai_edited=2)
  - `scripts/dataloader/dataset_loader.py`: Builds `DataLoader`, handles train/val split, batch loading, shuffling, and prints dataset statistics

### 7. Model Training
- **Scripts:**
  - `scripts/training/train_baseline.py`: Sets up and prepares ResNet18 for 3-class classification
  - `scripts/training/train_full.py`: Trains the model, tracks loss, saves weights

### 8. Evaluation
- **Scripts:**
  - `scripts/evaluation/evaluate.py`: Computes accuracy, precision, recall, F1, confusion matrix
  - `scripts/evaluation/evaluation_matrices.py`: Additional metrics and reporting
  - `scripts/evaluation/plot_confusion.py`, `plot_confusion_matrix.py`: Plots and saves confusion matrix visualizations

### 9. Explainability
- **Script:** `scripts/explainability/grad_cam.py`
- **Function:** Generates Grad-CAM heatmaps for model explainability

---

## Label Mapping
- Real: 0
- AI Generated: 1
- AI Edited: 2

---

## Usage Example
1. Clean and preprocess the dataset using scripts in `scripts/data/` and `scripts/preprocessing/`.
2. Load data and print statistics using `scripts/dataloader/`.
3. Train the model using `scripts/training/`.
4. Evaluate and visualize results using `scripts/evaluation/`.
5. Generate explainability outputs using `scripts/explainability/`.

---

## Contributors & Roles
- Data Collection: Person 1
- Data Cleaning/Preprocessing: Person 2
- Dataset Loader: Person 3
- Model Training: Person 4
- Evaluation/Explainability: Person 5

---

## Progress Statement
"We have completed dataset preparation, preprocessing, dataset loading, baseline model training, and initial evaluation. The next phase focuses on robustness testing, model optimization, and UI integration."

---

## Notes
- All scripts are modular and organized by function.
- Update import paths if moving scripts or using as a package.
- See each script for detailed function docstrings and usage.
