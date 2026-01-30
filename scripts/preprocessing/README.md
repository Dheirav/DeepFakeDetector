# Preprocessing Scripts

This directory contains scripts for image preprocessing and augmentation:

- `preprocessing.py`: Defines Albumentations pipelines for training and validation, including resizing, normalization, and augmentations (flip, rotation, brightness/contrast, compression).

**Usage:**
- Import the transforms in your dataset or training scripts:
  ```python
  from preprocessing.preprocessing import train_transform, val_transform
  ```
- These transforms are used automatically in the training pipeline.
