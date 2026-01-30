# Dataloader Scripts

This directory contains scripts for dataset loading and batching:

- `dataset.py`: Custom PyTorch `Dataset` for loading images and labels from the organized data folders.
- `dataset_loader.py`: Provides functions for creating train/validation DataLoaders and printing dataset statistics.

**Usage:**
- Import and use the `DeepfakeDataset` and loader functions in your training scripts.
- Example:
  ```python
  from dataloader.dataset import DeepfakeDataset
  from dataloader.dataset_loader import create_dataloaders
  ```
