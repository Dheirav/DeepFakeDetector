# Evaluation Scripts

This directory contains scripts for model evaluation and visualization:

- `evaluate.py`: Computes accuracy, precision, recall, F1, and confusion matrix.
- `evaluation_matrices.py`: Additional metrics and reporting utilities.
- `plot_confusion.py`, `plot_confusion_matrix.py`: Generate and save confusion matrix plots.

**Usage:**
- Run these scripts after training to evaluate model performance and visualize results.
- Example:
  ```bash
  python evaluate.py --model_path models/best_resnet18.pth
  ```
