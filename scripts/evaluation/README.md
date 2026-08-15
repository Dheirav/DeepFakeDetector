Cascade evaluation
===============

This folder contains helper scripts to run a two-pass cascade evaluation for deepfake detection.

Workflow
--------
- Stage 1: run `evaluate_stage1.py` with a 3-class model (Real, AI Generated, AI Edited). Outputs saved to `results/cascade_stage1/`:
  - `stage1_probs.npy`, `stage1_preds.npy`, `y_true.npy`, `image_paths.npy`

- Stage 2 subset selection: run `select_stage2_subset.py --threshold <float>` to select samples where
  `abs(P(real) - P(ai_edited)) <= threshold`. Writes `stage2_indices.npy` to `results/cascade_stage2/`.

- Stage 2: run `evaluate_stage2.py --stage2_model_path <path> --indices_file results/cascade_stage2/stage2_indices.npy`.
  Outputs `stage2_preds.npy` and `stage2_probs.npy` in `results/cascade_stage2/`.

- Merge: run `merge_cascade_results.py` to combine stage1 and stage2 predictions. Final outputs saved to `results/cascade_final/y_pred_final.npy` and a classification report/ confusion matrix will be printed.

Notes
-----
- These scripts reuse the existing `load_model`, `DeepfakeDataset` and `val_transform` from `evaluate.py` and do not modify any existing code.
- Default result directories: `results/cascade_stage1/`, `results/cascade_stage2/`, `results/cascade_final/`.
- Tune `--batch_size` and `--num_workers` on each script for performance.
# Stage-2 details
# ---------------
# - Purpose: Stage-2 is a binary refiner that only distinguishes `Real` vs `AI Edited` on a small subset
#   of samples where the Stage-1 (3-class) model is uncertain between those two classes. It MUST NOT be
#   used for, or overwrite, `AI Generated` predictions from Stage-1.
# - Workflow summary:
#   1. Run Stage-1 to produce `stage1_probs.npy`, `stage1_preds.npy`, `y_true.npy`, and `image_paths.npy` in `results/cascade_stage1/`.
#   2. Run `select_stage2_subset.py --threshold <t>` to create `results/cascade_stage2/stage2_indices.npy`.
#      - The selector excludes any sample where Stage-1 predicted `AI Generated` so Stage-2 never sees or
#        overwrites `AI Generated` predictions.
#   3. Run Stage-2 with `evaluate_stage2.py --stage2_model_path <model> --indices_file results/cascade_stage2/stage2_indices.npy`.
#   4. Merge using `merge_cascade_results.py` which preserves `AI Generated` labels and only applies Stage-2
#      refinements to samples that were forwarded to Stage-2.
#
# Example commands
# ----------------
# Run Stage-1:
# ```bash
# ./venv-linux/bin/python3 scripts/evaluation/evaluate_stage1.py \
#   --model_path <stage1.pth> --data_dir dataset_builder/test --save_dir results/cascade_stage1 \
#   --batch_size 8 --num_workers 2
# ```
# Select Stage-2 subset (threshold = 0.2):
# ```bash
# ./venv-linux/bin/python3 scripts/evaluation/select_stage2_subset.py --stage1_probs results/cascade_stage1/stage1_probs.npy --threshold 0.2 --save_dir results/cascade_stage2
# ```
# Run Stage-2 on selected subset:
# ```bash
# ./venv-linux/bin/python3 scripts/evaluation/evaluate_stage2.py --stage2_model_path <stage2.pth> --indices_file results/cascade_stage2/stage2_indices.npy --save_dir results/cascade_stage2
# ```
# Merge final predictions:
# ```bash
# ./venv-linux/bin/python3 scripts/evaluation/merge_cascade_results.py --stage1_dir results/cascade_stage1 --stage2_dir results/cascade_stage2 --save_dir results/cascade_final
# ```
#
# Stage-2 analysis
# ----------------
# - To inspect Stage-2 performance on the subset it was run on, use the `stage2_confusion.py` utility which
#   computes a binary classification report and confusion matrix (Real vs AI Edited) for only the forwarded
#   samples. This utility will ignore any indices that accidentally contain `AI Generated` ground-truth labels.
#
# Example:
# ```bash
# ./venv-linux/bin/python3 scripts/evaluation/stage2_confusion.py \
#   --stage1_dir results/cascade_stage1 --stage2_dir results/cascade_stage2 --save_dir results/cascade_stage2 --plot
# ```
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
