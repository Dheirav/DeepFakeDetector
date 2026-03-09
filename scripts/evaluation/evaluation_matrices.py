import os
import re
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def evaluate_model(y_true, y_pred):
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    cm = confusion_matrix(y_true, y_pred)
    return cm


def _results_dir_from_model_path(model_path: str) -> str:
    """Mirror the same logic as evaluate.py's _derive_save_dir."""
    abs_path = os.path.abspath(model_path)
    match = re.search(r"(run_\d{8}_\d{6})", abs_path)
    if match:
        return os.path.join(_PROJECT_ROOT, "results", match.group(1))
    parent = os.path.basename(os.path.dirname(abs_path))
    models_root = os.path.basename(os.path.abspath(os.path.join(_PROJECT_ROOT, "models")))
    if parent and parent != models_root:
        return os.path.join(_PROJECT_ROOT, "results", parent)
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    return os.path.join(_PROJECT_ROOT, "results", stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print metrics from saved y_true/y_pred produced by evaluate.py"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the .pth checkpoint — used to locate the matching results folder")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Explicit results directory (overrides auto-detection from --model_path)")
    args = parser.parse_args()

    results_dir = args.results_dir or _results_dir_from_model_path(args.model_path)

    y_true_path = os.path.join(results_dir, "y_true.npy")
    y_pred_path = os.path.join(results_dir, "y_pred.npy")

    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        print(f"ERROR: y_true.npy / y_pred.npy not found in {results_dir}")
        print("Run evaluate.py first to generate them.")
        raise SystemExit(1)

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    print(f"Loaded predictions from: {results_dir}")
    print(f"Samples: {len(y_true)}\n")

    cm = evaluate_model(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
