import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

# Project root = two levels above this script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_RESULTS_ROOT = os.path.join(_PROJECT_ROOT, "results")


def _latest_run_dir() -> str:
    """
    Return the results sub-directory for the most recent run (highest
    run_YYYYMMDD_HHMMSS folder name), or _RESULTS_ROOT if none exist.
    """
    pattern = re.compile(r"^run_\d{8}_\d{6}$")
    if os.path.isdir(_RESULTS_ROOT):
        runs = sorted(
            [d for d in os.listdir(_RESULTS_ROOT) if pattern.match(d)],
            reverse=True,
        )
        if runs:
            return os.path.join(_RESULTS_ROOT, runs[0])
    return _RESULTS_ROOT

def plot_cm(y_true, y_pred, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Determine which classes are actually present so the plot doesn't crash
    # when only a subset of the three classes exists in the data.
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names  = [CLASS_NAMES[i] for i in present_labels]
    n = len(present_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (counts)",      "d"),
        (axes[1], cm_norm, "Confusion Matrix (normalized)",  ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(present_names, rotation=30, ha="right")
        ax.set_yticklabels(present_names)
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                txt = f"{val:{fmt}}" if fmt == ".2f" else str(val)
                color = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_dir', type=str, default=None,
        help=(
            'Path to a specific run results directory '
            '(e.g. results/run_20260307_063053). '
            'Defaults to the most recent run_* folder found in results/.'
        ),
    )
    parser.add_argument('--y_true_path', type=str, default=None)
    parser.add_argument('--y_pred_path', type=str, default=None)
    parser.add_argument('--save_path',   type=str, default=None)
    args = parser.parse_args()

    # Resolve run directory: explicit flag > latest run folder > results/
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else _latest_run_dir()
    print(f"Using run directory: {run_dir}")

    y_true_path = args.y_true_path or os.path.join(run_dir, "y_true.npy")
    y_pred_path = args.y_pred_path or os.path.join(run_dir, "y_pred.npy")
    save_path   = args.save_path   or os.path.join(run_dir, "confusion_matrix.png")

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    plot_cm(y_true, y_pred, save_path=save_path)

