import os
import sys
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def compute_stage2_confusion(stage1_dir, stage2_dir, save_dir, plot=True):
    # Load necessary files
    s1_preds = np.load(os.path.join(stage1_dir, "stage1_preds.npy"))
    y_true = np.load(os.path.join(stage1_dir, "y_true.npy"))

    indices_path = os.path.join(stage2_dir, "stage2_indices.npy")
    if not os.path.exists(indices_path):
        raise RuntimeError(f"Missing {indices_path}")
    indices = np.load(indices_path)

    s2_preds = np.load(os.path.join(stage2_dir, "stage2_preds.npy"))

    if len(indices) != len(s2_preds):
        # tolerate if stage2 saved fewer preds (truncated); align by min length
        n = min(len(indices), len(s2_preds))
        indices = indices[:n]
        s2_preds = s2_preds[:n]

    # Filter out any indices where ground-truth is AI Generated (1) — Stage-2 does not handle that class
    valid_mask = [y_true[i] != 1 for i in indices]
    if not any(valid_mask):
        print("No valid Real/AI-Edited samples found in Stage-2 indices.")
        return

    indices = indices[np.array(valid_mask)]
    s2_preds = np.array(s2_preds)[np.array(valid_mask)]

    # Map ground-truth to binary: Real(0) -> 0, AI_Edited(2) -> 1
    y_true_binary = np.array([0 if y_true[i] == 0 else 1 for i in indices])

    # s2_preds are expected 0=Real,1=AI Edited
    y_pred_binary = np.array(s2_preds, dtype=int)

    # Classification report and confusion matrix
    report = classification_report(y_true_binary, y_pred_binary, target_names=["Real","AI_Edited"], output_dict=True)
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    os.makedirs(save_dir, exist_ok=True)
    # Save numeric outputs
    np.save(os.path.join(save_dir, "stage2_confusion_matrix.npy"), cm)
    with open(os.path.join(save_dir, "stage2_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Stage-2 binary classification report (Real vs AI_Edited):")
    print(classification_report(y_true_binary, y_pred_binary, target_names=["Real","AI_Edited"]))
    print("Confusion matrix:")
    print(cm)

    # Optional plot
    if plot:
        if not HAS_MPL:
            print("matplotlib not available; skipping plot generation")
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Stage-2 Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real', 'AI_Edited'])
        ax.set_yticklabels(['Real', 'AI_Edited'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2. else 'black')
        fig.colorbar(im)
        out_png = os.path.join(save_dir, 'stage2_confusion.png')
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved plot to: {out_png}")


def main():
    parser = argparse.ArgumentParser(description='Compute Stage-2 confusion matrix and report (Real vs AI_Edited)')
    parser.add_argument('--stage1_dir', default='results/cascade_stage1')
    parser.add_argument('--stage2_dir', default='results/cascade_stage2')
    parser.add_argument('--save_dir', default='results/cascade_stage2')
    parser.add_argument('--plot', action='store_true', help='Save a confusion matrix image (requires matplotlib)')
    args = parser.parse_args()

    compute_stage2_confusion(args.stage1_dir, args.stage2_dir, args.save_dir, plot=args.plot)


if __name__ == '__main__':
    main()
