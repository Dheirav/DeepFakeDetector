import os
import numpy as np
def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

def confusion_matrix(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm

def classification_report(y_true, y_pred, labels, target_names, digits=4):
    cm = confusion_matrix(y_true, y_pred, labels)
    lines = []
    widths = [max(9, max(len(n) for n in target_names))]
    header = f"{'':12s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>10s}"
    lines.append(header)
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        pred_pos = cm[:, i].sum()
        support = cm[i, :].sum()
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        lines.append(f"{target_names[i]:12s}{precision:10.{digits}f}{recall:10.{digits}f}{f1:10.{digits}f}{support:10d}")
    return "\n".join(lines)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

folders = sorted([os.path.join(ROOT, d) for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

any_found = False
for f in folders:
    y_true_p = os.path.join(f, 'y_true.npy')
    y_pred_p = os.path.join(f, 'y_pred.npy')
    if os.path.isfile(y_true_p) and os.path.isfile(y_pred_p):
        any_found = True
        y_true = np.load(y_true_p)
        y_pred = np.load(y_pred_p)
        acc = accuracy_score(y_true, y_pred)
        print(f"Run: {os.path.basename(f)}")
        print(f"  Samples: {len(y_true)} | Accuracy: {acc*100:.2f}%")
        labels = sorted(set(y_true))
        try:
            report = classification_report(y_true, y_pred, labels=labels, target_names=["Real","AI Generated","AI Edited"], digits=4)
            print(report)
        except Exception:
            print("  Could not produce classification report (label mismatch)")
        try:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            print("  Confusion matrix:")
            print(cm)
        except Exception:
            pass
        print("-"*60)

if not any_found:
    print("No result folders with y_true/y_pred found in results/.")
