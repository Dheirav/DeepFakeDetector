import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

    cm = confusion_matrix(y_true, y_pred)
    return cm


# ---------- TEMP TEST (REMOVE AFTER INTEGRATION) ----------
if __name__ == "__main__":
    # Dummy predictions for testing structure
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    cm = evaluate_model(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
