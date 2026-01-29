import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(3), CLASS_NAMES, rotation=45)
    plt.yticks(range(3), CLASS_NAMES)

    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center")

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()


# ---------- TEMP TEST ----------
if __name__ == "__main__":
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    plot_cm(y_true, y_pred)
