from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate(y_true, y_pred):
    print("Classification Report")
    print(classification_report(
        y_true, y_pred,
        target_names=["Real", "AI Generated", "AI Edited"]
    ))

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

# Dummy example (for structure)
y_true = np.array([0,1,2,0,1,2])
y_pred = np.array([0,1,1,0,2,2])

evaluate(y_true, y_pred)
