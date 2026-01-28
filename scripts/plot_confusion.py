import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

labels = ["Real", "AI Generated", "AI Edited"]

y_true = [0,1,2,0,1,2]
y_pred = [0,1,1,0,2,2]

cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(3), labels, rotation=45)
plt.yticks(range(3), labels)

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()
