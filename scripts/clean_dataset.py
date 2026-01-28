import os, cv2, pandas as pd

log = []

def clean(folder):
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            img = cv2.imread(path)
            if img is None or img.shape[2] != 3:
                raise ValueError("Invalid image")
        except:
            log.append([path, "Corrupt or non-RGB"])
            if os.path.exists(path):
                os.remove(path)

for cls in ["real", "ai_generated", "ai_edited"]:
    clean(f"data/{cls}")

pd.DataFrame(log, columns=["file","issue"]).to_csv(
    "results/cleaning_log.csv", index=False
)

print("Cleaning done")
