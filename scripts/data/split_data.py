from sklearn.model_selection import train_test_split
import os

def split_files(folder, test_size=0.2):
    files = os.listdir(folder)
    train, val = train_test_split(files, test_size=test_size, random_state=42)
    return train, val

for cls in ["real", "ai_generated", "ai_edited"]:
    train, val = split_files(f"data/{cls}")
    print(f"{cls}: Train={len(train)}, Val={len(val)}")
