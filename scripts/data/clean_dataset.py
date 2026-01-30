
import os
import cv2
import pandas as pd
import argparse
import logging

def clean(folder, log):
    removed = 0
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            img = cv2.imread(path)
            if img is None or img.shape[2] != 3:
                raise ValueError("Invalid image")
        except Exception as e:
            log.append([path, str(e)])
            if os.path.exists(path):
                os.remove(path)
                removed += 1
    return removed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean corrupted or non-RGB images from dataset folders.")
    parser.add_argument('--data_dir', type=str, default="data", help='Root data directory')
    parser.add_argument('--output_log', type=str, default="results/cleaning_log.csv", help='Path to save cleaning log')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    log = []
    total_removed = 0
    for cls in ["real", "ai_generated", "ai_edited"]:
        folder = os.path.join(args.data_dir, cls)
        removed = clean(folder, log)
        print(f"{cls}: {removed} files removed.")
        total_removed += removed

    pd.DataFrame(log, columns=["file","issue"]).to_csv(args.output_log, index=False)
    print(f"Cleaning done. Total files removed: {total_removed}. Log saved to {args.output_log}")
