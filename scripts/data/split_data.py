
from sklearn.model_selection import train_test_split
import os
import argparse
import pandas as pd

def split_files(folder, test_size=0.2):
    files = os.listdir(folder)
    train, val = train_test_split(files, test_size=test_size, random_state=42)
    return train, val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument('--data_dir', type=str, default="data", help='Root data directory')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output_log', type=str, default="results/split_log.csv", help='Path to save split log')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    split_log = []
    for cls in ["real", "ai_generated", "ai_edited"]:
        folder = os.path.join(args.data_dir, cls)
        train, val = split_files(folder, test_size=args.test_size)
        split_log.append({"class": cls, "train": len(train), "val": len(val)})
        print(f"{cls}: Train={len(train)}, Val={len(val)}")

    pd.DataFrame(split_log).to_csv(args.output_log, index=False)
    print(f"Split log saved to {args.output_log}")
