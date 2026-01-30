
import os
import argparse
import pandas as pd

def count_images(path):
    return len([
        f for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report dataset statistics (image counts per class)")
    parser.add_argument('--data_dir', type=str, default="data", help='Root data directory')
    parser.add_argument('--output_log', type=str, default="results/dataset_stats.csv", help='Path to save stats log')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    classes = ["real", "ai_generated", "ai_edited"]
    stats = []
    total = 0
    print("Dataset Statistics")
    print("------------------")
    for cls in classes:
        count = count_images(os.path.join(args.data_dir, cls))
        total += count
        stats.append({"class": cls, "count": count})
        print(f"{cls}: {count} images")
    print("------------------")
    print(f"Total images: {total}")
    pd.DataFrame(stats).to_csv(args.output_log, index=False)
    print(f"Stats log saved to {args.output_log}")
