import os
import argparse
import numpy as np


def select_subset(probs_file, threshold, save_dir):
    probs = np.load(probs_file)
    # expected shape (N, 3): indices 0=Real,1=AI_Generated,2=AI_Edited
    if probs.ndim != 2 or probs.shape[1] < 3:
        raise RuntimeError(f"Expected probs file with shape (N,3), got {probs.shape}")

    p_real = probs[:, 0]
    p_edited = probs[:, 2]
    uncertainty = np.abs(p_real - p_edited)
    # Determine Stage-1 hard predictions and exclude AI Generated (class 1)
    stage1_preds = probs.argmax(axis=1)
    # Only consider samples where Stage-1 predicted Real(0) or AI_Edited(2).
    candidate_mask = (stage1_preds != 1)
    # Now apply the uncertainty threshold on those candidates.
    indices = np.where((uncertainty <= threshold) & candidate_mask)[0]

    # For diagnostics, report how many AI_Generated samples were excluded from Stage-2
    excluded_ai_gen = np.sum(stage1_preds == 1)
    total = probs.shape[0]

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "stage2_indices.npy")
    np.save(out_path, indices)

    pct = 100.0 * len(indices) / total
    print(f"Selected {len(indices)} samples ({pct:.2f}%) for Stage-2 (threshold={threshold})")
    print(f"Excluded {excluded_ai_gen} samples predicted as AI_Generated (never sent to Stage-2)")
    print(f"Saved indices to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Select subset for Stage-2 based on Stage-1 uncertainty")
    parser.add_argument('--stage1_probs', default="results/cascade_stage1/stage1_probs.npy")
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--save_dir', default="results/cascade_stage2")
    args = parser.parse_args()

    select_subset(args.stage1_probs, args.threshold, args.save_dir)


if __name__ == "__main__":
    main()
