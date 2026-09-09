#!/usr/bin/env python3
"""Train a linear probe on frozen self-supervised features.

Why this rather than fine-tuning a backbone.

Fine-tuning lets the network move its features toward whatever separates the
training classes, and when a dataset carries a shortcut that is what it moves
toward. This project's own history is the example: the fine-tuned ConvNeXt
reached 89% by reading file provenance, and its accuracy inverted to 2.7% under a
routine JPEG re-save. Kumar et al. (ICLR 2022) measured the general form of this,
finding fine-tuning about 2 points better in-domain and about 7 points worse
out-of-domain than a linear probe on the same frozen features.

A probe cannot do that, because the backbone never moves. It is also the honest
control: if a fine-tuned network cannot beat a linear probe on frozen features,
it learned nothing the features did not already contain.

Features are cached to disk, so re-training with different hyperparameters costs
seconds rather than re-running the encoder.
"""

import argparse
import json
import os
import sys

CLASSES = ["real", "ai_generated", "ai_edited"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data_sources/opensdi")
    ap.add_argument("--encoder", default="dinov2_vits14")
    ap.add_argument("--out", default="results/linear_probe")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recompute", action="store_true", help="ignore the feature cache")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.model_selection import train_test_split
    from torchvision import transforms

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, f"features_{args.encoder}.npz")

    if os.path.exists(cache) and not args.recompute:
        blob = np.load(cache, allow_pickle=True)
        X, y, paths = blob["X"], blob["y"], blob["paths"]
        print(f"  loaded cached features: {X.shape}")
    else:
        files, labels = [], []
        for idx, cls in enumerate(CLASSES):
            d = os.path.join(args.data_dir, cls)
            if not os.path.isdir(d):
                sys.exit(f"missing class directory: {d}")
            for f in sorted(os.listdir(d)):
                files.append(os.path.join(d, f))
                labels.append(idx)
        print(f"  {len(files)} images across {len(CLASSES)} classes")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", args.encoder, verbose=False)
        model.eval().to(device)
        # DINOv2 patches are 14px, so the edge must be a multiple of 14.
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        feats = []
        for i in range(0, len(files), args.batch):
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for p in files[i:i + args.batch]]).to(device)
            with torch.no_grad():
                feats.append(model(batch).cpu().numpy())
            if (i // args.batch) % 20 == 0:
                print(f"    {min(i + args.batch, len(files))}/{len(files)}", flush=True)
        X = np.concatenate(feats)
        y = np.array(labels)
        paths = np.array(files)
        np.savez_compressed(cache, X=X, y=y, paths=paths)
        print(f"  features {X.shape} cached -> {cache}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    print(f"\n  train {len(ytr)}  test {len(yte)}")

    # multi_class was removed in sklearn 1.7; multinomial is the default.
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    print(f"\n  test accuracy: {acc:.4f}")
    print("\n" + classification_report(yte, pred, target_names=CLASSES, digits=4))
    print("  confusion matrix (rows = true):")
    for name, row in zip(CLASSES, confusion_matrix(yte, pred)):
        print(f"    {name:<14} {row}")

    np.save(os.path.join(args.out, "y_true.npy"), yte)
    np.save(os.path.join(args.out, "y_pred.npy"), pred)
    json.dump({"encoder": args.encoder, "test_accuracy": float(acc),
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "data_dir": args.data_dir, "seed": args.seed,
               "note": "linear probe on frozen features; backbone never updated"},
              open(os.path.join(args.out, "training_summary.json"), "w"), indent=2)
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
