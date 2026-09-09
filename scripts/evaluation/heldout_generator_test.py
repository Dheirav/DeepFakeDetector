#!/usr/bin/env python3
"""Evaluate a trained probe on generators it never saw.

OpenSDI trains on sd15 only and ships a test set spanning flux, sd2, sd3, sdxl
and sd15, which makes leave-one-generator-out the protocol the dataset was built
for. sd15 in the test set is the control: unseen images, seen generator. If sd15
holds while the others fall, the drop is attributable to the generator rather
than to the probe overfitting its training images.

Per-class recall rather than overall accuracy, because the converted test
directories do not all contain all three classes and an accuracy computed over
different class mixes is not comparable across generators.
"""
import argparse, json, os, sys

CLASSES = ["real", "ai_generated", "ai_edited"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data_sources/heldout")
    ap.add_argument("--features", default="results/linear_probe/features_dinov2_vits14.npz")
    ap.add_argument("--encoder", default="dinov2_vits14")
    ap.add_argument("--control", default="sd15", help="generator seen during training")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results/linear_probe/heldout_generators.json")
    args = ap.parse_args()

    import numpy as np, torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from torchvision import transforms

    b = np.load(args.features, allow_pickle=True)
    X, y = b["X"], b["y"]
    tr, _ = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
    print(f"  probe trained on {len(tr)} sd15 images\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = torch.hub.load("facebookresearch/dinov2", args.encoder, verbose=False)
    enc.eval().to(device)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])

    def recall(paths, true_idx):
        preds = []
        for i in range(0, len(paths), args.batch):
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + args.batch]]).to(device)
            with torch.no_grad():
                preds.append(clf.predict(enc(batch).cpu().numpy()))
        p = np.concatenate(preds)
        return float((p == true_idx).mean()), np.bincount(p, minlength=3).tolist()

    results = {}
    gens = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)))
    print(f"  {'generator':<10}{'class':<15}{'n':>6}{'recall':>9}   predicted as")
    for g in gens:
        results[g] = {}
        for ci, cls in enumerate(CLASSES):
            d = os.path.join(args.root, g, cls)
            if not os.path.isdir(d):
                continue
            paths = [os.path.join(d, f) for f in sorted(os.listdir(d))]
            if not paths:
                continue
            r, dist = recall(paths, ci)
            results[g][cls] = {"n": len(paths), "recall": r, "pred_dist": dist}
            tag = "  (control)" if g == args.control else ""
            print(f"  {g:<10}{cls:<15}{len(paths):>6}{r:>9.3f}   "
                  f"real={dist[0]} gen={dist[1]} edit={dist[2]}{tag}", flush=True)
        print()

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
