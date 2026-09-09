#!/usr/bin/env python3
"""Measure how a detector holds up under content-preserving image degradation.

This is the test that exposed the original model. Downscaling an AI-generated
image and re-saving it as JPEG cannot change whether it is AI-generated, so a
detector that reads generative artefacts should be roughly unaffected, while one
that reads file provenance collapses. The fine-tuned ConvNeXt went from 0.993 to
0.027 accuracy under a 256px JPEG q80 re-save, calling 280 of 300 AI-generated
images real at 87% mean confidence.

Every transform below preserves the label. Any accuracy drop is the detector
depending on something other than the content.
"""

import argparse
import io
import json
import os
import sys

CLASSES = ["real", "ai_generated", "ai_edited"]
CONDITIONS = [
    ("clean", None, None),
    ("448 / q90", 448, 90),
    ("384 / q75", 384, 75),
    ("320 / q85", 320, 85),
    ("256 / q80", 256, 80),
    ("256 / q60", 256, 60),
    ("224 / q50", 224, 50),
]


def degrade(img, size, quality):
    if size:
        img = img.resize((size, size), __import__("PIL.Image", fromlist=["Image"]).LANCZOS)
    if quality:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        from PIL import Image
        img = Image.open(buf).convert("RGB")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="results/linear_probe/features_dinov2_vits14.npz")
    ap.add_argument("--encoder", default="dinov2_vits14")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=600, help="test images per condition")
    ap.add_argument("--out", default="results/linear_probe/degradation.json")
    args = ap.parse_args()

    import numpy as np, torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from torchvision import transforms

    blob = np.load(args.features, allow_pickle=True)
    X, y, paths = blob["X"], blob["y"], blob["paths"]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=args.test_size,
                              random_state=args.seed, stratify=y)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
    print(f"  probe trained on {len(tr)}; evaluating {min(args.limit, len(te))} held-out images")

    rng = np.random.default_rng(args.seed)
    te = rng.permutation(te)[:args.limit]
    te_paths, te_y = paths[te], y[te]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = torch.hub.load("facebookresearch/dinov2", args.encoder, verbose=False)
    enc.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print(f"\n  {'condition':<14}{'accuracy':>10}{'real':>9}{'ai_gen':>9}{'ai_edit':>9}")
    results = {}
    for name, size, q in CONDITIONS:
        feats = []
        for i in range(0, len(te_paths), args.batch):
            batch = []
            for p in te_paths[i:i + args.batch]:
                with Image.open(p) as im:
                    batch.append(tf(degrade(im.convert("RGB"), size, q)))
            with torch.no_grad():
                feats.append(enc(torch.stack(batch).to(device)).cpu().numpy())
        pred = clf.predict(np.concatenate(feats))
        acc = float((pred == te_y).mean())
        per = [float((pred[te_y == c] == c).mean()) if (te_y == c).any() else float("nan")
               for c in range(3)]
        results[name] = {"accuracy": acc, "per_class_recall": per}
        print(f"  {name:<14}{acc:>10.3f}{per[0]:>9.3f}{per[1]:>9.3f}{per[2]:>9.3f}")

    clean = results["clean"]["accuracy"]
    worst = min(r["accuracy"] for r in results.values())
    print(f"\n  clean {clean:.3f}  worst {worst:.3f}  drop {clean - worst:.3f}")
    print("  the fine-tuned ConvNeXt on the old confounded corpus dropped 0.993 -> 0.027")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
