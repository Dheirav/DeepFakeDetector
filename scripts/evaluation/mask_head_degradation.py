#!/usr/bin/env python3
"""The degradation grid, for a mask-head checkpoint.

scripts/evaluation/degradation_test.py ran this on the linear probe: 0.667 clean
to 0.533 at 256px q80 against a 0.520 majority baseline, and it was never
repeated on the mask heads. Same conditions, same idea: every transform keeps
the label, so any drop is the model reading something other than content.

Two things are reported per condition that the probe version could not:
mask IoU on the edited class, because localisation should degrade first if the
cue is fine texture; and coverage under the abstain rule, because a model that
declines more as the image degrades is behaving correctly and one that stays
confident while getting worse is not.

The test split is rebuilt exactly as train_mask_head.py built it (sorted
listing, --max-per-class cap, stratified 30 percent at seed 42), so the images
scored here are ones the checkpoint never trained on.
"""

import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.degradation_test import CONDITIONS, degrade  # noqa: E402

CLASSES = ["real", "ai_generated", "ai_edited"]
EDITED = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="results/mask_head_clip448_balanced/best_model.pth")
    ap.add_argument("--data_dir", default="data_sources/opensdi_large")
    ap.add_argument("--mask_dir", default="data_sources/opensdi_large_masks")
    ap.add_argument("--max-per-class", type=int, default=4500)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=600, help="test images per condition")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="defaults to <checkpoint dir>/degradation.json")
    args = ap.parse_args()
    args.out = args.out or os.path.join(os.path.dirname(args.checkpoint), "degradation.json")

    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from training.train_mask_head import build_decoder, build_encoder

    files, labels = [], []
    for ci, cls in enumerate(CLASSES):
        names = sorted(os.listdir(os.path.join(args.data_dir, cls)))[:args.max_per_class]
        files += [os.path.join(args.data_dir, cls, f) for f in names]; labels += [ci] * len(names)
    files, labels = np.array(files), np.array(labels)
    _, te = train_test_split(np.arange(len(labels)), test_size=args.test_size,
                             random_state=args.seed, stratify=labels)
    te = np.random.default_rng(args.seed).permutation(te)[:args.limit]
    paths, y = files[te], labels[te]
    print(f"  {len(paths)} held-out test images, class counts {np.bincount(y, minlength=3).tolist()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu")
    size = ck["size"]
    enc, features, dim, grid = build_encoder(ck["encoder"], size, device, unfreeze=ck.get("unfreeze", 0))
    if ck.get("encoder_state"):
        enc.load_state_dict(ck["encoder_state"], strict=False)
    enc.eval()
    dec = build_decoder(dim).to(device); dec.load_state_dict(ck["decoder"]); dec.eval()
    clf = nn.Sequential(nn.Linear(dim + 3, 256), nn.GELU(), nn.Linear(256, 3)).to(device)
    clf.load_state_dict(ck["classifier"]); clf.eval()
    rule = os.path.join(os.path.dirname(args.checkpoint), "decision_rule.json")
    tau = json.load(open(rule))["abstain_below"] if os.path.isfile(rule) else 0.9
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    print(f"  {ck['encoder']} @ {size}px, epoch {ck['epoch']}, abstain below {tau:.2f}\n")

    def load_mask(p):
        mp = os.path.join(args.mask_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
        if not os.path.exists(mp):
            return None
        m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
        return (np.asarray(m) > 127)

    print(f"  {'condition':<14}{'accuracy':>10}{'real':>8}{'ai_gen':>8}{'ai_edit':>8}{'IoU':>8}{'coverage':>10}{'acc|ans':>9}")
    results = {}
    for name, dsize, q in CONDITIONS:
        preds, confs, ious = [], [], []
        for i in range(0, len(paths), args.batch):
            batch, masks = [], []
            for p in paths[i:i + args.batch]:
                with Image.open(p) as im:
                    img = degrade(im.convert("RGB"), dsize, q).resize((size, size), Image.BILINEAR)
                batch.append(norm(transforms.functional.to_tensor(img)))
                masks.append(load_mask(p))
            x = torch.stack(batch).to(device)
            with torch.no_grad():
                fmap, cls = features(x)
                up = F.interpolate(dec(fmap), size=(size, size), mode="bilinear",
                                   align_corners=False).squeeze(1)
                prob = torch.sigmoid(up)
                summary = torch.stack([prob.mean((1, 2)), prob.amax((1, 2)),
                                       (prob > 0.5).float().mean((1, 2))], dim=1)
                pr = F.softmax(clf(torch.cat([cls, summary], 1)), dim=1)
            preds += pr.argmax(1).cpu().tolist(); confs += pr.max(1).values.cpu().tolist()
            pm = (prob > 0.5).cpu().numpy()
            for k, p in enumerate(paths[i:i + args.batch]):
                yk = y[i + k]
                if yk == EDITED and masks[k] is not None:
                    inter = (pm[k] & masks[k]).sum(); union = (pm[k] | masks[k]).sum()
                    ious.append(inter / union if union else 0.0)
        preds, confs = np.array(preds), np.array(confs)
        acc = float((preds == y).mean())
        per = [float((preds[y == c] == c).mean()) for c in range(3)]
        ans = confs >= tau
        cov = float(ans.mean()); acc_ans = float((preds[ans] == y[ans]).mean()) if ans.any() else float("nan")
        miou = float(np.mean(ious)) if ious else float("nan")
        results[name] = {"accuracy": acc, "per_class_recall": per, "mask_iou": miou,
                         "coverage_at_tau": cov, "accuracy_when_answered": acc_ans, "tau": tau}
        print(f"  {name:<14}{acc:>10.3f}{per[0]:>8.3f}{per[1]:>8.3f}{per[2]:>8.3f}{miou:>8.3f}{cov:>10.3f}{acc_ans:>9.3f}", flush=True)

    clean, worst = results["clean"]["accuracy"], min(r["accuracy"] for r in results.values())
    base = float(np.bincount(y).max() / len(y))
    print(f"\n  clean {clean:.3f}  worst {worst:.3f}  drop {clean - worst:.3f}  majority baseline {base:.3f}")
    print("  linear probe on the same corpus: 0.667 clean, 0.533 at 256/q80; original ConvNeXt: 0.993 -> 0.027")
    results["_meta"] = {"checkpoint": args.checkpoint, "n": int(len(y)), "majority_baseline": base}
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
