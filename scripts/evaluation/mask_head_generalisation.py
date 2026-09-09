#!/usr/bin/env python3
"""Evaluate a trained mask head on generators it never saw.

The linear probe was tested this way and showed transfer tracking architectural
distance: 0.718 recall on sd2 and 0.667 on sd3, both close relatives of the sd15
training generator, against 0.463 on sdxl and 0.470 on flux. The mask head has
never been tested that way, and it is the number that decides whether any of the
in-distribution gains matter.

Reports mask IoU per generator as well as class recall, which the probe could not
do. That separates two different failure modes: the model may still find the
edited region on an unfamiliar generator while misclassifying it, or it may fail
to localise at all. Those call for different fixes.

Per-class recall rather than overall accuracy, because the converted held-out
directories hold different class mixes and an accuracy across them would not be
comparable between generators.
"""

import argparse, json, os, sys

CLASSES = ["real", "ai_generated", "ai_edited"]
EDITED = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="results/mask_head_weighted/best_model.pth")
    ap.add_argument("--root", default="data_sources/heldout")
    ap.add_argument("--control", default="sd15", help="generator seen during training")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/mask_head_weighted/heldout_generators.json")
    args = ap.parse_args()

    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from training.train_mask_head import build_decoder

    ck = torch.load(args.checkpoint, map_location="cpu")
    size, encoder_name = ck["size"], ck["encoder"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = torch.hub.load("facebookresearch/dinov2", encoder_name, verbose=False)
    enc.eval().to(device)
    dim, grid = enc.embed_dim, size // 14

    dec = build_decoder(dim).to(device); dec.load_state_dict(ck["decoder"]); dec.eval()
    clf = nn.Sequential(nn.Linear(dim + 3, 256), nn.GELU(), nn.Linear(256, 3)).to(device)
    clf.load_state_dict(ck["classifier"]); clf.eval()
    print(f"  {encoder_name} @ {size}px, checkpoint from epoch {ck['epoch']}\n")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def run(paths, mask_dir):
        preds, ious = [], []
        for i in range(0, len(paths), args.batch):
            chunk = paths[i:i + args.batch]
            xs = []
            for p in chunk:
                im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
                xs.append(norm(transforms.functional.to_tensor(im)))
            x = torch.stack(xs).to(device)
            with torch.no_grad():
                out = enc.forward_features(x)
                fmap = out["x_norm_patchtokens"].transpose(1, 2).reshape(
                    x.size(0), dim, grid, grid)
                up = F.interpolate(dec(fmap), size=(size, size), mode="bilinear",
                                   align_corners=False).squeeze(1)
                prob = torch.sigmoid(up)
                summary = torch.stack([prob.mean((1, 2)), prob.amax((1, 2)),
                                       (prob > 0.5).float().mean((1, 2))], dim=1)
                preds += clf(torch.cat([out["x_norm_clstoken"], summary], 1)) \
                             .argmax(1).cpu().tolist()
            if mask_dir:
                for k, p in enumerate(chunk):
                    mp = os.path.join(mask_dir,
                                      os.path.splitext(os.path.basename(p))[0] + ".png")
                    if not os.path.exists(mp):
                        continue
                    m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
                    t = (torch.from_numpy(np.array(m)).float() / 255.0 > 0.5).float().to(device)
                    pr = (prob[k] > 0.5).float()
                    union = ((pr + t) > 0).float().sum().item()
                    ious.append((pr * t).sum().item() / union if union else 0.0)
        return np.array(preds), ious

    results = {}
    gens = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)) and not d.endswith("_masks"))
    print(f"  {'generator':<10}{'class':<15}{'n':>6}{'recall':>9}{'IoU':>8}   predicted as")
    for g in gens:
        results[g] = {}
        for ci, cls in enumerate(CLASSES):
            d = os.path.join(args.root, g, cls)
            if not os.path.isdir(d):
                continue
            paths = [os.path.join(d, f) for f in sorted(os.listdir(d))]
            if not paths:
                continue
            md = os.path.join(args.root, g + "_masks") if ci == EDITED else None
            if md and not os.path.isdir(md):
                md = None
            pred, ious = run(paths, md)
            rec = float((pred == ci).mean())
            miou = float(np.mean(ious)) if ious else float("nan")
            dist = np.bincount(pred, minlength=3).tolist()
            results[g][cls] = {"n": len(paths), "recall": rec, "mask_iou": miou,
                               "pred_dist": dist}
            tag = "  (control)" if g == args.control else ""
            iou_s = f"{miou:>8.3f}" if ious else f"{'-':>8}"
            print(f"  {g:<10}{cls:<15}{len(paths):>6}{rec:>9.3f}{iou_s}   "
                  f"real={dist[0]} gen={dist[1]} edit={dist[2]}{tag}", flush=True)
        print()

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
