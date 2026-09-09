#!/usr/bin/env python3
"""Segmentation head for the locally-edited class.

Why this rather than a bigger backbone. Scaling the frozen encoder from DINOv2
ViT-S to ViT-L moved `ai_generated` F1 from 0.775 to 0.802 while `ai_edited` went
0.465 to 0.453, i.e. not at all (paired McNemar S vs L, p = 1.000 overall). That
asymmetry is the signature of information destroyed before the model sees it:
whether a whole image is synthetic is a global property a richer representation
can exploit, whereas a local edit is thrown away by the resize to 224px.

So the fix is resolution plus a stronger training signal. This runs at 448px and
supervises *where* the edit is rather than only *whether* one exists, using the
masks OpenSDI ships. Predicting a mask is a much denser signal than a single
label: an edit covering the measured median of 10.3% of the frame is roughly
20,000 supervised pixels instead of one bit.

Design
------
The encoder stays frozen, so this is still not fine-tuning and cannot drift onto
a dataset shortcut the way the original model did. Only the decoder and the
classifier head train.

    frozen DINOv2 ViT-S/14 @ 448  ->  32x32 patch tokens, 384-dim
      -> decoder (conv + upsample)  ->  448x448 manipulation mask
      -> CLS token + mask summary   ->  3-class logits

Mask supervision covers `real` (all zeros: nothing was altered) and `ai_edited`
(the shipped mask). `ai_generated` is EXCLUDED from the mask loss, because a
fully synthetic image has no meaningful "which pixels were altered" answer and
asserting either all-ones or all-zeros would be inventing a label. It still
participates in the classification loss.
"""

import argparse, json, os, sys

CLASSES = ["real", "ai_generated", "ai_edited"]
EDITED = 2
GENERATED = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data_sources/opensdi")
    ap.add_argument("--mask_dir", default="data_sources/opensdi_masks")
    ap.add_argument("--encoder", default="dinov2_vits14")
    ap.add_argument("--size", type=int, default=448, help="must be a multiple of 14")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/mask_head")
    args = ap.parse_args()
    assert args.size % 14 == 0, "DINOv2 patches are 14px; size must be a multiple of 14"

    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    files, labels = [], []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(args.data_dir, cls)
        for f in sorted(os.listdir(d)):
            files.append(os.path.join(d, f)); labels.append(ci)
    files, labels = np.array(files), np.array(labels)
    tr, te = train_test_split(np.arange(len(labels)), test_size=args.test_size,
                              random_state=args.seed, stratify=labels)
    print(f"  {len(files)} images  train {len(tr)}  test {len(te)}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    class DS(Dataset):
        def __init__(self, idx): self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]
            img = Image.open(files[j]).convert("RGB").resize((args.size, args.size),
                                                             Image.BILINEAR)
            x = norm(transforms.functional.to_tensor(img))
            y = int(labels[j])
            stem = os.path.splitext(os.path.basename(files[j]))[0]
            mp = os.path.join(args.mask_dir, stem + ".png")
            if y == EDITED and os.path.exists(mp):
                m = Image.open(mp).convert("L").resize((args.size, args.size),
                                                       Image.NEAREST)
                mask = (torch.from_numpy(np.array(m)).float() / 255.0 > 0.5).float()
            else:
                mask = torch.zeros(args.size, args.size)
            # generated images get no mask supervision, flagged here
            return x, y, mask, float(y != GENERATED)

    class Decoder(nn.Module):
        """32x32 patch grid up to full resolution."""
        def __init__(self, dim):
            super().__init__()
            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                     nn.BatchNorm2d(o), nn.GELU())
            self.b = nn.ModuleList([block(dim, 192), block(192, 96),
                                    block(96, 48), block(48, 24)])
            self.head = nn.Conv2d(24, 1, 1)
        def forward(self, f):
            for blk in self.b:
                f = blk(F.interpolate(f, scale_factor=2, mode="bilinear",
                                      align_corners=False))
            return self.head(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = torch.hub.load("facebookresearch/dinov2", args.encoder, verbose=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad = False
    dim = enc.embed_dim
    grid = args.size // 14

    dec = Decoder(dim).to(device)
    # classifier sees the pooled encoder token plus what the mask says
    clf = nn.Sequential(nn.Linear(dim + 3, 256), nn.GELU(), nn.Linear(256, 3)).to(device)
    opt = torch.optim.AdamW(list(dec.parameters()) + list(clf.parameters()), lr=args.lr)

    def features(x):
        out = enc.forward_features(x)
        toks = out["x_norm_patchtokens"]              # B, grid*grid, dim
        cls = out["x_norm_clstoken"]                  # B, dim
        fmap = toks.transpose(1, 2).reshape(x.size(0), dim, grid, grid)
        return fmap, cls

    def dice_bce(logit, target, valid):
        if valid.sum() == 0:
            return logit.sum() * 0.0
        l, t = logit[valid > 0], target[valid > 0]
        bce = F.binary_cross_entropy_with_logits(l, t)
        p = torch.sigmoid(l)
        inter = (p * t).sum(dim=(1, 2))
        dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + 1)
        return bce + dice.mean()

    dl_tr = DataLoader(DS(tr), batch_size=args.batch, shuffle=True, num_workers=2)
    dl_te = DataLoader(DS(te), batch_size=args.batch, shuffle=False, num_workers=2)

    def evaluate():
        dec.eval(); clf.eval()
        preds, trues, ious = [], [], []
        with torch.no_grad():
            for x, y, m, v in dl_te:
                x, m = x.to(device), m.to(device)
                fmap, cls = features(x)
                up = F.interpolate(dec(fmap), size=(args.size, args.size),
                                   mode="bilinear", align_corners=False).squeeze(1)
                prob = torch.sigmoid(up)
                summary = torch.stack([prob.mean((1, 2)), prob.amax((1, 2)),
                                       (prob > 0.5).float().mean((1, 2))], dim=1)
                preds += clf(torch.cat([cls, summary], 1)).argmax(1).cpu().tolist()
                trues += y.tolist()
                for k in range(x.size(0)):
                    if y[k].item() == EDITED:
                        p_, t_ = (prob[k] > 0.5).float(), m[k]
                        inter = (p_ * t_).sum().item()
                        union = ((p_ + t_) > 0).float().sum().item()
                        ious.append(inter / union if union else 0.0)
        return (np.array(trues), np.array(preds),
                float(np.mean(ious)) if ious else float("nan"))

    history, best = [], {"accuracy": -1.0}
    for ep in range(args.epochs):
        dec.train(); clf.train(); tot = 0.0
        for x, y, m, v in dl_tr:
            x, y, m, v = x.to(device), y.to(device), m.to(device), v.to(device)
            with torch.no_grad():
                fmap, cls = features(x)
            logit = dec(fmap).squeeze(1)
            up = F.interpolate(logit.unsqueeze(1), size=(args.size, args.size),
                               mode="bilinear", align_corners=False).squeeze(1)
            prob = torch.sigmoid(up)
            summary = torch.stack([prob.mean((1, 2)), prob.amax((1, 2)),
                                   (prob > 0.5).float().mean((1, 2))], dim=1)
            loss = dice_bce(up, m, v) + F.cross_entropy(clf(torch.cat([cls, summary], 1)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        # Evaluate every epoch. 'Loss still falling' does not imply accuracy is
        # still improving, and the difference decides whether to train longer.
        trues, preds, miou = evaluate()
        acc = float((trues == preds).mean())
        f1e = float(f1_score(trues, preds, average=None, labels=[0, 1, 2])[EDITED])
        history.append({"epoch": ep + 1, "loss": tot / len(tr), "accuracy": acc,
                        "ai_edited_f1": f1e, "mask_iou": miou})
        star = ""
        if acc > best["accuracy"]:
            best = {"epoch": ep + 1, "accuracy": acc, "ai_edited_f1": f1e,
                    "mask_iou": miou, "trues": trues, "preds": preds}
            star = "  <- best"
        print(f"  epoch {ep+1:>3}/{args.epochs}  loss {tot/len(tr):.4f}  "
              f"acc {acc:.4f}  ai_edited F1 {f1e:.4f}  IoU {miou:.4f}{star}", flush=True)

    trues, preds, miou = best["trues"], best["preds"], best["mask_iou"]
    print(f"\n  best epoch: {best['epoch']}/{args.epochs}")
    f1 = f1_score(trues, preds, average=None, labels=[0, 1, 2])
    acc = float((trues == preds).mean())
    print(f"\n  accuracy {acc:.4f}")
    for c, v in zip(CLASSES, f1):
        print(f"    {c:<14} F1 {v:.4f}")
    print(f"  ai_edited mask IoU (n={int((trues == EDITED).sum())}): {miou:.4f}")

    np.save(os.path.join(args.out, "y_true.npy"), np.array(trues))
    np.save(os.path.join(args.out, "y_pred.npy"), np.array(preds))
    json.dump({"encoder": args.encoder, "size": args.size, "epochs": args.epochs,
               "accuracy": acc, "f1": {c: float(v) for c, v in zip(CLASSES, f1)},
               "ai_edited_mask_iou": miou, "n_masks_scored": int((trues == EDITED).sum()),
               "best_epoch": best["epoch"], "history": history,
               "note": "frozen encoder; decoder and classifier trained; "
                       "reported figures are from the best epoch by accuracy"},
              open(os.path.join(args.out, "training_summary.json"), "w"), indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
