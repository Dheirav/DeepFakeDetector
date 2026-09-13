#!/usr/bin/env python3
"""Run a mask-head checkpoint on your own images.

Prints the three class probabilities per image and writes a PNG with the
predicted edit mask drawn over the photo. This is the test the original model
failed: images that came from nowhere in the training corpus.

The training images were squashed to 512x512, saved as JPEG q90 with EXIF
stripped, then resized to 448 in the loader. By default every input goes
through the same 512/q90 re-encode so the model sees what it was trained on;
--raw skips that step, which is a distribution shift and worth measuring
separately because it is what a phone photo actually looks like.

    venv-linux/bin/python scripts/inference/predict_mask_head.py photo1.jpg photo2.png
    venv-linux/bin/python scripts/inference/predict_mask_head.py --checkpoint \\
        results/mask_head_clip448_ft4/best_model.pth my_photos/
"""

import argparse, io, json, os, sys

CLASSES = ["real", "ai_generated", "ai_edited"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or directories")
    ap.add_argument("--checkpoint", default="results/mask_head_clip448_balanced/best_model.pth")
    ap.add_argument("--raw", action="store_true",
                    help="skip the 512px JPEG q90 re-encode the training data had")
    ap.add_argument("--out", default="results/predictions",
                    help="where the mask overlays go")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--abstain-below", type=float, default=None,
                    help="top probability under which the verdict is 'cannot tell'. "
                         "Default: the decision_rule.json next to the checkpoint, "
                         "else 0.9.")
    args = ap.parse_args()

    paths = []
    for p in args.inputs:
        if os.path.isdir(p):
            paths += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if os.path.splitext(f)[1].lower() in EXTS]
        else:
            paths.append(p)
    if not paths:
        sys.exit("no images found")

    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image, ImageOps
    from torchvision import transforms
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from training.train_mask_head import build_decoder, build_encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu")
    size = ck["size"]
    enc, features, dim, grid = build_encoder(ck["encoder"], size, device,
                                             unfreeze=ck.get("unfreeze", 0))
    if ck.get("encoder_state"):
        res = enc.load_state_dict(ck["encoder_state"], strict=False)
        assert not res.unexpected_keys
    enc.eval()
    dec = build_decoder(dim).to(device); dec.load_state_dict(ck["decoder"]); dec.eval()
    clf = nn.Sequential(nn.Linear(dim + 3, 256), nn.GELU(), nn.Linear(256, 3)).to(device)
    clf.load_state_dict(ck["classifier"]); clf.eval()
    tau = args.abstain_below
    if tau is None:
        rp = os.path.join(os.path.dirname(args.checkpoint), "decision_rule.json")
        tau = json.load(open(rp))["abstain_below"] if os.path.isfile(rp) else 0.9
    tag = ("fine-tuned, last %d blocks" % ck["unfreeze"]) if ck.get("unfreeze") else "frozen encoder"
    print(f"  {ck['encoder']} @ {size}px, {tag}, epoch {ck['epoch']}"
          f"  |  input: {'raw' if args.raw else '512px JPEG q90 re-encode, like training'}"
          f"  |  abstain below {tau:.2f}\n")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def load(path):
        # exif_transpose so a phone photo is upright before anything else;
        # the training data had its EXIF stripped after normalisation.
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if not args.raw:
            im = im.resize((512, 512), Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=90, optimize=True)
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        shown = im.copy()
        x = norm(transforms.functional.to_tensor(im.resize((size, size), Image.BILINEAR)))
        return x, shown

    os.makedirs(args.out, exist_ok=True)
    print(f"  {'image':<40}{'real':>7}{'gen':>7}{'edit':>7}   verdict        mask area")
    with torch.no_grad():
        for p in paths:
            try:
                x, shown = load(p)
            except Exception as e:
                print(f"  {os.path.basename(p)[:40]:<40} could not read: {e}")
                continue
            x = x.unsqueeze(0).to(device)
            fmap, cls = features(x)
            up = F.interpolate(dec(fmap), size=(size, size), mode="bilinear",
                               align_corners=False).squeeze(1)
            prob = torch.sigmoid(up)
            summary = torch.stack([prob.mean((1, 2)), prob.amax((1, 2)),
                                   (prob > 0.5).float().mean((1, 2))], dim=1)
            pr = F.softmax(clf(torch.cat([cls, summary], 1)), dim=1)[0].cpu().numpy()
            k = int(pr.argmax())
            area = float((prob[0] > 0.5).float().mean())
            verdict = CLASSES[k] if pr[k] >= tau else "cannot tell"
            print(f"  {os.path.basename(p)[:40]:<40}{pr[0]:>7.3f}{pr[1]:>7.3f}{pr[2]:>7.3f}"
                  f"   {verdict:<14} {area:5.1%}", flush=True)
            if not args.no_overlay:
                m = (prob[0].cpu().numpy() * 255).astype(np.uint8)
                m = Image.fromarray(m).resize(shown.size, Image.BILINEAR)
                red = Image.new("RGB", shown.size, (255, 0, 0))
                over = Image.composite(red, shown, m.point(lambda v: int(v * 0.6)))
                stem = os.path.splitext(os.path.basename(p))[0]
                over.save(os.path.join(args.out, f"{stem}_{CLASSES[k]}_{pr[k]:.2f}.png"))
    if not args.no_overlay:
        print(f"\n  overlays -> {args.out}/  (red = predicted edited region)")


if __name__ == "__main__":
    main()
