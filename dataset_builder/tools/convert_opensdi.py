#!/usr/bin/env python3
"""Convert OpenSDI parquet shards into this project's three-class image layout.

OpenSDI ships a binary label, but its `key` field encodes the distinction this
project actually needs:

    entire/<gen>/real  +  partial/<gen>/real   ->  real
    entire/<gen>/fake                          ->  ai_generated   (fully synthetic)
    partial/<gen>/fake                         ->  ai_edited      (locally edited)

Shards are homogeneous, one class each, so a contiguous slice is single-class and
useless. Measured layout of the sd15 split (70 shards, 2869 rows each):

    0 to ~34    partial/fake
    ~35 to ~60  partial/real
    ~61 to ~64  entire/fake
    ~65 to 69   entire/real

Why every image is re-encoded
-----------------------------
Measured on shards 0 and 35, which hold the same photographs edited and unedited:
geometry is clean (format, resolution, megapixels and aspect ratio all score
BELOW the majority-class baseline), but the JPEG quantisation table separates the
classes at 93.8%. The originals carry 3 distinct tables while the edited images
carry 16, because editing requires re-saving and the editor used different
quality settings. That signal is "was this re-encoded by the editor", not a
manipulation trace.

This is intrinsic to any locally-manipulated dataset rather than an OpenSDI
defect. Re-encoding both classes identically removes it: measured 93.0% before,
49.0% after, against a 51.0% baseline. So normalisation here is mandatory, not a
tuning option. Training on the shards as shipped gives you a detector for the
editing tool's JPEG encoder.

Masks are written alongside the edited images, because at 224 pixels the edited
class is not learnable from image-level labels: DEFACTO-style manipulations average
1.7% tampered pixels, and published methods score 0.8 to 6.9 percent on tampered
detection at that resolution.
"""

import argparse
import collections
import io
import os
import sys

CLASS_OF = {"entire/real": "real", "partial/real": "real",
            "entire/fake": "ai_generated", "partial/fake": "ai_edited"}

# One shard per region, from the measured layout above.
DEFAULT_SHARDS = [0, 35, 63, 66]


def class_of(key):
    parts = str(key).split("/")
    if len(parts) < 3:
        return None
    return CLASS_OF.get(f"{parts[0]}/{parts[2]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_sources/opensdi", help="output root")
    ap.add_argument("--shards", type=int, nargs="+", default=DEFAULT_SHARDS)
    ap.add_argument("--repo", default="nebula/OpenSDI_train")
    ap.add_argument("--split", default="sd15")
    ap.add_argument("--total-shards", type=int, default=70)
    ap.add_argument("--size", type=int, default=512, help="output edge, pixels")
    ap.add_argument("--quality", type=int, default=90, help="output JPEG quality")
    ap.add_argument("--per-shard", type=int, default=1000, help="images per shard")
    ap.add_argument("--cache", default=os.path.expanduser("~/opensdi_probe"))
    ap.add_argument("--keep-shards", action="store_true",
                    help="do not delete a shard after converting it")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    counts = collections.Counter()
    masks = 0

    for n in args.shards:
        name = f"data/{args.split}-{n:05d}-of-{args.total_shards:05d}.parquet"
        local = os.path.join(args.cache, name)
        if not os.path.exists(local):
            print(f"  fetching shard {n} ...", flush=True)
            local = hf_hub_download(args.repo, name, repo_type="dataset",
                                    local_dir=args.cache)
        pf = pq.ParquetFile(local)
        wrote = 0
        # Stream row groups. Reading the whole shard as Python objects is several
        # GB for a 490MB image shard, which is enough to kill an 8GB WSL VM.
        for batch in pf.iter_batches(batch_size=32,
                                     columns=["key", "image", "mask"]):
            for rec in batch.to_pylist():
                cls = class_of(rec["key"])
                if cls is None:
                    continue
                stem = os.path.splitext(os.path.basename(str(rec["key"])))[0]
                sub = f"{'partial' if 'partial' in str(rec['key']) else 'entire'}_{stem}"

                d = os.path.join(args.out, cls)
                os.makedirs(d, exist_ok=True)
                blob = rec["image"]
                blob = blob["bytes"] if isinstance(blob, dict) else blob
                with Image.open(io.BytesIO(blob)) as im:
                    im = im.convert("RGB").resize((args.size, args.size), Image.LANCZOS)
                    im.save(os.path.join(d, f"{sub}.jpg"), "JPEG",
                            quality=args.quality, optimize=True)
                counts[cls] += 1

                mblob = rec.get("mask")
                mblob = mblob["bytes"] if isinstance(mblob, dict) else mblob
                if mblob and cls == "ai_edited":
                    # Masks live OUTSIDE the class root. DeepfakeDataset only
                    # scans known class names so it would ignore them, but any
                    # tool that treats every subdirectory as a class (the
                    # confound checker does) would score "masks" as a fourth.
                    md = args.out.rstrip("/") + "_masks"
                    os.makedirs(md, exist_ok=True)
                    with Image.open(io.BytesIO(mblob)) as mk:
                        mk.convert("L").resize((args.size, args.size), Image.NEAREST) \
                          .save(os.path.join(md, f"{sub}.png"), "PNG", optimize=True)
                    masks += 1

                wrote += 1
                if wrote >= args.per_shard:
                    break
            if wrote >= args.per_shard:
                break
        print(f"  shard {n:>2}: wrote {wrote}", flush=True)
        if not args.keep_shards and os.path.exists(local):
            os.remove(local)

    print(f"\n  classes: {dict(counts)}")
    print(f"  masks written: {masks}")
    print(f"  all images: {args.size}x{args.size} JPEG q{args.quality}, EXIF stripped")
    print(f"  -> {args.out}")
    print("\n  Verify before training:")
    print(f"    venv-linux/bin/python scripts/data/metadata_confound.py {args.out}")


if __name__ == "__main__":
    main()
