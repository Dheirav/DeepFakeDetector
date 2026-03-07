"""
Download Midjourney and DALL-E 3 generated images from HuggingFace.

Sources:
  - ehristoforu/midjourney-images   (~8.6k images, Midjourney v5/v6)
  - ehristoforu/dalle-3-images      (~6.6k images, DALL-E 3)

Output: data_sources/ai_generated/Midjourney_DALLE/
File naming:
  - mj_XXXXXX.jpg  for Midjourney images
  - dalle_XXXXXX.jpg for DALL-E 3 images
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_source(ds_id, prefix, out_dir, max_images, min_size=256, start_idx=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count already downloaded images for this prefix
    existing = len(list(out_dir.glob(f"{prefix}_*.jpg")))
    print(f"\n[{prefix}] Resuming: {existing} images already exist")

    if existing >= max_images:
        print(f"[{prefix}] Already have {existing} >= {max_images}, skipping.")
        return existing

    print(f"[{prefix}] Downloading from {ds_id} (target: {max_images}, already have: {existing})")
    ds = load_dataset(ds_id, split="train", streaming=True)

    saved = existing
    scanned = 0

    with tqdm(total=max_images - existing, desc=f"Downloading {prefix}") as pbar:
        for example in ds:
            if saved >= max_images:
                break

            img = example.get("image")
            if img is None:
                continue

            scanned += 1

            try:
                # Convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                w, h = img.size
                if w < min_size or h < min_size:
                    continue

                fname = out_dir / f"{prefix}_{saved:06d}.jpg"
                img.save(str(fname), "JPEG", quality=95)
                saved += 1
                pbar.update(1)

            except Exception as e:
                print(f"\nWarning: skipped image {scanned}: {e}", file=sys.stderr)
                continue

    print(f"[{prefix}] Done. Saved {saved} images (scanned {scanned}).")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download Midjourney + DALL-E 3 images")
    parser.add_argument("--out-dir", default="data_sources/ai_generated/Midjourney_DALLE",
                        help="Output directory")
    parser.add_argument("--max-mj", type=int, default=3500,
                        help="Max Midjourney images to download")
    parser.add_argument("--max-dalle", type=int, default=3500,
                        help="Max DALL-E 3 images to download")
    parser.add_argument("--min-size", type=int, default=256,
                        help="Min image width/height in pixels")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir.resolve()}")
    print(f"Targets: {args.max_mj} Midjourney + {args.max_dalle} DALL-E 3")

    mj_count = download_source(
        ds_id="ehristoforu/midjourney-images",
        prefix="mj",
        out_dir=out_dir,
        max_images=args.max_mj,
        min_size=args.min_size,
    )

    dalle_count = download_source(
        ds_id="ehristoforu/dalle-3-images",
        prefix="dalle",
        out_dir=out_dir,
        max_images=args.max_dalle,
        min_size=args.min_size,
    )

    total = mj_count + dalle_count
    print(f"\nTotal images saved: {total} ({mj_count} Midjourney + {dalle_count} DALL-E 3)")
    print(f"Location: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
