#!/usr/bin/env python3
"""Download FLUX.1 generated images from HuggingFace into data_sources/ai_generated/FLUX/"""
import os
from pathlib import Path

DATASET_NAME = "ash12321/flux-1-dev-generated-10k"
TARGET_DIR = Path(__file__).resolve().parents[2] / "data_sources" / "ai_generated" / "FLUX"
TARGET_COUNT = 4500  # download extra buffer beyond the 3000 pipeline target

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Saving to: {TARGET_DIR}")
print(f"Target count: {TARGET_COUNT}")
print(f"Dataset: {DATASET_NAME}")
print()

from datasets import load_dataset

ds = load_dataset(DATASET_NAME, split="train", streaming=True)

saved = 0
skipped = 0
for i, example in enumerate(ds):
    if saved >= TARGET_COUNT:
        break
    try:
        img = example["image"]
        # Skip non-RGB
        if img.mode not in ("RGB", "RGBA", "L"):
            skipped += 1
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Skip small images
        if img.width < 256 or img.height < 256:
            skipped += 1
            continue
        out_path = TARGET_DIR / f"flux_{saved:06d}.jpg"
        img.save(out_path, "JPEG", quality=95)
        saved += 1
        if saved % 100 == 0:
            print(f"  {saved}/{TARGET_COUNT} saved (scanned {i+1}, skipped {skipped})")
    except Exception as e:
        skipped += 1
        print(f"  Warning: skipped example {i}: {e}")

print(f"\nDone. Saved {saved} images to {TARGET_DIR}")
print(f"Skipped {skipped} images.")
