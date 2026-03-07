#!/usr/bin/env python3
"""
Download Midjourney/DALL-E top-up images to close the ai_generated gap.

Saves to data_sources/ai_generated/Midjourney_TopUp/ — separate from the original
Midjourney_DALLE/ directory to avoid cross-run export overlap.

Sources (same HuggingFace datasets as original, different output dir):
  - ehristoforu/midjourney-images   (~8.6k images, Midjourney v5/v6)
  - ehristoforu/dalle-3-images      (~6.6k images, DALL-E 3)

Target: 2,200 images total (buffer for the 1,778 pipeline target after filtering)

Usage:
    cd dataset_builder/scripts
    python download_mj_topup.py
"""

import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

TARGET_DIR = (
    Path(__file__).resolve().parents[2]
    / "data_sources" / "ai_generated" / "Midjourney_TopUp"
)

SOURCES = [
    {"dataset": "ehristoforu/midjourney-images", "prefix": "mj_topup", "max": 1200},
    {"dataset": "ehristoforu/dalle-3-images",    "prefix": "dalle_topup", "max": 1000},
]

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output : {TARGET_DIR}")
print(f"Target : {sum(s['max'] for s in SOURCES)} images total")
print()

total_saved = sum(1 for f in TARGET_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png"})
print(f"Existing: {total_saved} images\n")


def download_source(dataset_id: str, prefix: str, max_images: int):
    existing = len(list(TARGET_DIR.glob(f"{prefix}_*.jpg")))
    if existing >= max_images:
        print(f"[{prefix}] Already have {existing}/{max_images} — skipping.")
        return existing

    print(f"[{prefix}] Streaming {dataset_id} (target: {max_images}, have: {existing})")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    saved = existing
    skipped = 0
    with tqdm(total=max_images - existing, desc=f"  {prefix}") as pbar:
        for example in ds:
            if saved >= max_images:
                break
            try:
                img = example.get("image")
                if img is None:
                    skipped += 1
                    continue
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.width < 256 or img.height < 256:
                    skipped += 1
                    continue
                out = TARGET_DIR / f"{prefix}_{saved:06d}.jpg"
                if out.exists():
                    saved += 1
                    pbar.update(1)
                    continue
                img.save(out, "JPEG", quality=95)
                saved += 1
                pbar.update(1)
            except Exception as e:
                skipped += 1
    print(f"  Saved {saved}, skipped {skipped}")
    return saved


for src in SOURCES:
    download_source(src["dataset"], src["prefix"], src["max"])

final = sum(1 for f in TARGET_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png"})
print(f"\nDone. {final} total images in {TARGET_DIR}")
print("\nNext step:")
print("  cd dataset_builder && python main.py --config config/ai_generated_mj_topup_config.yaml")
