#!/usr/bin/env python3
"""
Download additional FLUX.1 generated images (top-up batch) to a separate directory.

Saves to data_sources/ai_generated/FLUX_TopUp/ — separate from the original FLUX/
directory — to avoid any cross-run export overlap.

Target: 2,500 images (buffer for the 2,000 pipeline target after filtering)

Usage:
    cd dataset_builder/scripts
    python download_flux_topup.py
"""

from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "ash12321/flux-1-dev-generated-10k"
TARGET_DIR = Path(__file__).resolve().parents[2] / "data_sources" / "ai_generated" / "FLUX_TopUp"
TARGET_COUNT = 2500

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset : {DATASET_NAME}")
print(f"Output  : {TARGET_DIR}")
print(f"Target  : {TARGET_COUNT} images")
print()

existing = sum(1 for f in TARGET_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png"})
print(f"Existing: {existing} images")
if existing >= TARGET_COUNT:
    print(f"Already have {existing} — nothing to do.")
    raise SystemExit(0)

ds = load_dataset(DATASET_NAME, split="train", streaming=True)

saved = existing
skipped = 0
for i, example in enumerate(ds):
    if saved >= TARGET_COUNT:
        break
    try:
        img = example["image"]
        if img.mode not in ("RGB", "RGBA"):
            skipped += 1
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.width < 256 or img.height < 256:
            skipped += 1
            continue
        out_path = TARGET_DIR / f"flux_topup_{saved:06d}.jpg"
        if out_path.exists():
            saved += 1
            continue
        img.save(out_path, "JPEG", quality=95)
        saved += 1
        if saved % 100 == 0:
            print(f"  {saved}/{TARGET_COUNT} saved (scanned {i+1}, skipped {skipped})")
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"  Warning: skipped example {i}: {e}")

print(f"\nDone. {saved} images in {TARGET_DIR}")
print(f"Skipped: {skipped}")
print("\nNext step:")
print("  cd dataset_builder && python main.py --config config/ai_generated_flux_topup_config.yaml")
