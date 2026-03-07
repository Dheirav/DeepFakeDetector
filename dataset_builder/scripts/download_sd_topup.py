#!/usr/bin/env python3
"""
Download StableDiffusion images from DiffusionDB (top-up for ai_generated class).

Source  : poloclub/diffusiondb  (2M images, fully public, no login required)
          Downloads ZIP files directly from HuggingFace CDN — no loading script.
Output  : data_sources/ai_generated/StableDiffusion_TopUp/
Target  : 2,500 images (buffer for the 2,000 pipeline target after filtering)

DiffusionDB contains Stable Diffusion 1.x generated images with diverse prompts.
Each part ZIP (diffusiondb-2m-part-XXXX.zip) contains exactly 1,000 PNG images.
We download enough parts to reach the target and then stop.

Usage:
    cd dataset_builder/scripts
    python download_sd_topup.py
"""

import io
import shutil
import zipfile
from pathlib import Path

import requests
from PIL import Image

BASE_URL = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images"
TARGET_DIR = (
    Path(__file__).resolve().parents[2]
    / "data_sources" / "ai_generated" / "StableDiffusion_TopUp"
)
TARGET_COUNT = 2500
# Parts 000001-002000 each contain ~1000 images; stored as part-000001.zip, part-000002.zip, ...
FIRST_PART = 1
LAST_PART  = 10    # use first 10 parts (~10k images available), stop early at target

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset : poloclub/diffusiondb  (direct ZIP download)")
print(f"Output  : {TARGET_DIR}")
print(f"Target  : {TARGET_COUNT} images")
print()

existing = sum(1 for f in TARGET_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png", ".webp"})
print(f"Existing: {existing} images")
if existing >= TARGET_COUNT:
    print(f"Already have {existing} — nothing to do.")
    raise SystemExit(0)

saved = existing
skipped = 0

for part in range(FIRST_PART, LAST_PART + 1):
    if saved >= TARGET_COUNT:
        break
    part_str = f"{part:06d}"
    zip_name = f"part-{part_str}.zip"
    url = f"{BASE_URL}/{zip_name}"
    print(f"\nDownloading {zip_name} ...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        raw = b"".join(resp.iter_content(chunk_size=1 << 20))
    except Exception as e:
        print(f"  Failed to download {zip_name}: {e} — skipping.")
        continue

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".png")]
            for name in members:
                if saved >= TARGET_COUNT:
                    break
                try:
                    with zf.open(name) as f:
                        img = Image.open(f)
                        img.load()
                    if img.mode not in ("RGB", "RGBA", "L"):
                        skipped += 1
                        continue
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.width < 256 or img.height < 256:
                        skipped += 1
                        continue
                    stem = Path(name).stem
                    out_path = TARGET_DIR / f"diffusiondb_{part_str}_{stem}.jpg"
                    img.save(out_path, "JPEG", quality=95)
                    saved += 1
                    if saved % 100 == 0:
                        print(f"  {saved}/{TARGET_COUNT} saved (skipped {skipped})")
                except Exception:
                    skipped += 1
    except zipfile.BadZipFile as e:
        print(f"  Bad ZIP {zip_name}: {e} — skipping.")
        continue

print(f"\nDone. {saved} images in {TARGET_DIR}")
print(f"Skipped: {skipped}")
print("\nNext step:")
print("  cd dataset_builder && python main.py --config config/ai_generated_sd_topup_config.yaml")
