#!/usr/bin/env python3
"""
Download extra StableDiffusion images from DiffusionDB parts 4-6
to fill the remaining ai_generated gap (~641 images needed).

Output : data_sources/ai_generated/SD_TopUp2/
"""

import io
import zipfile
from pathlib import Path
import requests
from PIL import Image

BASE_URL = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images"
TARGET_DIR = (
    Path(__file__).resolve().parents[2]
    / "data_sources" / "ai_generated" / "SD_TopUp2"
)
TARGET_COUNT = 1000   # buffer (~1.5x the 641 we need)
FIRST_PART = 4
LAST_PART  = 6

TARGET_DIR.mkdir(parents=True, exist_ok=True)

existing = sum(1 for f in TARGET_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png"})
print(f"Output : {TARGET_DIR}")
print(f"Target : {TARGET_COUNT} images  (existing: {existing})")
if existing >= TARGET_COUNT:
    print("Already have enough — nothing to do.")
    raise SystemExit(0)

saved, skipped = existing, 0

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
        print(f"  Failed: {e} — skipping.")
        continue
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if saved >= TARGET_COUNT:
                    break
                if not name.lower().endswith(".png"):
                    continue
                try:
                    with zf.open(name) as f:
                        img = Image.open(f); img.load()
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.width < 256 or img.height < 256:
                        skipped += 1; continue
                    stem = Path(name).stem
                    img.save(TARGET_DIR / f"sd2_{part_str}_{stem}.jpg", "JPEG", quality=95)
                    saved += 1
                    if saved % 100 == 0:
                        print(f"  {saved}/{TARGET_COUNT} saved")
                except Exception:
                    skipped += 1
    except zipfile.BadZipFile as e:
        print(f"  Bad ZIP: {e} — skipping.")

print(f"\nDone. {saved} images in {TARGET_DIR}  (skipped {skipped})")
print("\nNext step:")
print("  cd dataset_builder && python3 main.py --config config/ai_generated_sd_topup2_config.yaml")
