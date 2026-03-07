#!/usr/bin/env python3
"""
Download COCO test2017 images for the real-image class.

No login, no account required — direct HTTP download from COCO servers.

  URL  : http://images.cocodataset.org/zips/test2017.zip
  Size : ~6.3 GB
  Images: 40,670 real photographs (diverse scenes, objects, people, nature)

These are the official COCO test images — completely separate from the
train2017/val2017 sets already used in real_coco_config.yaml, so there
is zero overlap. The pipeline deduplicator will verify this.

Usage:
    cd dataset_builder/scripts
    python download_coco_test.py
    python download_coco_test.py --output-path ../../data_sources/real/COCO_Test
"""

import argparse
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

COCO_TEST_URL = "http://images.cocodataset.org/zips/test2017.zip"
COCO_TEST_MD5 = "77ad2c53ac5c0b5e1bc3048e0ed3f426"
ZIP_SIZE_GB = 6.3


class DownloadProgress:
    def __init__(self):
        self.start = time.time()

    def __call__(self, block_count: int, block_size: int, total_size: int):
        downloaded = block_count * block_size
        pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0
        elapsed = time.time() - self.start
        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        eta = (total_size - downloaded) / (downloaded / elapsed) if downloaded > 0 else 0
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  "
            f"{downloaded/1024/1024:.0f}/{total_size/1024/1024:.0f} MB  "
            f"{speed:.1f} MB/s  ETA {eta:.0f}s"
        )
        sys.stdout.flush()


def count_images(directory: Path) -> int:
    return sum(1 for f in directory.iterdir()
               if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download COCO test2017 real images (no login required)"
    )
    parser.add_argument(
        "--output-path",
        default="../../data_sources/real/COCO_Test",
        help="Destination directory for extracted images",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COCO test2017 Downloader (Real Photos)")
    print(f"  Output : {output_dir}")
    print(f"  Size   : ~{ZIP_SIZE_GB} GB download, ~40,670 images")
    print(f"  Source : {COCO_TEST_URL}")
    print("=" * 60)

    existing = count_images(output_dir)
    if existing > 0:
        print(f"\n  Found {existing} existing images.")
        if existing >= 40000:
            print("  Already fully downloaded — nothing to do.")
            return

    zip_path = output_dir / "test2017.zip"

    # Download
    if not zip_path.exists():
        print(f"\nDownloading test2017.zip (~{ZIP_SIZE_GB} GB) ...")
        try:
            urllib.request.urlretrieve(COCO_TEST_URL, zip_path, reporthook=DownloadProgress())
            print()  # newline after progress bar
            print(f"  Saved: {zip_path}")
        except KeyboardInterrupt:
            print("\n  Interrupted. Partial zip deleted.")
            if zip_path.exists():
                zip_path.unlink()
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            if zip_path.exists():
                zip_path.unlink()
            sys.exit(1)
    else:
        print(f"\n  Zip already exists: {zip_path} — skipping download.")

    # Extract
    print(f"\nExtracting images to {output_dir} ...")
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.infolist()
            if not m.filename.endswith("/")
            and m.filename.lower().endswith(".jpg")
        ]
        total = len(members)
        print(f"  Total images in zip: {total}")
        for i, member in enumerate(members, 1):
            fname = Path(member.filename).name
            dest = output_dir / fname
            if dest.exists():
                continue
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1
            if i % 2000 == 0 or i == total:
                sys.stdout.write(f"\r  Extracted {i}/{total} ...")
                sys.stdout.flush()
    print(f"\n  Done. {extracted} new images extracted.")

    if not args.keep_zip:
        zip_path.unlink()
        print(f"  Deleted zip to free space.")

    final = count_images(output_dir)
    print(f"\n{'=' * 60}")
    print(f"  Complete. {final} images in {output_dir}")
    print(f"{'=' * 60}")
    print("\nNext step — run the pipeline:")
    print("  cd dataset_builder")
    print("  python main.py --config config/real_coco_test_config.yaml")


if __name__ == "__main__":
    main()
