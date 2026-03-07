#!/usr/bin/env python3
"""
Download Places365 validation images for the real-image class.

Dataset : Places365-Standard validation set (val_256.tar)
URL     : http://data.csail.mit.edu/places/places365/val_256.tar
Size    : ~2.2 GB
Images  : 36,500 images × 365 scene categories (50 per category)
License : MIT / Free for research use. No login required.

Why Places365?
  Scene photographs spanning 365 semantic indoor/outdoor categories:
  airports, bedrooms, beaches, forests, kitchens, streets, etc.
  Completely different from COCO/FFHQ/OpenImages — maximises real-class diversity.

Usage:
    cd dataset_builder/scripts
    python download_places365.py
    python download_places365.py --output-path ../../data_sources/real/Places365 --target 3500
"""

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

PLACES365_VAL_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
APPROX_SIZE_GB = 2.2
APPROX_TOTAL_IMAGES = 36500


class DownloadProgress:
    def __init__(self):
        self.start = time.time()

    def __call__(self, block_count, block_size, total_size):
        downloaded = block_count * block_size
        pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0
        elapsed = time.time() - self.start
        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        bar = "=" * int(40 * pct / 100) + "-" * (40 - int(40 * pct / 100))
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {downloaded/1024/1024:.0f} MB  {speed:.1f} MB/s"
        )
        sys.stdout.flush()


def count_images(directory: Path) -> int:
    return sum(1 for f in directory.iterdir()
               if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Places365 validation images (real photos, no login)"
    )
    parser.add_argument(
        "--output-path",
        default="../../data_sources/real/Places365",
        help="Destination directory for extracted images",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3500,
        help="Target number of images to extract (default 3500 — buffer for pipeline filtering)",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep the downloaded .tar file after extraction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Places365 Validation Set Downloader (Real Scene Photos)")
    print(f"  Output : {output_dir}")
    print(f"  Source : {PLACES365_VAL_URL}")
    print(f"  Size   : ~{APPROX_SIZE_GB} GB  ({APPROX_TOTAL_IMAGES:,} images, 365 scene categories)")
    print(f"  Target : extract first {args.target} images then stop")
    print("=" * 60)

    existing = count_images(output_dir)
    print(f"\n  Existing images: {existing}")
    if existing >= args.target:
        print(f"  Already have {existing} — nothing to do.")
        return

    tar_path = output_dir / "val_256.tar"

    # Download
    if not tar_path.exists():
        print(f"\nDownloading val_256.tar (~{APPROX_SIZE_GB} GB) ...")
        try:
            urllib.request.urlretrieve(PLACES365_VAL_URL, tar_path, reporthook=DownloadProgress())
            print()
            print(f"  Saved: {tar_path}")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            if tar_path.exists():
                tar_path.unlink()
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            if tar_path.exists():
                tar_path.unlink()
            sys.exit(1)
    else:
        print(f"\n  Tar already exists — skipping download.")

    # Extract up to target
    print(f"\nExtracting up to {args.target} images to {output_dir} ...")
    extracted = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if extracted >= args.target:
                break
            fname = Path(member.name).name
            dest = output_dir / fname
            if dest.exists():
                extracted += 1
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            dest.write_bytes(f.read())
            extracted += 1
            if extracted % 500 == 0:
                sys.stdout.write(f"\r  Extracted {extracted}/{args.target} ...")
                sys.stdout.flush()

    print(f"\n  Done. {extracted} images extracted.")

    if not args.keep_tar:
        tar_path.unlink()
        print(f"  Deleted tar to free space.")

    final = count_images(output_dir)
    print(f"\n{'=' * 60}")
    print(f"  Complete. {final} images in {output_dir}")
    print(f"{'=' * 60}")
    print("\nNext step:")
    print("  cd dataset_builder && python main.py --config config/real_places365_config.yaml")


if __name__ == "__main__":
    main()
