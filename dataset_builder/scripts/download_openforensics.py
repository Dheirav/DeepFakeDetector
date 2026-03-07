#!/usr/bin/env python3
"""
Download OpenForensics dataset from Zenodo.
https://zenodo.org/records/5528418

OpenForensics is a large-scale face forgery dataset (ICCV 2021), covering
multiple GAN-based face synthesis methods. It is a direct substitute for
ForgeryNet for face manipulation coverage.

License: CC BY 4.0 (no login or application required)

Strategy:
  - We only need 5,000 images, so we download the smallest parts first.
  - Val.zip  (3.1 GB)  →  ~10,000 face images  →  enough for 5,000 target
  - Train_part_5.zip (2.0 GB) downloaded as fallback if Val alone is not enough.

Usage:
    python download_openforensics.py --output-path ../../data_sources/ai_edited/OpenForensics
    python download_openforensics.py --output-path ../../data_sources/ai_edited/OpenForensics --include-train-part5
"""

import argparse
import os
import sys
import zipfile
import urllib.request
import time
from pathlib import Path

ZENODO_BASE = "https://zenodo.org/records/5528418/files"

FILES = {
    "Val.zip": {
        "url": f"{ZENODO_BASE}/Val.zip",
        "size_gb": 3.1,
        "md5": "80e7f692665d1758ac9ce581196e9d8d",
        "description": "Validation set (~10K images, 3.1 GB) — primary download",
    },
    "Train_part_5.zip": {
        "url": f"{ZENODO_BASE}/Train_part_5.zip",
        "size_gb": 2.0,
        "md5": "4d60a35c3956f36876c815a73e3dce39",
        "description": "Train part 5 (~7K images, 2.0 GB) — fallback if Val not enough",
    },
}


class DownloadProgress:
    def __init__(self, filename: str, total: int):
        self.filename = filename
        self.total = total
        self.downloaded = 0
        self.start = time.time()

    def __call__(self, block_count: int, block_size: int, total_size: int):
        if total_size > 0:
            self.total = total_size
        self.downloaded = block_count * block_size
        pct = min(100.0, self.downloaded / self.total * 100) if self.total else 0
        elapsed = time.time() - self.start
        speed = self.downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {self.downloaded/1024/1024:.1f} MB  {speed:.1f} MB/s"
        )
        sys.stdout.flush()


def download_file(url: str, dest_path: Path, description: str) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  Already downloaded: {dest_path.name} — skipping.")
        return True

    print(f"\nDownloading {dest_path.name}  ({description})")
    print(f"  URL: {url}")
    try:
        progress = DownloadProgress(dest_path.name, 0)
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        print()  # newline after progress bar
        print(f"  Saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  ERROR downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> int:
    print(f"\nExtracting {zip_path.name} → {output_dir} ...")
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.infolist()
            if not m.filename.endswith("/")
            and any(m.filename.lower().endswith(ext)
                    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        ]
        total = len(members)
        for i, member in enumerate(members, 1):
            # Flatten structure — put all images directly into output_dir
            fname = Path(member.filename).name
            dest = output_dir / fname
            if dest.exists():
                continue
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1
            if i % 500 == 0 or i == total:
                sys.stdout.write(f"\r  Extracted {i}/{total} images ...")
                sys.stdout.flush()
    print(f"\n  Done. {extracted} new images extracted to {output_dir}")
    return extracted


def count_images(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download OpenForensics dataset (face forgery, ICCV 2021)"
    )
    parser.add_argument(
        "--output-path",
        default="../../data_sources/ai_edited/OpenForensics",
        help="Destination directory for extracted images",
    )
    parser.add_argument(
        "--include-train-part5",
        action="store_true",
        help="Also download Train_part_5.zip (2.0 GB) as a supplemental source",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded .zip files after extraction (default: delete them)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=5000,
        help="Stop downloading once this many images are in output-path (default: 5000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OpenForensics Downloader")
    print(f"  Output directory : {output_dir}")
    print(f"  Image target     : {args.target}")
    print("=" * 60)

    # Check how many images already exist
    existing = count_images(output_dir)
    print(f"\n  Existing images in output dir: {existing}")
    if existing >= args.target:
        print(f"  Already have {existing} images — nothing to do.")
        return

    tmp_dir = output_dir / "_zips"
    tmp_dir.mkdir(exist_ok=True)

    to_download = ["Val.zip"]
    if args.include_train_part5:
        to_download.append("Train_part_5.zip")

    for filename in to_download:
        info = FILES[filename]
        zip_path = tmp_dir / filename

        ok = download_file(info["url"], zip_path, info["description"])
        if not ok:
            print(f"  Skipping extraction of {filename} due to download failure.")
            continue

        extracted = extract_zip(zip_path, output_dir)

        if not args.keep_zips:
            zip_path.unlink()
            print(f"  Deleted zip: {filename}")

        current = count_images(output_dir)
        print(f"  Total images now: {current}")
        if current >= args.target:
            print(f"  Reached target of {args.target} — stopping early.")
            break

    # Clean up empty tmp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    final = count_images(output_dir)
    print("\n" + "=" * 60)
    print(f"  Download complete.  {final} images in {output_dir}")
    if final < args.target:
        print(
            f"  WARNING: Only {final} images available (target {args.target}).\n"
            f"  Re-run with --include-train-part5 to fetch Train_part_5.zip (2.0 GB)."
        )
    print("=" * 60)
    print("\nNext step — run the pipeline:")
    print(
        "  cd dataset_builder && python main.py "
        "--config config/ai_edited_openforensics_config.yaml"
    )


if __name__ == "__main__":
    main()
