import os
import csv
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import time

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def compute_md5(file_path, logger, chunk_size=8192):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"MD5 hash failed for {file_path}: {e}")
        return None


def index_dataset(source_dirs, output_csv, logger, dry_run=False, project_root=None, class_map=None):
    """
    Recursively scan source_dirs, index image files, extract metadata, and write to output_csv.
    class_map: dict mapping source_dir to class_label
    """
    start_time = time.time()
    if project_root is None:
        project_root = Path.cwd()
    header = [
        "path", "dataset_source", "class_label", "width", "height", "aspect_ratio",
        "format", "file_size_bytes", "md5_hash"
    ]
    output_csv = Path(output_csv)
    if not dry_run:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(output_csv, "w", newline="")
        writer = csv.DictWriter(out_f, fieldnames=header)
        writer.writeheader()
    else:
        writer = None

    total_files = 0
    indexed = 0
    skipped = 0

    for source_dir in source_dirs:
        source_path = Path(source_dir)
        # Normalize dataset_source for portability (relative to project root)
        dataset_source = str(source_path.relative_to(project_root))
        # Handle missing class_map keys explicitly
        if class_map:
            if source_dir in class_map:
                class_label = class_map[source_dir]
            else:
                logger.error(f"class_map missing key for source_dir: {source_dir}")
                class_label = "unknown"
        else:
            class_label = dataset_source
        logger.info(f"Indexing {source_dir} as class '{class_label}' and source '{dataset_source}'")
        # Build flat list of image files
        image_files = []
        for root, _, files in os.walk(source_path):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_files.append(Path(root) / fname)
        total_files += len(image_files)
        for fpath in tqdm(image_files, desc=f"Scanning {dataset_source}"):
            rel_path = str(fpath.relative_to(project_root))
            try:
                with Image.open(fpath) as img:
                    img.verify()  # Check for corruption
                with Image.open(fpath) as img:
                    width, height = img.size
                    aspect_ratio = round(width / height, 4) if height else None
                    fmt = img.format
            except (UnidentifiedImageError, OSError, ValueError) as e:
                logger.error(f"Corrupted or unreadable image: {rel_path} ({e})")
                skipped += 1
                continue
            file_size = fpath.stat().st_size if fpath.exists() else None
            md5 = compute_md5(fpath, logger)
            row = {
                "path": rel_path,
                "dataset_source": dataset_source,
                "class_label": class_label,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "format": fmt,
                "file_size_bytes": file_size,
                "md5_hash": md5
            }
            if not dry_run:
                writer.writerow(row)
            else:
                logger.info(f"[DRY RUN] Indexed: {row}")
            indexed += 1
    if not dry_run:
        out_f.close()
    elapsed = time.time() - start_time
    logger.info(f"Indexing complete. Output: {output_csv if not dry_run else '[DRY RUN]'}")
    logger.info(f"Summary: Total scanned: {total_files}, Indexed: {indexed}, Skipped: {skipped}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
