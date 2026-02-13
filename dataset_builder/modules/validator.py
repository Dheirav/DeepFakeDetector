import csv
import os
from pathlib import Path
import cv2
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def normalized_quality_score(quality_flags):
    """Compute normalized quality score from flags"""
    if not quality_flags or quality_flags == ['ok']:
        return 1.0
    # Weighted penalties
    penalties = {
        'corrupt': 1.0,
        'low_resolution': 0.3,
        'blurry': 0.4,
        'aspect_ratio_extreme': 0.2,
        'compressed': 0.1,
        'opencv_read_failed': 0.5
    }
    total_penalty = sum(penalties.get(flag, 0.2) for flag in quality_flags)
    return max(0.0, 1.0 - min(total_penalty, 1.0))


def check_resolution(row, min_width, min_height):
    try:
        width = int(row.get('width', 0))
        height = int(row.get('height', 0))
        return width >= min_width and height >= min_height
    except Exception:
        return False

def check_aspect_ratio(row, aspect_ratio_min, aspect_ratio_max):
    try:
        aspect_ratio = float(row.get('aspect_ratio', 0))
        return aspect_ratio_min <= aspect_ratio <= aspect_ratio_max
    except Exception:
        return False

def compute_blur_score(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        return blur_score
    except Exception:
        return None

def check_compression_artifact(row, config):
    # Optional: Heuristic based on file size, format, etc.
    # Example: flag JPEGs with very small file size per pixel
    try:
        if row.get('format', '').lower() == 'jpeg':
            file_size = int(row.get('file_size_bytes', 0))
            width = int(row.get('width', 0))
            height = int(row.get('height', 0))
            if width > 0 and height > 0:
                pixels = width * height
                size_per_pixel = file_size / pixels if pixels else 0
                threshold = config.get('compression_size_per_pixel', 0.15)
                return size_per_pixel < threshold
        return False
    except Exception:
        return False

def validate_images(index_csv, output_csv, config, logger, dry_run=False):
    min_width = config['image_rules']['min_width']
    min_height = config['image_rules']['min_height']
    blur_threshold = config['image_rules']['blur_threshold']
    aspect_ratio_min = config['image_rules'].get('aspect_ratio_min', 0.5)
    aspect_ratio_max = config['image_rules'].get('aspect_ratio_max', 2.0)
    compression_heuristic = config['image_rules'].get('compression_heuristic', False)
    project_root = Path(config.get('project_root', os.getcwd()))

    input_path = Path(index_csv)
    output_path = Path(output_csv)
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(output_path, 'w', newline='')
        writer = None
    else:
        out_f = None
        writer = None

    summary = {}
    total = 0
    kept = 0
    flagged = 0
    per_class = {}
    per_source = {}
    per_flag = {'blurry': 0, 'low_resolution': 0, 'corrupt': 0, 'compressed': 0, 'aspect_ratio_extreme': 0, 'opencv_read_failed': 0}

    with open(input_path, 'r', newline='') as in_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames + [
            'quality_flag', 'quality_score', 'blur_score', 'resolution_ok', 'aspect_ratio_ok'
        ]
        if not dry_run:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
        for row in tqdm(reader, desc='Validating images'):
            total += 1
            quality_flags = []
            blur_score = None
            resolution_ok = check_resolution(row, min_width, min_height)
            aspect_ratio_ok = check_aspect_ratio(row, aspect_ratio_min, aspect_ratio_max)
            image_path = row['path']
            abs_image_path = str((project_root / image_path).absolute())
            # Try to load image for blur check
            try:
                blur_score = compute_blur_score(abs_image_path)
                if blur_score is None:
                    quality_flags.append('opencv_read_failed')
                elif blur_score < blur_threshold:
                    quality_flags.append('blurry')
            except Exception:
                quality_flags.append('opencv_read_failed')
            if not resolution_ok:
                quality_flags.append('low_resolution')
            if not aspect_ratio_ok:
                quality_flags.append('aspect_ratio_extreme')
            if compression_heuristic and check_compression_artifact(row, config['image_rules']):
                quality_flags.append('compressed')
            # Try to load with PIL for corruption check
            try:
                with Image.open(abs_image_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, ValueError):
                quality_flags = ['corrupt']
            quality_flag_str = '|'.join(quality_flags) if quality_flags else 'ok'
            quality_score = normalized_quality_score(quality_flags)
            row.update({
                'quality_flag': quality_flag_str,
                'quality_score': quality_score,
                'blur_score': blur_score if blur_score is not None else '',
                'resolution_ok': resolution_ok,
                'aspect_ratio_ok': aspect_ratio_ok
            })
            if not dry_run:
                writer.writerow(row)
            else:
                logger.info(f"[DRY RUN] Validated: {row['path']} -> {quality_flag_str}")
            # Stats
            class_label = row.get('class_label', 'unknown')
            dataset_source = row.get('dataset_source', 'unknown')
            per_class.setdefault(class_label, {'total': 0, 'flagged': 0})
            per_source.setdefault(dataset_source, {'total': 0, 'flagged': 0})
            per_class[class_label]['total'] += 1
            per_source[dataset_source]['total'] += 1
            if quality_flag_str != 'ok':
                flagged += 1
                per_class[class_label]['flagged'] += 1
                per_source[dataset_source]['flagged'] += 1
            else:
                kept += 1
            # Per-flag analytics
            for flag in quality_flags:
                if flag in per_flag:
                    per_flag[flag] += 1
    if not dry_run:
        out_f.close()
    logger.info(f"Validation complete. Output: {output_path if not dry_run else '[DRY RUN]'}")
    logger.info(f"Summary: Total: {total}, Kept: {kept}, Flagged: {flagged}")
    for cl, stats in per_class.items():
        pct = 100.0 * stats['flagged'] / stats['total'] if stats['total'] else 0.0
        logger.info(f"Class '{cl}': Total: {stats['total']}, Flagged: {stats['flagged']} ({pct:.2f}%)")
    for src, stats in per_source.items():
        pct = 100.0 * stats['flagged'] / stats['total'] if stats['total'] else 0.0
        logger.info(f"Source '{src}': Total: {stats['total']}, Flagged: {stats['flagged']} ({pct:.2f}%)")
    for flag, count in per_flag.items():
        pct = 100.0 * count / total if total else 0.0
        logger.info(f"Flag '{flag}': {count} ({pct:.2f}%)")
