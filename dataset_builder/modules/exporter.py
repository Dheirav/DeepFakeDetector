
import os
import csv
import json
import hashlib
import shutil
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm

def safe_copy_file(src, dst, overwrite=False, dry_run=False):
    dst = Path(dst)
    if dst.exists() and not overwrite:
        base, ext = os.path.splitext(dst.name)
        parent = dst.parent
        i = 1
        while (parent / f"{base}_{i}{ext}").exists():
            i += 1
        dst = parent / f"{base}_{i}{ext}"
    final_dst = dst
    if not dry_run:
        final_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, final_dst)
        except Exception as e:
            return None, str(e)
    return str(final_dst), None

def compute_sha256(file_path, chunk_size=65536):
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256.update(chunk)
        return sha256.hexdigest(), None
    except Exception as e:
        return None, str(e)

def summarize_stats(rows):
    stats = {
        'total': len(rows),
        'splits': Counter(),
        'classes': Counter(),
        'sources': Counter()
    }
    for row in rows:
        stats['splits'][row['split']] += 1
        stats['classes'][row['class_label']] += 1
        stats['sources'][row['dataset_source']] += 1
    return stats

def generate_manifest(export_path, rows, config, stats):
    # Only store minimal config snapshot (no secrets)
    config_snapshot = {k: v for k, v in config.items() if k not in {'secrets', 'api_keys', 'password', 'token'}}
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'dataset_version': config.get('dataset_version', 'unknown'),
        'config_snapshot': config_snapshot,
        'image_counts_per_split': dict(stats['splits']),
        'class_distribution': dict(stats['classes']),
        'source_distribution': dict(stats['sources'])
    }
    with open(os.path.join(export_path, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

def write_readme(export_path, stats, config):
    lines = [
        f"# Exported Deepfake Dataset",
        f"",
        f"**Version:** {config.get('dataset_version', 'unknown')}",
        f"**Exported:** {datetime.now().isoformat()}",
        f"",
        f"## Structure",
        f"- Root: `{export_path}`",
        f"- Splits: {', '.join(stats['splits'].keys())}",
        f"- Classes: {', '.join(stats['classes'].keys())}",
        f"",
        f"## Counts",
        f"- Total images: {stats['total']}",
    ]
    for split, count in stats['splits'].items():
        lines.append(f"  - {split}: {count}")
    lines.append("")
    lines.append("## Class Distribution")
    for cls, count in stats['classes'].items():
        lines.append(f"  - {cls}: {count}")
    lines.append("")
    lines.append("## Source Distribution")
    for src, count in stats['sources'].items():
        lines.append(f"  - {src}: {count}")
    lines.append("")
    lines.append("## Notes\n- See manifest.json for full config and stats.")
    with open(os.path.join(export_path, 'README.md'), 'w') as f:
        f.write('\n'.join(lines))

def export_dataset(split_index_csv, export_root, config, logger, dry_run=False):
    export_root = Path(export_root)
    project_root = Path(config.get('project_root', '.'))
    overwrite = config.get('overwrite_existing', False)
    verify_hashes = config.get('verify_hashes', True)
    max_images_per_split = config.get('max_images_per_split')
    random_seed = config.get('random_seed', 42)
    strict_mode = config.get('strict_mode', False)
    # Read split_index.csv
    with open(split_index_csv, 'r', newline='') as f:
        reader = list(csv.DictReader(f))
    if not reader:
        logger.warning(f"No rows in {split_index_csv}. Nothing to export.")
        return
    # Deterministic ordering: shuffle first, then sort
    rng = random.Random(random_seed)
    rng.shuffle(reader)
    reader = sorted(reader, key=lambda r: (
        r.get('split', ''), r.get('class_label', ''), r.get('dataset_source', ''), r.get('quality_score', ''), r.get('path', '')))
    # Apply max_images_per_split if set
    split_counts = defaultdict(int)
    filtered = []
    skipped_due_to_cap = defaultdict(int)
    for row in reader:
        split = row.get('split', '')
        if max_images_per_split and split_counts[split] >= max_images_per_split:
            skipped_due_to_cap[split] += 1
            continue
        split_counts[split] += 1
        filtered.append(row)
    if max_images_per_split:
        for split, count in skipped_due_to_cap.items():
            logger.info(f"Skipped {count} images in split '{split}' due to max_images_per_split cap")
    reader = filtered
    stats = summarize_stats(reader)
    logger.info(f"Exporting {len(reader)} images to {export_root}")
    # Prepare outputs
    export_index_path = export_root / 'export_index.csv'
    checksums_path = export_root / 'checksums.csv'
    export_rows = []
    checksums = []
    missing_files = 0
    copied = 0
    copied_per_split = defaultdict(int)
    logger.info("Beginning file export...")
    for row in tqdm(reader, desc="Exporting files", unit="file"):
        # Always copy row to avoid in-place modification
        row_out = dict(row)
        split = row_out.get('split', '')
        class_label = row_out.get('class_label', '')
        rel_path = row_out.get('path', '')
        if not split or not class_label or not rel_path:
            logger.warning(f"Row missing split/class_label/path: {row_out}")
            continue
        src_path = (project_root / rel_path).resolve()
        rel_dst = Path(split) / class_label / Path(rel_path).name
        abs_dst = export_root / rel_dst
        if not src_path.exists():
            logger.warning(f"Missing file: {src_path}")
            missing_files += 1
            if dry_run:
                # Simulate export path for dry_run
                row_out['export_path'] = str(abs_dst)
                export_rows.append(row_out)
            continue
        export_path, err = safe_copy_file(src_path, abs_dst, overwrite, dry_run)
        if err:
            logger.error(f"Failed to copy {src_path} to {abs_dst}: {err}")
            continue
        copied += 1
        copied_per_split[split] += 1
        row_out['export_path'] = str(export_path)
        export_rows.append(row_out)
        if not dry_run and verify_hashes and os.path.exists(export_path):
            sha256, err = compute_sha256(export_path)
            if err:
                logger.error(f"Checksum failed for {export_path}: {err}")
            checksums.append({'export_path': export_path, 'sha256': sha256 or '', 'error': err or ''})
        elif not dry_run and verify_hashes:
            checksums.append({'export_path': export_path, 'sha256': '', 'error': 'File missing for hash'})
    # Strict mode: fail fast if missing files
    if strict_mode and missing_files > 0:
        logger.error(f"Strict mode enabled: {missing_files} files missing. Aborting export.")
        raise RuntimeError(f"Strict mode: {missing_files} files missing. Aborting export.")
    # Write export_index.csv
    if not dry_run and export_rows:
        with open(export_index_path, 'w', newline='') as f:
            fieldnames = list(export_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in export_rows:
                writer.writerow(row)
    elif not dry_run and not export_rows:
        logger.warning("No export rows to write to export_index.csv.")
    # Write checksums.csv
    if not dry_run and checksums:
        with open(checksums_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['export_path', 'sha256', 'error'])
            writer.writeheader()
            for row in checksums:
                writer.writerow(row)
    elif not dry_run and not checksums:
        logger.warning("No checksums to write.")
    # Manifest and README
    if not dry_run:
        generate_manifest(export_root, export_rows, config, stats)
        write_readme(export_root, stats, config)
    for split, count in copied_per_split.items():
        logger.info(f"Copied {count} files to split '{split}'")
    logger.info(f"Final class distribution: {dict(stats['classes'])}")
    logger.info(f"Final split distribution: {dict(stats['splits'])}")
    if copied + missing_files > 0:
        miss_rate = 100.0 * missing_files / (copied + missing_files)
        logger.info(f"Missing file rate: {miss_rate:.2f}% ({missing_files}/{copied + missing_files})")
    logger.info(f"Copied {copied} files. Missing: {missing_files}. Export root: {export_root}")
    logger.info(f"Export complete. Manifest: {export_root / 'manifest.json'}")
