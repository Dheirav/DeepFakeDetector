#!/usr/bin/env python3
"""Utility to merge per-dataset `export_index.csv` and `sampled_index.csv` into master manifests.

Usage:
  python3 tools/merge_exports.py --artifacts-dirs path/to/artifacts1 path/to/artifacts2 \
    --out-dir dataset_builder/merged_exports --export-root final_exports --hardlink

This script deduplicates rows by (md5_hash, sha256) and copies/hardlinks exported files into the
specified `--export-root`. It writes `master_export_index.csv` and `master_sampled_index.csv`.
"""
import argparse
import csv
from pathlib import Path
import shutil
import os


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def write_csv_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def merge_exports(artifacts_dirs, out_dir: Path, export_root: Path, hardlink=False):
    seen = set()
    master_export_rows = []
    master_sampled_rows = []
    export_fieldnames = None
    sampled_fieldnames = None

    for a in artifacts_dirs:
        a = Path(a)
        export_csv = a / 'export_index.csv'
        sampled_csv = a / 'sampled_index.csv'
        # Merge sampled rows (smaller)
        srows = read_csv_rows(sampled_csv)
        if srows:
            if sampled_fieldnames is None:
                sampled_fieldnames = list(srows[0].keys())
            for r in srows:
                key = (r.get('md5_hash',''), r.get('sha256',''))
                if key in seen:
                    continue
                seen.add(key)
                master_sampled_rows.append(r)
        # Merge export rows and copy files
        erows = read_csv_rows(export_csv)
        if erows:
            if export_fieldnames is None:
                export_fieldnames = list(erows[0].keys())
            for r in erows:
                key = (r.get('md5_hash',''), r.get('sha256',''))
                if key in seen:
                    continue
                seen.add(key)
                master_export_rows.append(r)
                src = Path(r.get('export_path',''))
                if not src.exists():
                    # try relative to artifacts parent (export root used by pipeline)
                    candidate = a.parent / src.name
                    if candidate.exists():
                        src = candidate
                if src.exists():
                    dest_root = export_root
                    dest_root.mkdir(parents=True, exist_ok=True)
                    dest = dest_root / src.name
                    try:
                        if hardlink:
                            os.link(str(src), str(dest))
                        else:
                            shutil.copy2(str(src), str(dest))
                    except Exception:
                        try:
                            shutil.copy2(str(src), str(dest))
                        except Exception as e:
                            print(f"Warning: failed to copy {src} -> {dest}: {e}")

    # write outputs
    if export_fieldnames is None and master_export_rows:
        export_fieldnames = list(master_export_rows[0].keys())
    if sampled_fieldnames is None and master_sampled_rows:
        sampled_fieldnames = list(master_sampled_rows[0].keys())

    write_csv_rows(out_dir / 'master_export_index.csv', master_export_rows, export_fieldnames or [])
    write_csv_rows(out_dir / 'master_sampled_index.csv', master_sampled_rows, sampled_fieldnames or [])


def main():
    p = argparse.ArgumentParser(description="Merge per-dataset export/sample manifests into a master manifest")
    p.add_argument('--artifacts-dirs', nargs='+', required=True, help='List of per-dataset artifacts directories')
    p.add_argument('--out-dir', default='dataset_builder/merged_exports', help='Directory to write master manifests')
    p.add_argument('--export-root', default='final_exports', help='Where to place merged exported image files')
    p.add_argument('--hardlink', action='store_true', help='Try to hardlink exported files instead of copying')
    args = p.parse_args()

    artifacts = args.artifacts_dirs
    out_dir = Path(args.out_dir)
    export_root = Path(args.export_root)

    merge_exports(artifacts, out_dir, export_root, hardlink=args.hardlink)
    print(f"Wrote master manifests to {out_dir}; exported files are in {export_root}")


if __name__ == '__main__':
    main()
