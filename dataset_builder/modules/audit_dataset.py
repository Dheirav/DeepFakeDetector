
import os
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import time
import math
from tqdm import tqdm

def audit_dataset(split_index_csv, output_report_path, config, logger, dry_run=False):
    start_time = time.time()
    project_root = Path(config.get('project_root', '.')).resolve()
    export_root = Path(config.get('export_root')).resolve() if config.get('export_root') else None
    min_quality_score = config.get('min_quality_score', 0)
    valid_splits = set(config.get('split_ratios', {'train':0.8,'val':0.1,'test':0.1}).keys())
    known_sources = set(config.get('known_sources', []))
    required_fields = ['path','export_path','split','class_label','dataset_source','quality_score']
    max_missing_pct = config.get('max_missing_pct', 1.0)
    max_leakage_allowed = config.get('max_leakage_allowed', 0)
    max_malformed_pct = config.get('max_malformed_pct', 1.0)
    max_low_quality_pct = config.get('max_low_quality_pct', 10)
    quality_hist_bins = config.get('quality_hist_bins', [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    anomaly_cap = config.get('anomaly_cap', 1000)
    # Streaming-safe CSV scan and memory safety
    rows = []
    malformed_rows = []
    total_rows = 0
    with open(split_index_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            total_rows += 1
            # Only keep required fields in memory for valid rows
            if not all(field in row and row[field] not in [None, ''] for field in required_fields):
                if len(malformed_rows) < anomaly_cap:
                    malformed_rows.append({'row_idx': idx, 'issue': 'Malformed row'})
            else:
                filtered_row = {k: row[k] for k in required_fields}
                for h in ('md5_hash', 'sha256', 'width', 'height'):
                    if h in row:
                        filtered_row[h] = row[h]
                rows.append(filtered_row)
            if total_rows % 100000 == 0:
                logger.info(f"Scanned {total_rows} rows...")
    logger.info(f"Loaded {len(rows)} valid rows, {len(malformed_rows)} malformed rows (of {total_rows} total).")
    # File integrity checks
    missing_files = []
    zero_byte_files = []
    unreadable_files = []
    export_path_set = set()
    orig_path_set = set()
    duplicate_export_paths = set()
    duplicate_orig_paths = set()
    for idx, row in enumerate(rows):
        export_path = row['export_path']
        orig_path = row['path']
        # Path resolution: export_path relative to export_root unless absolute
        abs_export = None
        if Path(export_path).is_absolute():
            abs_export = Path(export_path)
        elif export_root:
            abs_export = (export_root / export_path).resolve()
        else:
            abs_export = Path(export_path).resolve()
        # orig_path always relative to project_root unless absolute
        abs_orig = Path(orig_path).resolve() if Path(orig_path).is_absolute() else (project_root / orig_path).resolve()
        # Path traversal safety: ensure abs_export is under export_root if export_root is set
        if export_root and not str(abs_export).startswith(str(export_root)):
            logger.warning(f"Export path {abs_export} is outside export_root {export_root} (row {idx})")
        if not str(abs_orig).startswith(str(project_root)):
            logger.warning(f"Original path {abs_orig} is outside project_root {project_root} (row {idx})")
        # File integrity checks
        if not abs_export.exists():
            missing_files.append(export_path)
        else:
            try:
                size = abs_export.stat().st_size
                if size == 0:
                    zero_byte_files.append(export_path)
            except Exception:
                unreadable_files.append(export_path)
        # Duplicates
        if export_path in export_path_set:
            duplicate_export_paths.add(export_path)
        else:
            export_path_set.add(export_path)
        if orig_path in orig_path_set:
            duplicate_orig_paths.add(orig_path)
        else:
            orig_path_set.add(orig_path)
    # Deterministic ordering for duplicates and file issues
    duplicate_export_paths = sorted(duplicate_export_paths)
    duplicate_orig_paths = sorted(duplicate_orig_paths)
    missing_files = sorted(missing_files)
    zero_byte_files = sorted(zero_byte_files)
    unreadable_files = sorted(unreadable_files)
    # Deterministic ordering for duplicates
    duplicate_export_paths = sorted(duplicate_export_paths)
    duplicate_orig_paths = sorted(duplicate_orig_paths)
    missing_files = sorted(missing_files)
    zero_byte_files = sorted(zero_byte_files)
    unreadable_files = sorted(unreadable_files)
    # Distribution analysis
    split_counts = Counter()
    class_counts = Counter()
    source_counts = Counter()
    class_split_matrix = defaultdict(Counter)
    source_split_matrix = defaultdict(Counter)
    for row in rows:
        split = row['split']
        cls = row['class_label']
        src = row['dataset_source']
        split_counts[split] += 1
        class_counts[cls] += 1
        source_counts[src] += 1
        class_split_matrix[split][cls] += 1
        source_split_matrix[split][src] += 1
    # Deterministic ordering for reporting
    split_counts = dict(sorted(split_counts.items()))
    class_counts = dict(sorted(class_counts.items()))
    source_counts = dict(sorted(source_counts.items()))
    class_split_matrix = {k: dict(sorted(v.items())) for k, v in sorted(class_split_matrix.items())}
    source_split_matrix = {k: dict(sorted(v.items())) for k, v in sorted(source_split_matrix.items())}
    # Imbalance warnings
    warnings = []
    total = sum(split_counts.values())
    for split, count in split_counts.items():
        ratio = count / total if total else 0
        if ratio < 0.05:
            warnings.append(f"Severe split imbalance: {split} has only {count} images.")
    for cls, count in class_counts.items():
        if count < 0.05 * total:
            warnings.append(f"Severe class imbalance: {cls} has only {count} images.")
    for src, count in source_counts.items():
        if count < 0.01 * total:
            warnings.append(f"Severe source imbalance: {src} has only {count} images.")
    warnings = sorted(warnings)
    # Leakage risk detection (critical: hash, soft: filename, origpath)
    hash_split_map = defaultdict(set)
    filename_split_map = defaultdict(set)
    origpath_split_map = defaultdict(set)
    hash_split_pairs = defaultdict(Counter)
    filename_split_pairs = defaultdict(Counter)
    origpath_split_pairs = defaultdict(Counter)
    for row in rows:
        split = row['split']
        # Critical: hash (md5/sha256)
        for hash_field in ('md5_hash', 'sha256'):
            if hash_field in row and row[hash_field]:
                hash_val = row[hash_field]
                hash_split_map[hash_val].add(split)
                for other in hash_split_map[hash_val]:
                    if other != split:
                        key = tuple(sorted([split, other]))
                        hash_split_pairs[hash_val][key] += 1
        # Soft: filename
        filename = Path(row['path']).name
        filename_split_map[filename].add(split)
        for other in filename_split_map[filename]:
            if other != split:
                key = tuple(sorted([split, other]))
                filename_split_pairs[filename][key] += 1
        # Soft: origpath
        origpath_split_map[row['path']].add(split)
        for other in origpath_split_map[row['path']]:
            if other != split:
                key = tuple(sorted([split, other]))
                origpath_split_pairs[row['path']][key] += 1
    leakage_hash = [h for h, splits in hash_split_map.items() if len(splits) > 1]
    leakage_filename = [f for f, splits in filename_split_map.items() if len(splits) > 1]
    leakage_origpath = [p for p, splits in origpath_split_map.items() if len(splits) > 1]
    # Breakdown by split pairs
    hash_leakage_pairs = {h: dict(sorted(hash_split_pairs[h].items())) for h in sorted(leakage_hash)}
    filename_leakage_pairs = {f: dict(sorted(filename_split_pairs[f].items())) for f in sorted(leakage_filename)}
    origpath_leakage_pairs = {p: dict(sorted(origpath_split_pairs[p].items())) for p in sorted(leakage_origpath)}
    leakage_hash = sorted(leakage_hash)
    leakage_filename = sorted(leakage_filename)
    leakage_origpath = sorted(leakage_origpath)
    if leakage_hash:
        warnings.append(f"Critical leakage: {len(leakage_hash)} hashes appear in multiple splits.")
    if leakage_filename:
        warnings.append(f"Soft leakage: {len(leakage_filename)} filenames appear in multiple splits.")
    if leakage_origpath:
        warnings.append(f"Soft leakage: {len(leakage_origpath)} original paths appear in multiple splits.")
    warnings = sorted(warnings)
    # Quality health evaluation (metadata only)
    quality_scores = []
    below_quality = 0
    for row in rows:
        try:
            q = float(row['quality_score'])
            quality_scores.append(q)
            if q < min_quality_score:
                below_quality += 1
        except Exception:
            continue
    # Histogram and percentiles
    hist_bins = quality_hist_bins
    hist_counts = [0] * (len(hist_bins) - 1)
    for q in quality_scores:
        for i in range(len(hist_bins) - 1):
            if hist_bins[i] <= q < hist_bins[i+1] or (i == len(hist_bins)-2 and q == hist_bins[-1]):
                hist_counts[i] += 1
                break
    percentiles = {}
    if quality_scores:
        sorted_q = sorted(quality_scores)
        for p in [10, 25, 50, 75, 90, 99]:
            k = int(len(sorted_q) * p / 100)
            percentiles[f'P{p}'] = sorted_q[min(k, len(sorted_q)-1)]
    quality_stats = {
        'min': min(quality_scores) if quality_scores else None,
        'max': max(quality_scores) if quality_scores else None,
        'mean': sum(quality_scores)/len(quality_scores) if quality_scores else None,
        'median': sorted(quality_scores)[len(quality_scores)//2] if quality_scores else None,
        'percentiles': percentiles,
        'below_min_quality': below_quality,
        'below_min_quality_pct': 100.0 * below_quality / len(quality_scores) if quality_scores else 0,
        'histogram_bins': hist_bins,
        'histogram_counts': hist_counts
    }
    if quality_stats['below_min_quality_pct'] > max_low_quality_pct:
        warnings.append(f"{quality_stats['below_min_quality_pct']:.2f}% of samples below min_quality_score {min_quality_score}")
    warnings = sorted(warnings)
    # Metadata consistency
    anomalies = []
    for idx, row in enumerate(rows):
        if len(anomalies) >= anomaly_cap:
            break
        if not row['class_label']:
            anomalies.append({'row_idx': idx, 'issue': 'Missing class_label'})
        if known_sources and row['dataset_source'] not in known_sources:
            anomalies.append({'row_idx': idx, 'issue': 'Unknown dataset_source'})
        if row['split'] not in valid_splits:
            anomalies.append({'row_idx': idx, 'issue': 'Invalid split name'})
        if not row['path'] or not row['export_path']:
            anomalies.append({'row_idx': idx, 'issue': 'Missing path or export_path'})
        if 'width' in row and 'height' in row:
            try:
                w = int(row['width'])
                h = int(row['height'])
                if w <= 0 or h <= 0:
                    anomalies.append({'row_idx': idx, 'issue': 'Non-positive resolution'})
            except Exception:
                anomalies.append({'row_idx': idx, 'issue': 'Invalid resolution'})
        for k, v in row.items():
            if v in [None, '']:
                anomalies.append({'row_idx': idx, 'issue': f'Empty field: {k}'})
    if len(anomalies) >= anomaly_cap:
        logger.warning(f"Anomaly cap reached ({anomaly_cap}). Further anomalies not stored.")
    anomalies = sorted(anomalies, key=lambda x: (x['row_idx'], x['issue']))
    # Report output
    thresholds_used = {
        'max_missing_pct': max_missing_pct,
        'max_leakage_allowed': max_leakage_allowed,
        'max_malformed_pct': max_malformed_pct,
        'max_low_quality_pct': max_low_quality_pct,
        'min_quality_score': min_quality_score,
        'quality_hist_bins': quality_hist_bins,
        'anomaly_cap': anomaly_cap
    }
    # Verdict logic
    verdict = 'PASS'
    missing_pct = 100.0 * len(missing_files) / max(1, len(rows))
    malformed_pct = 100.0 * len(malformed_rows) / max(1, total_rows)
    critical_leakage = len(leakage_hash)
    # Deterministic verdict logic
    if len(rows) == 0:
        verdict = 'FAIL'
    elif missing_pct > max_missing_pct:
        verdict = 'FAIL'
    elif malformed_pct > max_malformed_pct:
        verdict = 'FAIL'
    elif critical_leakage > max_leakage_allowed:
        verdict = 'FAIL'
    elif warnings or anomalies or malformed_rows or missing_files or unreadable_files or duplicate_export_paths or duplicate_orig_paths or quality_stats['below_min_quality_pct'] > max_low_quality_pct:
        verdict = 'WARN'
    # Report output
    report = {
        'timestamp': datetime.now().isoformat(),
        'config_snapshot': {k: v for k, v in sorted(config.items()) if k not in {'secrets', 'api_keys', 'password', 'token'}},
        'summary': {
            'total_rows': len(rows),
            'malformed_rows': len(malformed_rows),
            'missing_files': len(missing_files),
            'zero_byte_files': len(zero_byte_files),
            'unreadable_files': len(unreadable_files),
            'duplicate_export_paths': len(duplicate_export_paths),
            'duplicate_orig_paths': len(duplicate_orig_paths),
            'warnings': len(warnings),
            'anomalies': len(anomalies),
            'thresholds_used': thresholds_used,
            'verdict': verdict,
        },
        'distribution': {
            'split_counts': split_counts,
            'class_counts': class_counts,
            'source_counts': source_counts,
            'class_split_matrix': class_split_matrix,
            'source_split_matrix': source_split_matrix,
        },
        'integrity': {
            'missing_files': missing_files,
            'zero_byte_files': zero_byte_files,
            'unreadable_files': unreadable_files,
            'duplicate_export_paths': duplicate_export_paths,
            'duplicate_orig_paths': duplicate_orig_paths,
        },
        'leakage': {
            'critical_hash_leakage': leakage_hash,
            'soft_filename_leakage': leakage_filename,
            'soft_origpath_leakage': leakage_origpath,
            'hash_leakage_pairs': hash_leakage_pairs,
            'filename_leakage_pairs': filename_leakage_pairs,
            'origpath_leakage_pairs': origpath_leakage_pairs,
        },
        'quality': quality_stats,
        'anomalies': anomalies,
        'warnings': warnings,
        'thresholds_used': thresholds_used,
        'verdict': verdict,
        'audit_duration_sec': round(time.time() - start_time, 2)
    }
    logger.info(f"\n=== DATASET AUDIT SUMMARY ===")
    logger.info(f"Rows: {len(rows)} | Malformed: {len(malformed_rows)} | Missing files: {len(missing_files)} | Zero-byte: {len(zero_byte_files)} | Unreadable: {len(unreadable_files)}")
    logger.info(f"Duplicates: export_path={len(duplicate_export_paths)}, orig_path={len(duplicate_orig_paths)}")
    logger.info(f"Critical leakage (hash): {len(leakage_hash)} | Soft leakage (filename): {len(leakage_filename)} | Soft leakage (origpath): {len(leakage_origpath)}")
    logger.info(f"Quality: min={quality_stats['min']} max={quality_stats['max']} mean={quality_stats['mean']} median={quality_stats['median']} below_min={quality_stats['below_min_quality']}%")
    logger.info(f"Warnings: {len(warnings)} | Anomalies: {len(anomalies)} | Verdict: {verdict}")
    logger.info(f"Audit completed in {report['audit_duration_sec']} seconds.")
    if not dry_run:
        with open(output_report_path, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
    else:
        logger.info("[DRY RUN] No report written.")
