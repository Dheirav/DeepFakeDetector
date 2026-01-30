import csv
import random
from collections import defaultdict, Counter
from pathlib import Path


def group_by_class(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['class_label']].append(row)
    return groups

def rank_rows(rows):
    # Centralized ranking: quality_score desc, resolution desc, file_size_bytes desc, path asc
    return sorted(rows, key=lambda r: (
        -float(r.get('quality_score', 0) or 0),
        -int(r.get('width', 0)) * int(r.get('height', 0)),
        -int(r.get('file_size_bytes', 0)),
        r.get('path', '')
    ))

def apply_quality_filter(rows, min_quality_score, logger):
    if min_quality_score is None:
        return rows, 0
    filtered = []
    skipped = 0
    for row in rows:
        try:
            q = float(row.get('quality_score', 0))
        except Exception:
            q = 0
        if q >= min_quality_score:
            filtered.append(row)
        else:
            skipped += 1
    if skipped:
        logger.info(f"Dropped {skipped} images below min_quality_score {min_quality_score}")
    return filtered, skipped

def balance_sources(rows, class_target, per_source_max_fraction, seed, logger):
    # Group by source
    sources = defaultdict(list)
    for row in rows:
        sources[row['dataset_source']].append(row)
    max_per_source = int(per_source_max_fraction * class_target) if per_source_max_fraction else class_target
    selected = []
    skipped = 0
    rng = random.Random(seed)
    # First, cap each source
    capped = {}
    for source, items in sources.items():
        ranked = rank_rows(items)
        if len(ranked) > max_per_source:
            capped[source] = ranked[:max_per_source]
            skipped += len(ranked) - max_per_source
        else:
            capped[source] = ranked
    # Fill up to class_target, refilling from sources with remaining quota
    pool = []
    for source, items in capped.items():
        pool.extend(items)
    # If more than needed, rank and trim
    if len(pool) > class_target:
        pool = rank_rows(pool)[:class_target]
    # If less than needed, try to refill from sources that were capped
    elif len(pool) < class_target:
        deficit = class_target - len(pool)
        refill = []
        for source, items in sources.items():
            if len(items) > max_per_source:
                overflow = rank_rows(items)[max_per_source:]
                refill.extend(overflow)
        if refill:
            refill = rank_rows(refill)
            pool.extend(refill[:deficit])
    # Final trim if over
    if len(pool) > class_target:
        pool = rank_rows(pool)[:class_target]
    if skipped:
        logger.info(f"Dropped {skipped} images due to per_source_max_fraction cap")
    if len(pool) < class_target:
        logger.warning(f"Not enough images after source balancing: needed {class_target}, got {len(pool)}")
    return pool, skipped

def sample_class(rows, class_label, class_target, per_source_max_fraction, min_quality_score, seed, logger):
    filtered, skipped_quality = apply_quality_filter(rows, min_quality_score, logger)
    if not filtered:
        logger.warning(f"No images for class {class_label} after quality filtering.")
        return [], skipped_quality, 0
    filtered = rank_rows(filtered)
    selected, skipped_source = balance_sources(filtered, class_target, per_source_max_fraction, seed, logger)
    if len(selected) > class_target:
        selected = rank_rows(selected)[:class_target]
    return selected, skipped_quality, skipped_source

def sample_dataset(input_csv, output_csv, config, logger, dry_run=False):
    class_targets = config.get('class_targets', {})
    max_total_images = config.get('max_total_images')
    per_source_max_fraction = config.get('per_source_max_fraction')
    min_quality_score = config.get('min_quality_score')
    seed = config.get('random_seed', 42)
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    with open(input_path, 'r', newline='') as f:
        reader = list(csv.DictReader(f))
    if not reader:
        logger.warning(f"Input file {input_csv} is empty. No sampling performed.")
        return
    class_groups = group_by_class(reader)
    all_sampled = []
    per_class_sampled = {}
    total_skipped_quality = 0
    total_skipped_source = 0
    logger.info(f"Sampling targets: {class_targets}")
    for class_label, class_target in class_targets.items():
        rows = class_groups.get(class_label, [])
        if not rows:
            logger.warning(f"No images found for class {class_label}.")
            continue
        sampled, skipped_quality, skipped_source = sample_class(
            rows, class_label, class_target, per_source_max_fraction, min_quality_score, seed, logger)
        logger.info(f"Requested {class_target} for class {class_label}, sampled {len(sampled)}")
        src_counts = Counter(r['dataset_source'] for r in sampled)
        logger.info(f"Per-source counts for class {class_label}: {dict(src_counts)}")
        if len(sampled) < class_target:
            logger.warning(f"Class {class_label}: only {len(sampled)} samples available (target {class_target})")
        all_sampled.extend(sampled)
        per_class_sampled[class_label] = sampled
        total_skipped_quality += skipped_quality
        total_skipped_source += skipped_source
    # Global cap: preserve class balance as much as possible
    if max_total_images and len(all_sampled) > max_total_images:
        logger.info(f"Applying max_total_images cap: {max_total_images}")
        # Compute per-class quotas proportional to sampled counts
        total = sum(len(v) for v in per_class_sampled.values())
        quotas = {k: int(max_total_images * len(v) / total) for k, v in per_class_sampled.items()}
        # Ensure at least 1 per class if possible
        for k in quotas:
            if quotas[k] == 0 and len(per_class_sampled[k]) > 0:
                quotas[k] = 1
        # Sample per class
        new_sampled = []
        for class_label, samples in per_class_sampled.items():
            if len(samples) > quotas[class_label]:
                new_sampled.extend(rank_rows(samples)[:quotas[class_label]])
            else:
                new_sampled.extend(samples)
        # If still over, trim
        if len(new_sampled) > max_total_images:
            new_sampled = rank_rows(new_sampled)[:max_total_images]
        all_sampled = new_sampled
    logger.info(f"Final sampled images: {len(all_sampled)}")
    logger.info(f"Total dropped due to quality: {total_skipped_quality}")
    logger.info(f"Total dropped due to source cap: {total_skipped_source}")
    # Per-class and per-source summary
    for class_label in class_targets:
        count = len([r for r in all_sampled if r['class_label'] == class_label])
        logger.info(f"Final count for class {class_label}: {count}")
        src_counts = Counter(r['dataset_source'] for r in all_sampled if r['class_label'] == class_label)
        logger.info(f"Final per-source for class {class_label}: {dict(src_counts)}")
    if not dry_run:
        if all_sampled:
            with open(output_path, 'w', newline='') as f:
                fieldnames = list(all_sampled[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_sampled:
                    writer.writerow(row)
        else:
            logger.warning("No images selected for sampling. Output not written.")
    else:
        logger.info("[DRY RUN] No output written.")
