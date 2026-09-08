import csv
import random
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from modules.clustering import build_clusters_from_hashes, hamming_distance as _hamming

def build_clusters(rows, phash_threshold, phash_bucket_chars, logger):
    """Group rows into near-duplicate clusters by perceptual hash.

    ``phash_bucket_chars`` is accepted for config compatibility but no longer
    used: prefix bucketing is what made this stage find nothing (0 of 1,287
    genuine pairs on this project's own hashes). Clustering now goes through
    ``modules.clustering``, which uses multi-index hashing and recovers
    100% of pairs at Hamming <= 7. See that module for the measurements.
    """
    items = [(row['path'], row.get('phash')) for row in rows]
    assignment = build_clusters_from_hashes(items, phash_threshold, logger=logger)

    grouped = defaultdict(list)
    for row in rows:
        grouped[assignment[row['path']]].append(row)
    clusters = list(grouped.values())

    multi_class = [c for c in clusters if len({r['class_label'] for r in c}) > 1]
    logger.info(
        f"Total clusters: {len(clusters)} | Clusters with >1 class_label: "
        f"{len(multi_class)} ({100 * len(multi_class) / max(len(clusters), 1):.2f}%)"
    )
    sizes = [len(c) for c in clusters]
    for pct in (50, 90, 99):
        logger.info(f"Cluster size P{pct}: {int(np.percentile(sizes, pct)) if sizes else 0}")
    if sizes and max(sizes) == 1:
        logger.warning(
            "Every cluster is a singleton -- no near-duplicates were found at all. "
            "Check that the phash column is populated; a vacuous clustering makes "
            "the leakage check below trivially pass."
        )
    return clusters


def assign_clusters_to_splits(clusters, split_ratios, seed, logger, holdout_sources=None):
    """Assign clusters to splits, optionally holding whole sources out for test.

    ``holdout_sources`` switches on leave-one-source-out. Every cluster touching a
    held-out source goes entirely to ``test``; the rest are balanced over the
    remaining splits, and the source-distribution cost term is dropped -- that
    term exists to force every corpus into every split, which is precisely the
    arrangement that makes the test set unable to measure generalisation.

    Why this matters here: all 20 source corpora map to exactly one class, so a
    random split lets the model answer "which dataset is this from" instead of
    "has this been manipulated". Holding out whole sources is the only
    configuration in this pipeline that measures transfer. Expect the number to
    drop substantially; that drop is the honest figure.
    """
    # Deterministic shuffle
    rng = random.Random(seed)
    clusters = clusters[:]
    rng.shuffle(clusters)
    split_names = list(split_ratios.keys())
    # Compute total image count and per-split targets
    total_images = sum(len(c) for c in clusters)
    split_targets = {k: split_ratios[k] * total_images for k in split_names}
    # Initialize per-split stats
    splits = {k: [] for k in split_names}
    split_img_counts = {k: 0 for k in split_names}
    split_class_counts = {k: Counter() for k in split_names}
    split_source_counts = {k: Counter() for k in split_names}
    # Compute global class/source distributions
    global_class = Counter()
    global_source = Counter()
    for cluster in clusters:
        for row in cluster:
            global_class[row['class_label']] += 1
            global_source[row['dataset_source']] += 1
    holdout = set(holdout_sources or ())
    if holdout:
        if 'test' not in split_names:
            raise ValueError("holdout_sources requires a 'test' split in split_ratios")
        held, remaining = [], []
        for cluster in clusters:
            (held if any(r['dataset_source'] in holdout for r in cluster) else remaining).append(cluster)
        logger.info(
            f"Leave-one-source-out: holding out {sorted(holdout)} -- "
            f"{sum(len(c) for c in held)} images to test, "
            f"{sum(len(c) for c in remaining)} to be split across the rest"
        )
        for cluster in held:
            splits['test'].append(cluster)
        clusters = remaining
        # Re-target the remaining splits over what is actually left.
        split_names = [k for k in split_names if k != 'test']
        if not split_names:
            raise ValueError("holdout_sources consumed every split; leave train/val configured")
        left = sum(len(c) for c in clusters)
        scale = sum(split_ratios[k] for k in split_names)
        split_targets = {k: split_ratios[k] / scale * left for k in split_names}

    # Assign clusters greedily by cost
    for cluster in clusters:
        # Gather cluster stats
        cluster_size = len(cluster)
        cluster_class = Counter(row['class_label'] for row in cluster)
        cluster_source = Counter(row['dataset_source'] for row in cluster)
        # Evaluate cost for each split.
        #
        # Cost is the split's *resulting relative fill* -- how full it would be
        # against its own target after taking this cluster -- so argmin hands the
        # cluster to whichever split is furthest behind. Filling proportionally
        # like this converges on the configured ratios.
        #
        # It previously used abs(resulting - target)/target. While a split is
        # under target that equals 1 - fill_fraction, so the *emptiest* split
        # scored highest and argmin picked the one already proportionally
        # fullest. A small split kept winning until its relative overshoot
        # exceeded a large split's relative deficit, i.e. until it reached
        # 2 x target -- which pinned val and test at 30% each and left train the
        # remainder. Every dataset built before 2026-09-08 is 40/30/30 rather
        # than the configured 70/15/15, so those models trained on 31,183 images
        # instead of 54,505. See docs/DATASET_BUILDER_AUDIT.md section 1.
        costs = []
        for split in split_names:
            # Image count fill
            img_count = split_img_counts[split] + cluster_size
            img_cost = img_count / (split_targets[split] + 1e-9)
            # Class distribution fill (mean over classes present in this cluster)
            class_cost = 0.0
            for cl in global_class:
                after = split_class_counts[split][cl] + cluster_class[cl]
                target = global_class[cl] * split_ratios[split]
                class_cost += after / (target + 1e-9) if target > 0 else 0
            class_cost /= max(len(global_class), 1)
            # Source distribution fill
            source_cost = 0.0
            for src in global_source:
                after = split_source_counts[split][src] + cluster_source[src]
                target = global_source[src] * split_ratios[split]
                source_cost += after / (target + 1e-9) if target > 0 else 0
            source_cost /= max(len(global_source), 1)
            if holdout:
                # Balancing sources across splits is the confound, not the goal.
                source_cost = 0.0
            # Weighted sum (tune weights if needed)
            total_cost = img_cost + class_cost + source_cost
            costs.append((total_cost, split))
        # Deterministic tie-breaker: sort by (cost, split name)
        costs.sort()
        best_split = costs[0][1]
        splits[best_split].append(cluster)
        split_img_counts[best_split] += cluster_size
        for cl in cluster_class:
            split_class_counts[best_split][cl] += cluster_class[cl]
        for src in cluster_source:
            split_source_counts[best_split][src] += cluster_source[src]
    # Flatten
    split_map = {}
    for split, clist in splits.items():
        for cluster in clist:
            for row in cluster:
                split_map[row['path']] = split
    return split_map

def validate_no_leakage(rows, split_map, cluster_ids, logger):
    cluster_split = {}
    for row in rows:
        cid = cluster_ids[row['path']]
        split = split_map[row['path']]
        if cid not in cluster_split:
            cluster_split[cid] = split
        elif cluster_split[cid] != split:
            logger.error(f"Cluster leakage: cluster_id {cid} in multiple splits!")
            raise AssertionError(f"Cluster {cid} assigned to multiple splits.")
    logger.info("Leakage check passed: No cluster spans multiple splits.")

def generate_split_report(rows, split_map, cluster_ids, output_csv, logger):
    # Prepare stats
    split_stats = defaultdict(lambda: {'clusters': set(), 'images': 0, 'class_label': Counter(), 'dataset_source': Counter()})
    for row in rows:
        split = split_map[row['path']]
        cid = cluster_ids[row['path']]
        split_stats[split]['clusters'].add(cid)
        split_stats[split]['images'] += 1
        split_stats[split]['class_label'][row['class_label']] += 1
        split_stats[split]['dataset_source'][row['dataset_source']] += 1
    # Write report
    fieldnames = ['split','num_clusters','num_images','class_label_counts','dataset_source_counts']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, stats in split_stats.items():
            writer.writerow({
                'split': split,
                'num_clusters': len(stats['clusters']),
                'num_images': stats['images'],
                'class_label_counts': dict(stats['class_label']),
                'dataset_source_counts': dict(stats['dataset_source'])
            })
    # Log summary
    for split, stats in split_stats.items():
        logger.info(f"Split {split}: {len(stats['clusters'])} clusters, {stats['images']} images, class dist: {dict(stats['class_label'])}, source dist: {dict(stats['dataset_source'])}")

def split_dataset(input_csv, output_csv, report_csv, config, logger, dry_run=False):
    phash_threshold = config.get('phash_cluster_threshold', 8)
    phash_bucket_chars = config.get('phash_bucket_chars', 12)
    split_ratios = config.get('split_ratios', {'train':0.8,'val':0.1,'test':0.1})
    seed = config.get('random_seed', 42)
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    report_path = Path(report_csv)
    with open(input_path, 'r', newline='') as f:
        reader = list(csv.DictReader(f))
    # Clean rows
    rows = []
    for row in reader:
        if not row.get('path') or not row.get('class_label') or not row.get('dataset_source'):
            logger.warning(f"Malformed row skipped: {row}")
            continue
        rows.append(row)
    logger.info(f"Loaded {len(rows)} valid rows from {input_csv}")
    clusters = build_clusters(rows, phash_threshold, phash_bucket_chars, logger)
    # Assign cluster_id
    cluster_ids = {}
    for i, cluster in enumerate(clusters):
        cid = f"cluster_{i+1:05d}"
        for row in cluster:
            cluster_ids[row['path']] = cid
    split_map = assign_clusters_to_splits(clusters, split_ratios, seed, logger)
    validate_no_leakage(rows, split_map, cluster_ids, logger)
    if not dry_run:
        with open(output_path, 'w', newline='') as f:
            fieldnames = list(rows[0].keys()) + ['cluster_id','split']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row_out = dict(row)
                row_out['cluster_id'] = cluster_ids[row['path']]
                row_out['split'] = split_map[row['path']]
                writer.writerow(row_out)
        generate_split_report(rows, split_map, cluster_ids, report_path, logger)
    else:
        logger.info("[DRY RUN] No output written.")
        generate_split_report(rows, split_map, cluster_ids, report_path, logger)
    logger.info("Split complete.")
