import csv
import random
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np

def hamming_distance(phash1, phash2):
    try:
        int(phash1, 16)
        int(phash2, 16)
        return bin(int(phash1, 16) ^ int(phash2, 16)).count('1')
    except Exception:
        return 999

def build_clusters(rows, phash_threshold, phash_bucket_chars, logger):
    # Group by phash bucket, then cluster within bucket
    phash_buckets = defaultdict(list)
    singleton_clusters = []
    for row in rows:
        phash = row.get('phash')
        if not phash or len(phash) < phash_bucket_chars:
            singleton_clusters.append([row])
        else:
            bucket = phash[:phash_bucket_chars]
            phash_buckets[bucket].append(row)
    clusters = []
    for bucket, group in tqdm(phash_buckets.items(), desc='Clustering by phash'):
        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x
        def union(x, y):
            parent[find(x)] = find(y)
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                d = hamming_distance(group[i]['phash'], group[j]['phash'])
                if d <= phash_threshold:
                    union(group[i]['path'], group[j]['path'])
        cluster_map = defaultdict(list)
        for r in group:
            cluster_map[find(r['path'])].append(r)
        clusters.extend(list(cluster_map.values()))
    clusters.extend(singleton_clusters)
    # Cluster purity logging
    multi_class_clusters = [c for c in clusters if len(set(r['class_label'] for r in c)) > 1]
    logger.info(f"Total clusters: {len(clusters)} | Clusters with >1 class_label: {len(multi_class_clusters)} ({100*len(multi_class_clusters)/len(clusters):.2f}%)")
    # Cluster size percentiles
    sizes = [len(c) for c in clusters]
    for p in [50, 90, 99]:
        logger.info(f"Cluster size P{p}: {int(np.percentile(sizes, p)) if sizes else 0}")
    return clusters

def assign_clusters_to_splits(clusters, split_ratios, seed, logger):
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
    # Assign clusters greedily by cost
    for cluster in clusters:
        # Gather cluster stats
        cluster_size = len(cluster)
        cluster_class = Counter(row['class_label'] for row in cluster)
        cluster_source = Counter(row['dataset_source'] for row in cluster)
        # Evaluate cost for each split
        costs = []
        for split in split_names:
            # Image count cost
            img_count = split_img_counts[split] + cluster_size
            img_cost = abs(img_count - split_targets[split]) / (split_targets[split] + 1e-9)
            # Class dist cost (L1 norm)
            class_cost = 0.0
            for cl in global_class:
                after = split_class_counts[split][cl] + cluster_class[cl]
                target = global_class[cl] * split_ratios[split]
                class_cost += abs(after - target) / (target + 1e-9) if target > 0 else 0
            # Source dist cost (L1 norm)
            source_cost = 0.0
            for src in global_source:
                after = split_source_counts[split][src] + cluster_source[src]
                target = global_source[src] * split_ratios[split]
                source_cost += abs(after - target) / (target + 1e-9) if target > 0 else 0
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
