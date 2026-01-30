import csv
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

try:
    import imagehash
    from PIL import Image
except ImportError:
    imagehash = None
    Image = None


def tie_breaker(rows):
    def sort_key(row):
        q = float(row.get('quality_score', 0) or 0)
        width = int(row.get('width', 0))
        height = int(row.get('height', 0))
        res = width * height
        size = int(row.get('file_size_bytes', 0))
        path = row.get('path', '')
        return (-q, -res, -size, path)
    return sorted(rows, key=sort_key)[0]


def group_by_key(rows, key):
    groups = defaultdict(list)
    for row in rows:
        k = row.get(key)
        if k and k.strip():
            groups[k].append(row)
    return groups


def hamming_distance(phash1, phash2):
    try:
        if not (phash1 and phash2):
            return 999
        if imagehash and hasattr(imagehash, 'hex_to_hash'):
            h1 = imagehash.hex_to_hash(phash1)
            h2 = imagehash.hex_to_hash(phash2)
            return h1 - h2
        int(phash1, 16)
        int(phash2, 16)
        return bin(int(phash1, 16) ^ int(phash2, 16)).count('1')
    except Exception:
        return 999


def deduplicate_images(index_csv, output_csv, config, logger, dry_run=False):
    phash_threshold = config.get('phash_hamming_threshold', 8)
    phash_bucket_chars = config.get('phash_bucket_chars', 12)  # clarify: chars, not bits
    project_root = Path(config.get('project_root', '.'))
    input_path = Path(index_csv)
    output_path = Path(output_csv)
    conflict_path = output_path.parent / 'conflict_report.csv'
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(output_path, 'w', newline='')
        conflict_f = open(conflict_path, 'w', newline='')
        writer = None
        conflict_writer = None
    else:
        out_f = None
        conflict_f = None
        writer = None
        conflict_writer = None

    with open(input_path, 'r', newline='') as in_f:
        reader = list(csv.DictReader(in_f))
        fieldnames = list(reader[0].keys()) if reader else []
        conflict_fields = fieldnames + ['conflict_type','conflict_paths','conflict_labels','conflict_hash','winner_path','winner_quality_score','winner_resolution']
        if not dry_run:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            conflict_writer = csv.DictWriter(conflict_f, fieldnames=conflict_fields)
            conflict_writer.writeheader()
        kept = set()
        removed = set()
        cross_class_conflicts = []
        exact_dupes_removed = 0
        near_dupes_removed = 0
        cross_class_count = 0
        # 1. Exact duplicate removal
        md5_groups = group_by_key(reader, 'md5_hash')
        all_md5_rows = set()
        for md5, group in tqdm(md5_groups.items(), desc='Exact deduplication'):
            if not md5 or len(group) == 1:
                for r in group:
                    kept.add(r['path'])
                continue
            all_md5_rows.update(r['path'] for r in group)
            class_labels = set(r['class_label'] for r in group)
            best = tie_breaker(group)
            kept.add(best['path'])
            for r in group:
                if r['path'] != best['path']:
                    removed.add(r['path'])
                    exact_dupes_removed += 1
                    if len(class_labels) > 1:
                        cross_class_count += 1
                        conflict_context = {
                            **r,
                            'conflict_type': 'exact',
                            'conflict_paths': '|'.join([x['path'] for x in group]),
                            'conflict_labels': '|'.join([x['class_label'] for x in group]),
                            'conflict_hash': md5,
                            'winner_path': best['path'],
                            'winner_quality_score': best.get('quality_score',''),
                            'winner_resolution': f"{best.get('width','')}x{best.get('height','')}"
                        }
                        cross_class_conflicts.append(conflict_context)
                        logger.error(f"Cross-class exact duplicate: {r['path']} md5={md5} winner={best['path']}")
        # Ensure rows without md5_hash are not excluded
        for row in reader:
            if not row.get('md5_hash') and row['path'] not in kept and row['path'] not in removed:
                kept.add(row['path'])
        # 2. Near-duplicate removal (phash, transitive)
        for row in reader:
            if not row.get('phash') and imagehash and Image:
                try:
                    abs_path = str((project_root / row['path']).absolute())
                    img = Image.open(abs_path)
                    row['phash'] = str(imagehash.phash(img))
                except Exception:
                    continue
        # Bucket by first N chars
        phash_groups = defaultdict(list)
        for row in reader:
            if row['path'] in kept and row.get('phash') and len(row['phash']) >= phash_bucket_chars:
                bucket = row['phash'][:phash_bucket_chars]
                phash_groups[bucket].append(row)
        # Transitive clustering
        for bucket, group in tqdm(phash_groups.items(), desc='Near-duplicate deduplication'):
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
            clusters = defaultdict(list)
            for r in group:
                clusters[find(r['path'])].append(r)
            for cluster in clusters.values():
                if len(cluster) == 1:
                    continue
                class_labels = set(r['class_label'] for r in cluster)
                best = tie_breaker(cluster)
                for r in cluster:
                    if r['path'] == best['path'] or r['path'] not in kept:
                        continue
                    kept.discard(r['path'])
                    removed.add(r['path'])
                    near_dupes_removed += 1
                    if len(class_labels) > 1:
                        cross_class_count += 1
                        conflict_context = {
                            **r,
                            'conflict_type': 'near',
                            'conflict_paths': '|'.join([x['path'] for x in cluster]),
                            'conflict_labels': '|'.join([x['class_label'] for x in cluster]),
                            'conflict_hash': r.get('phash',''),
                            'winner_path': best['path'],
                            'winner_quality_score': best.get('quality_score',''),
                            'winner_resolution': f"{best.get('width','')}x{best.get('height','')}"
                        }
                        cross_class_conflicts.append(conflict_context)
                        logger.error(f"Cross-class near-duplicate: {r['path']} phash={r.get('phash','')} winner={best['path']}")
        # Write outputs
        deduped_rows = [row for row in reader if row['path'] in kept]
        if not dry_run:
            for row in deduped_rows:
                writer.writerow(row)
            for row in cross_class_conflicts:
                conflict_writer.writerow(row)
    if not dry_run:
        out_f.close()
        conflict_f.close()
    logger.info(f"Deduplication complete. Output: {output_path if not dry_run else '[DRY RUN]'}")
    logger.info(f"Total images: {len(reader)} | Kept: {len(kept)} | Removed: {len(removed)} | Exact duplicates removed: {exact_dupes_removed} | Near duplicates removed: {near_dupes_removed} | Cross-class conflicts: {cross_class_count}")
