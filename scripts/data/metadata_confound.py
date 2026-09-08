#!/usr/bin/env python
"""Measure how much of a dataset's label is predictable from file metadata alone.

This is the test that found the fatal confound in this project's own 20-corpus dataset:
container format plus resolution predicted the class at 87.4%, above the trained model's
89%. Run it on any candidate dataset *before* training on it, and report the number it
gives alongside every accuracy you publish.

Why held-out. Fitting and scoring a metadata lookup on the same images inflates the result
badly whenever resolutions are near-unique -- on NTIRE 2026 val that difference is 92.6%
(fit-on-all) versus 56.5% (held-out). Only the held-out number means anything.

Usage:
    venv-linux/bin/python scripts/data/metadata_confound.py DIR
    venv-linux/bin/python scripts/data/metadata_confound.py ARCHIVE.zip
    venv-linux/bin/python scripts/data/metadata_confound.py SHARD.parquet --label-col label

For a directory, the class label is the name of the first subdirectory under DIR, so the
conventional 0_real/1_fake and real/full_synthetic/tampered layouts both work as-is.
"""

import argparse
import collections
import io
import pathlib
import random
import sys
import zipfile

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def describe(blob):
    """Everything a metadata-only attacker can read without decoding pixels."""
    im = Image.open(io.BytesIO(blob))
    qtables = getattr(im, "quantization", None)
    return {
        "fmt": im.format,
        "size": im.size,
        "bytes": len(blob),
        # The luminance quantisation table sums to a compact proxy for JPEG quality.
        "qsum": sum(qtables[0]) if qtables and 0 in qtables else -1,
        "mode": im.mode,
    }


FEATURES = [
    ("container format", lambda f: f["fmt"]),
    ("colour mode", lambda f: f["mode"]),
    ("resolution (exact)", lambda f: f["size"]),
    ("megapixels (0.1 bucket)", lambda f: round(f["size"][0] * f["size"][1] / 1e6, 1)),
    ("aspect ratio (0.1 bucket)", lambda f: round(f["size"][0] / max(f["size"][1], 1), 1)),
    ("file size (10 KB bucket)", lambda f: f["bytes"] // 10_000),
    ("JPEG quantisation table", lambda f: f["qsum"]),
    ("all of the above", lambda f: (f["fmt"], f["size"], f["qsum"], f["bytes"] // 10_000)),
]


def held_out_accuracy(rows, keyfn, seed=0):
    """Fit a majority-vote lookup on half the rows, score it on the other half."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split = len(shuffled) // 2
    train, test = shuffled[:split], shuffled[split:]

    table = collections.defaultdict(collections.Counter)
    for label, feats in train:
        table[keyfn(feats)][label] += 1
    fallback = collections.Counter(l for l, _ in train).most_common(1)[0][0]

    correct = 0
    for label, feats in test:
        votes = table.get(keyfn(feats))
        predicted = votes.most_common(1)[0][0] if votes else fallback
        correct += predicted == label
    # The baseline must come from the same held-out half, not from the whole set.
    baseline = max(collections.Counter(l for l, _ in test).values()) / len(test)
    return correct / len(test), baseline, len(test)


def load_directory(root, limit):
    root = pathlib.Path(root)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    random.Random(0).shuffle(paths)
    for path in paths[:limit]:
        label = path.relative_to(root).parts[0]
        yield label, path.read_bytes()


def load_zip(archive, limit):
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist()
                 if pathlib.PurePath(n).suffix.lower() in IMAGE_SUFFIXES]
        random.Random(0).shuffle(names)
        for name in names[:limit]:
            parts = pathlib.PurePath(name).parts
            # Skip the archive's own top-level wrapper directory if there is one.
            # A single-component path is a file at the archive root, which carries
            # no directory label -- returning parts[0] there would hand back the
            # filename and give every image its own "class", quietly producing a
            # meaningless 100% separation instead of an error.
            if len(parts) < 2:
                continue
            label = parts[1] if len(parts) > 2 else parts[0]
            yield label, zf.read(name)


def load_parquet(path, limit, label_col, image_col, key_col):
    import pyarrow.parquet as pq

    for record in pq.read_table(path).to_pylist()[:limit]:
        image = record[image_col]
        blob = image["bytes"] if isinstance(image, dict) else image
        if key_col and record.get(key_col):
            # OpenSDI encodes the real 3-class label in the key path, not in `label`:
            # entire/<gen>/fake is fully synthetic, partial/<gen>/fake is locally edited.
            parts = str(record[key_col]).split("/")
            if len(parts) >= 3:
                yield f"{parts[0]}/{parts[2]}", blob
                continue
        yield str(record[label_col]), blob


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="directory, .zip, or .parquet shard")
    parser.add_argument("--limit", type=int, default=5000,
                        help="max images to sample (default 5000)")
    parser.add_argument("--label-col", default="label", help="parquet label column")
    parser.add_argument("--image-col", default="image", help="parquet image column")
    parser.add_argument("--key-col", default=None,
                        help="parquet path-like column that encodes a finer label "
                             "(use --key-col key for OpenSDI)")
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    if source.is_dir():
        loader = load_directory(source, args.limit)
    elif source.suffix == ".zip":
        loader = load_zip(source, args.limit)
    elif source.suffix == ".parquet":
        loader = load_parquet(source, args.limit, args.label_col,
                              args.image_col, args.key_col)
    else:
        sys.exit(f"don't know how to read {source}")

    rows, skipped = [], 0
    for label, blob in loader:
        try:
            rows.append((label, describe(blob)))
        except Exception:
            skipped += 1

    if len(rows) < 20:
        sys.exit(f"only {len(rows)} readable images -- not enough to measure anything")

    counts = collections.Counter(l for l, _ in rows)
    print(f"{source}  n={len(rows)}" + (f" ({skipped} unreadable)" if skipped else ""))
    print(f"classes: {dict(counts)}")

    if len(counts) < 2:
        sys.exit("only one class present -- nothing to measure. Some datasets order their "
                 "shards by class (OpenSDI does); sample across shards instead.")

    _, baseline, n_test = held_out_accuracy(rows, lambda _: 0)
    print(f"majority-class baseline: {baseline * 100:.1f}%  (held-out n={n_test})\n")

    results = []
    for name, keyfn in FEATURES:
        accuracy, _, _ = held_out_accuracy(rows, keyfn)
        results.append((accuracy, name))
        margin = (accuracy - baseline) * 100
        flag = "  <-- LEAK" if margin > 10 else ""
        print(f"  {name:28s} {accuracy * 100:5.1f}%   ({margin:+5.1f} pts){flag}")

    best, best_name = max(results)
    margin = (best - baseline) * 100
    print()
    if margin > 10:
        print(f"CONFOUNDED: '{best_name}' alone reaches {best * 100:.1f}%, "
              f"{margin:.1f} points over baseline.")
        print("Re-encode every image to one container at one quality and resize to a fixed "
              "size before drawing any split, then re-run this.")
    else:
        print(f"Clean enough: best feature '{best_name}' is only "
              f"{margin:.1f} points over baseline.")


if __name__ == "__main__":
    main()
