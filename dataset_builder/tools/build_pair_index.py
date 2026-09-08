#!/usr/bin/env python3
"""Recover manipulation/original pairs from the surviving export metadata.

Why this exists
---------------
Every source corpus in this dataset maps to exactly one class, so corpus
identity is a perfect substitute for the label and the model learns to read file
provenance instead of manipulation traces (see ``LIMITATIONS.md``). No split
strategy fixes that; the confound is a property of the corpus.

The construction that *does* fix it is matched pairs: use each manipulated
image's **own original** as its ``real`` counterpart. Both halves then share a
camera, a codec, a resolution and a compression history, so the only remaining
difference is the manipulation itself.

Four of the ``ai_edited`` sources encode their originating image's identifier
directly in the filename, so the pairing is recoverable from metadata alone --
no pixels required, and it still works with the exported images deleted:

    DEFACTO             0_000000000322.tif        -> COCO train2017 id 000000000322
    DEFACTO_Inpainting  0_000000000322_1.tif      -> COCO train2017 id 000000000322
    CASIA               Tp_D_CNN_M_B_nat10139_nat00059_11949.jpg
                                                  -> CASIA v2 authentic ids nat10139 / nat00059
    IMD2020             c95adiz_0.jpg             -> IMD2020 directory c95adiz, file <id>_orig.jpg

FaceForensics and OpenForensics cannot be paired from the export: the frame
extractor named files by a global counter (``ff_0000004.jpg``), discarding video,
identity and frame index, and the OpenForensics downloader flattened its archive
without keeping the per-face annotations. Both need re-extraction with a
provenance-preserving naming scheme.

Output
------
``pair_index.csv`` -- one row per manipulated image, with the identifier of the
original it was built from and where that original has to be fetched from. This
is the shopping list for the rebuild, not a finished dataset.

Usage
-----
    venv-linux/bin/python dataset_builder/tools/build_pair_index.py \
        --artifacts dataset_builder/output/artifacts \
        --out dataset_builder/pair_index.csv
"""

import argparse
import csv
import glob
import os
import re
from collections import Counter, defaultdict

# DEFACTO / DEFACTO_Inpainting: <prefix>_<12-digit COCO id>[_<n>].tif
_COCO_ID = re.compile(r"(?<!\d)(\d{12})(?!\d)")
# CASIA v2 tampered: Tp_<S|D>_<ops>_<f4>_<f5>_<id1>_<id2>_<serial>[_<n>].<ext>
# Fields 4 and 5 vary more than the commonly-quoted M|N / N|B (S also occurs),
# so they are matched permissively rather than enumerated.
_CASIA = re.compile(r"^Tp_[SD]_[A-Za-z]{3}_[A-Za-z]_[A-Za-z]_([a-z]{3}\d+)_([a-z]{3}\d+)_\d+", re.I)
# IMD2020 uses two naming forms: <base-id>_<n> and <base-id>_fake.
_IMD = re.compile(r"^([0-9a-z]+)_(?:\d+|fake)$", re.I)


def parse(source, filename):
    """Return ``(original_id, origin_corpus, note)`` or ``None`` if unpairable."""
    stem = os.path.splitext(filename)[0]

    if source in ("DEFACTO", "DEFACTO_Inpainting"):
        m = _COCO_ID.search(stem)
        return (m.group(1), "COCO train2017", "") if m else None

    if source == "CASIA":
        m = _CASIA.match(stem)
        if not m:
            return None
        donor, host = m.group(1), m.group(2)
        # CASIA v2 puts two authentic ids in the name. Which is the tampered
        # host and which donated the pasted region is a property of the release,
        # not of the filename -- confirm against the Au/ directory before
        # treating either as the matched original.
        return (host, "CASIA v2 authentic (Au/)", f"donor={donor};host_unconfirmed")

    if source == "IMD2020":
        m = _IMD.match(stem)
        return (m.group(1), "IMD2020 <id>/<id>_orig.jpg", "") if m else None

    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="dataset_builder/output/artifacts")
    ap.add_argument("--out", default="dataset_builder/pair_index.csv")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.artifacts, "*", "export_index.csv"))):
        rows.extend(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"No export_index.csv found under {args.artifacts}")

    # Originals already present in the build, so they need no re-download.
    have = defaultdict(set)
    for r in rows:
        src = os.path.basename(r["dataset_source"])
        if src == "COCO":
            m = _COCO_ID.search(os.path.splitext(os.path.basename(r["export_path"]))[0])
            if m:
                have["COCO train2017"].add(m.group(1))

    out, stats, distinct = [], Counter(), defaultdict(set)
    for r in rows:
        src = os.path.basename(r["dataset_source"])
        if r["class_label"] != "ai_edited":
            continue
        stats[f"{src}:total"] += 1
        parsed = parse(src, os.path.basename(r["export_path"]))
        if parsed is None:
            stats[f"{src}:unpairable"] += 1
            continue
        original_id, origin, note = parsed
        stats[f"{src}:pairable"] += 1
        distinct[src].add(original_id)
        out.append({
            "manipulated_export_path": r["export_path"],
            "manipulated_split": r["split"],
            "source": src,
            "original_id": original_id,
            "original_corpus": origin,
            "original_already_in_build": original_id in have.get(origin, ()),
            "note": note,
        })

    print(f"Scanned {len(rows)} rows across {len(set(os.path.basename(r['dataset_source']) for r in rows))} sources\n")
    print(f"{'source':<22}{'ai_edited':>10}{'pairable':>10}{'distinct originals':>20}")
    for src in sorted({r["source"] for r in out} | {k.split(':')[0] for k in stats}):
        tot, ok = stats.get(f"{src}:total", 0), stats.get(f"{src}:pairable", 0)
        if not tot:
            continue
        print(f"{src:<22}{tot:>10}{ok:>10}{len(distinct.get(src, ())):>20}")
    print(f"\n{'TOTAL PAIRABLE':<22}{'':>10}{len(out):>10}{sum(len(v) for v in distinct.values()):>20}")

    already = sum(1 for r in out if r["original_already_in_build"])
    print(f"\nOriginals already present in the current build: {already}")
    print("Everything else must be re-fetched from the origin corpus listed per row.")

    if args.dry_run:
        print("\n[dry run] nothing written")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nWrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
