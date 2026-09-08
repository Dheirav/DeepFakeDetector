#!/usr/bin/env python3
"""Ordered, resumable fetcher for the source corpora.

Reachability of every `auto` entry below was verified on 2026-09-08. Anything
this script cannot fetch without a human is marked `manual` or `gated` and says
why -- it does not silently skip or invent a URL.

Order matters. Manipulation corpora come first because their filenames determine
which originals are needed (see ``dataset_builder/pair_index.csv``); COCO is then
fetched once and supplies both the ``real`` sample and DEFACTO's originals.
Downloading COCO twice costs 18 GB, so ``--plan`` exists to stop that happening.

Usage
-----
    fetch.py --list                 status of every source
    fetch.py --plan                 the order to fetch them in, and why
    fetch.py coco                   fetch one source (asks first)
    fetch.py coco --yes             ...without asking
    fetch.py --all-auto --yes       everything that needs no human

Downloads resume, so an interrupted fetch continues rather than restarting.
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data_sources")

# status: auto   -- verified reachable, no human needed
#         gated  -- automatable but needs an account or an accepted licence
#         manual -- no canonical scriptable URL found; fetch by hand
SOURCES = [
    # --- manipulations first: their filenames define which originals we need ---
    dict(name="defacto", cls="ai_edited", status="gated", order=1, dir="DEFACTO",
         how="kaggle datasets download -d defactodataset/defactoinpainting",
         note="Kaggle account + `kaggle` CLI configured (~/.kaggle/kaggle.json). "
              "Splice/copy-move/inpainting are separate Kaggle datasets. "
              "DOWNLOAD THE MASKS TOO -- the first build took images only."),
    dict(name="casia", cls="ai_edited", status="manual", order=1, dir="CASIA",
         how="CASIA v2.0 tampered detection dataset",
         note="No canonical URL survives in this repo. Community mirrors exist on "
              "GitHub and Kaggle; verify against the paper before trusting one. "
              "Fetch the Au/ authentic set as well -- it supplies the matched "
              "originals for 1,720 of the tampered images."),
    dict(name="imd2020", cls="ai_edited", status="manual", order=1, dir="IMD2020",
         how="IMD2020 real-life manipulated image dataset",
         note="Published by UTIA CAS. Each directory ships <id>_orig.jpg, which is "
              "the matched original for 1,719 pairs. Take the masks."),
    dict(name="faceforensics", cls="ai_edited", status="gated", order=1, dir="FaceForensics",
         how="dataset_builder/scripts/download_faceforensics.py",
         note="Requires the signed FF++ EULA (form linked from the official repo). "
              "Pass --all-sequences to extract_ff_frames.py this time: the first "
              "build took manipulated sequences only, so no pristine counterparts "
              "exist. Also fetch TYPE=masks. Re-extract with provenance-preserving "
              "filenames -- the global counter destroyed video/identity/frame."),
    dict(name="openforensics", cls="ai_edited", status="auto", order=1, dir="OpenForensics",
         url="https://zenodo.org/api/records/5528418",
         script="download_openforensics.py", size="~10 GB",
         note="Zenodo, CC-BY-4.0. Keep the annotation JSON this time; without it "
              "the real/fake face split is unrecoverable."),

    # --- originals: COCO supplies DEFACTO's matched pairs ---
    dict(name="coco", cls="real", status="auto", order=2, dir="COCO",
         url="http://images.cocodataset.org/zips/train2017.zip", size="18.0 GB",
         note="Supplies BOTH the `real` sample AND the 13,110 originals named in "
              "pair_index.csv. Export both in one pass or you fetch 18 GB twice."),

    # --- remaining real ---
    dict(name="places365", cls="real", status="auto", order=3, dir="Places365",
         url="http://data.csail.mit.edu/places/places365/val_256.tar",
         script="download_places365.py", size="~2 GB"),
    dict(name="ffhq", cls="real", status="manual", order=3, dir="FFHQ",
         how="NVlabs/ffhq-dataset",
         note="Official downloader uses Google Drive and is heavily rate-limited. "
              "Mirrors exist on Hugging Face and academic torrents; check the "
              "licence terms of whichever you use."),
    dict(name="openimages", cls="real", status="manual", order=3, dir="OpenImages",
         how="Open Images V7 via the CVDF S3 buckets or FiftyOne",
         note="`fiftyone` can pull a bounded subset, which is what you want -- the "
              "full set is far larger than this project needs."),
    dict(name="coco_test", cls="real", status="skip", order=3, dir="COCO_Test",
         note="DO NOT FETCH. Verified 2026-09-08: 100.0% of its hashes are already "
              "in `coco` (40,657 of 40,661), and 1,489 exported images were "
              "byte-identical duplicates with 987 straddling a split boundary. "
              "download_coco_test.py names test2017.zip but train2017 content was "
              "what got indexed. Investigate before ever running it again."),

    # --- ai_generated ---
    dict(name="flux", cls="ai_generated", status="auto", order=4, dir="FLUX",
         hf="ash12321/flux-1-dev-generated-10k", script="download_flux.py",
         note="flux_topup re-streamed this same repo from index 0 with no offset, "
              "producing 2,500 exact duplicates. Use an offset or skip the top-up."),
    dict(name="midjourney_dalle", cls="ai_generated", status="auto", order=4, dir="Midjourney_DALLE",
         hf="ehristoforu/midjourney-images + ehristoforu/dalle-3-images",
         script="download_midjourney_dalle.py",
         note="mj_topup re-streamed these from index 0 and added ~2 novel images "
              "out of 1,137. Offset or skip."),
    dict(name="stablediffusion", cls="ai_generated", status="auto", order=4, dir="StableDiffusion",
         hf="poloclub/diffusiondb", script="download_sd_topup.py", size="parts 1-10",
         note="sd_topup and sd_topup2 draw from the same parts; 3,521 duplicates."),
    dict(name="synthbuster", cls="ai_generated", status="auto", order=4, dir="Synthbuster",
         url="https://zenodo.org/api/records/10066460", size="~5 GB",
         note="Nine generators, Zenodo."),
    dict(name="stylegan", cls="ai_generated", status="manual", order=4, dir="StyleGAN",
         how="was huggan/fake-faces",
         note="That repo returned 401 on 2026-09-08 -- gated or removed. Find an "
              "alternative StyleGAN2/3 FFHQ dump, and note it shares FFHQ's source "
              "images (2 cross-class near-duplicates were found)."),
]

BY_NAME = {s["name"]: s for s in SOURCES}


def human(cmd):
    return " ".join(cmd)


def fetch_http(src, dest, yes):
    url, name = src["url"], src["name"]
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, os.path.basename(url.split("?")[0]))
    if not yes:
        print(f"\n  {name}: {url}\n  -> {out}   ({src.get('size','size unknown')})")
        if input("  proceed? [y/N] ").strip().lower() != "y":
            print("  skipped"); return False
    # -C - resumes a partial file; -L follows the redirects both hosts use.
    cmd = ["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5",
           "-o", out, url]
    print(f"  $ {human(cmd)}")
    return subprocess.call(cmd) == 0


def fetch_script(src, yes):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), src["script"])
    if not os.path.exists(path):
        print(f"  missing script: {path}"); return False
    cmd = [sys.executable, path]
    if not yes:
        print(f"\n  {src['name']}: {human(cmd)}")
        if input("  proceed? [y/N] ").strip().lower() != "y":
            print("  skipped"); return False
    return subprocess.call(cmd) == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="source name (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--all-auto", action="store_true", help="fetch every 'auto' source")
    ap.add_argument("--yes", action="store_true", help="do not ask before each fetch")
    args = ap.parse_args()

    if args.list or not (args.source or args.plan or args.all_auto):
        mark = {"auto": "auto  ", "gated": "GATED ", "manual": "MANUAL", "skip": "SKIP  "}
        print(f"\n  {'source':<18}{'class':<15}{'status':<8}{'size':<10}{'on disk':<12}how")
        print("  " + "-" * 96)
        for s in sorted(SOURCES, key=lambda x: (x["order"], x["name"])):
            how = s.get("url") or s.get("hf") or s.get("how") or ""
            d = os.path.join(DATA, s["cls"], s["dir"])
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            here = f"{n:,} files" if n else "-"
            print(f"  {s['name']:<18}{s['cls']:<15}{mark[s['status']]:<8}"
                  f"{s.get('size',''):<10}{here:<12}{how[:34]}")
        n = sum(1 for s in SOURCES if s["status"] == "auto")
        print(f"\n  {n} of {len(SOURCES)} need no human. Run --plan for the order, "
              f"and read the notes: several corpora need their MASKS, which the "
              f"first build did not fetch.\n")
        return

    if args.plan:
        print("\n  Fetch in this order. The ordering is not cosmetic:\n")
        for o, why in [(1, "Manipulations first -- their filenames name the originals you need"),
                       (2, "COCO once, supplying both the real sample and DEFACTO's originals"),
                       (3, "Remaining real corpora"),
                       (4, "AI-generated corpora")]:
            print(f"  {o}. {why}")
            for s in [x for x in SOURCES if x["order"] == o]:
                flag = "" if s["status"] == "auto" else f"  [{s['status'].upper()}]"
                print(f"       - {s['name']}{flag}")
            print()
        print("  After each corpus: index it fully, export with normalise_on_export,")
        print("  then delete the raw files. Pool the metadata and dedup globally")
        print("  BEFORE sampling -- see docs/SALVAGE_PLAN.md Phase 3.\n")
        return

    targets = ([s for s in SOURCES if s["status"] == "auto"] if args.all_auto
               else [BY_NAME[args.source]] if args.source in BY_NAME else None)
    if targets is None:
        print(f"unknown source {args.source!r}; see --list"); sys.exit(2)

    for src in sorted(targets, key=lambda x: x["order"]):
        print(f"\n=== {src['name']} ({src['cls']}) ===")
        if src.get("note"):
            print(f"  note: {src['note']}")
        if src["status"] == "skip":
            print("  refusing: this source is marked SKIP."); continue
        if src["status"] in ("gated", "manual"):
            print(f"  needs a human: {src.get('how','')}"); continue
        # Config files name these directories explicitly and inconsistently
        # (OpenForensics, Places365, FLUX, COCO_Test), so no case transform
        # works -- the mapping has to be spelled out or the pipeline will not
        # find the data it just downloaded.
        dest = os.path.join(DATA, src["cls"], src["dir"])
        ok = fetch_script(src, args.yes) if src.get("script") else fetch_http(src, dest, args.yes)
        print(f"  {'done' if ok else 'FAILED / skipped'}")


if __name__ == "__main__":
    main()
