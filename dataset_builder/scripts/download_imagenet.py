#!/usr/bin/env python3
"""
Download ImageNet-1k photo subset from HuggingFace for real-image class.

Dataset: ILSVRC/imagenet-1k  (gated — requires free HuggingFace account)

SETUP (one time only):
  1. Visit https://huggingface.co/datasets/ILSVRC/imagenet-1k
     Click "Agree and access repository"
  2. Get your HF token: https://huggingface.co/settings/tokens
  3. Login:
       huggingface-cli login
     or set environment variable:
       export HF_TOKEN=hf_your_token_here

Usage:
    python download_imagenet.py
    python download_imagenet.py --target 4500 --output-path ../../data_sources/real/ImageNet
    python download_imagenet.py --split validation  # faster, 50K images across 1000 classes
"""

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Photo-realistic synset categories to include.
# All ImageNet-1k images are real photographs, but we explicitly select
# synset groups that match DATASET.md goal: "natural objects, tools, animals,
# landscapes" — and avoid any edge cases.
# ---------------------------------------------------------------------------
INCLUDE_SYNSET_GROUPS = {
    # Animals
    "n01440764", "n01443537", "n01484850", "n01491361", "n01494475",  # fish
    "n01496331", "n01498041",
    "n01514668", "n01514859", "n01518878", "n01530575", "n01531178",  # birds
    "n01532829", "n01534433", "n01537544", "n01558993", "n01560419",
    "n01580077", "n01582220", "n01592084", "n01601694", "n01608432",
    "n01614925", "n01616318", "n01622779", "n01629819", "n01630670",
    "n01631663", "n01632458", "n01632777", "n01641577", "n01644373",
    "n01644900",
    "n01664065", "n01665541", "n01667114", "n01667778", "n01669191",  # reptiles
    "n01675722", "n01677366", "n01682714", "n01685808", "n01687978",
    "n01688243", "n01689811", "n01692333", "n01693334", "n01694178",
    "n01695060", "n01697457", "n01698640", "n01704323",
    "n01728572", "n01728920", "n01729322", "n01729977", "n01734418",  # snakes
    "n01735189", "n01737021", "n01739381", "n01740131", "n01742172",
    "n01743086", "n01744401", "n01748264",
    "n01749939", "n01751748", "n01753488", "n01755581", "n01756291",
    # mammals
    "n02056570", "n02058221",  # penguins
    "n02085620", "n02085782", "n02085936", "n02086079", "n02086240",  # dogs
    "n02086646", "n02086910", "n02087046", "n02087394", "n02088094",
    "n02088238", "n02088364", "n02088466", "n02088632", "n02089078",
    "n02089867", "n02089973", "n02090379", "n02090622", "n02090721",
    "n02091032", "n02091134", "n02091244", "n02091467", "n02091635",
    "n02091831", "n02092002", "n02092339", "n02093256", "n02093428",
    "n02093647", "n02093754", "n02093859", "n02093991", "n02094114",
    "n02094258", "n02094433", "n02095314", "n02095570", "n02095889",
    "n02096051", "n02096177", "n02096294", "n02096437", "n02096585",
    "n02097047", "n02097130", "n02097209", "n02097298", "n02097474",
    "n02097658", "n02098105", "n02098286", "n02098413", "n02099267",
    "n02099429", "n02099601", "n02099712", "n02099849", "n02100236",
    "n02100583", "n02100735", "n02100877", "n02101006", "n02101388",
    "n02101556", "n02102040", "n02102177", "n02102318", "n02102480",
    "n02102973",
    "n02104029", "n02104365",
    "n02105056", "n02105162", "n02105251", "n02105412", "n02105505",
    "n02105641", "n02105855", "n02106030", "n02106166", "n02106382",
    "n02106550", "n02106662", "n02107142", "n02107312", "n02107574",
    "n02107683", "n02107908", "n02108000", "n02108089", "n02108422",
    "n02108551", "n02108915", "n02109047", "n02109525", "n02109961",
    "n02110063", "n02110185", "n02110341", "n02110627", "n02110806",
    "n02110958", "n02111129", "n02111277", "n02111500", "n02111889",
    "n02112018", "n02112137", "n02112350", "n02112706", "n02113023",
    "n02113186", "n02113624", "n02113712", "n02113799", "n02113978",
    "n02114367", "n02114548", "n02114712",
    "n02115641", "n02115913",
    "n02117135",
    "n02119022", "n02119789", "n02120079", "n02120505",
    "n02123045", "n02123159", "n02123394", "n02123597", "n02124075",  # cats
    "n02125311", "n02127052",
    "n02128385", "n02128757", "n02128925", "n02129165", "n02129604",  # big cats
    "n02130308",
    "n02132136", "n02133161", "n02134084", "n02134418",  # bears
    "n02137549",
    "n02138441",
    "n02165105", "n02165456", "n02167151", "n02168699", "n02169497",  # insects
    "n02172182", "n02174001", "n02177972", "n02190166", "n02206856",
    "n02219486", "n02226429", "n02229544", "n02231487", "n02233338",
    "n02236044", "n02256656", "n02259212", "n02264363", "n02268443",
    "n02268853", "n02276258", "n02277742", "n02279972", "n02280649",
    "n02281406", "n02281787",
    # vehicles
    "n02701002", "n02704792", "n02708093",  # cars
    "n02727426",
    "n02730930",
    "n02747177",
    "n02769748",
    "n02776631",
    "n02791270",
    "n02793495",
    "n02799071",
    "n02802426",
    "n02814533", "n02814860",  # vehicles
    "n02930766",  # garage
    "n03100240",  # convertible
    "n03417042",  # garbage truck
    "n03425413",  # gas pump
    "n03444034",  # glider
    "n03445777",  # golf ball
    "n03594945",  # jeep
    "n03670208",  # limousine
    "n03777568",  # motor scooter
    "n03785016",  # mountain bike
    "n03796401",  # moving van
    "n03832673",  # notebook
    "n03891332",  # parking meter
    "n03930630",  # pickup truck
    "n03977966",  # police van
    "n04037443",  # racer
    "n04065272",  # recreational vehicle
    "n04147183",  # schooner
    "n04252077",  # snowmobile
    "n04285008",  # sports car
    "n04461696",  # tow truck
    "n04467665",  # trailer truck
    "n04509417",  # unicycle
    "n04552348",  # warplane
    "n04612504",  # yawl
    # natural scenes / objects
    "n02892201", "n02895154", "n02906734", "n02909870", "n02916936",
    "n02950826", "n02963159", "n02966193", "n02966687", "n02971356",
    "n02974003", "n02977058", "n02978881", "n02980441", "n02981792",
    "n02988304", "n02992211", "n02992529", "n02999410",
    "n03000134", "n03000247", "n03000684", "n03014705",
    "n03016953", "n03017168", "n03018349", "n03026506",
    # food
    "n07556406", "n07565083", "n07583066", "n07584110", "n07590611",
    "n07613480", "n07614500", "n07615774", "n07684084", "n07693725",
    "n07695742", "n07697313", "n07697537", "n07711569", "n07714571",
    "n07714990", "n07715103", "n07716358", "n07716906", "n07717410",
    "n07717556", "n07718472", "n07718747", "n07720875", "n07730033",
    "n07734744", "n07742313", "n07745940", "n07747607", "n07749582",
    "n07753113", "n07753275", "n07753592", "n07754684", "n07760859",
    "n07768694",
}

# Use ALL synsets if the set above is too restrictive (imagenet is all real photos)
USE_ALL_SYNSETS = True  # set False to use only the filtered set above


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ImageNet-1k photo subset from HuggingFace"
    )
    parser.add_argument(
        "--output-path",
        default="../../data_sources/real/ImageNet",
        help="Destination directory for images",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=4500,
        help="Number of images to download (default: 4500 — buffer for pipeline filtering)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Which split to stream from. 'validation' is faster (50K images). "
             "'train' has 1.28M images. Default: validation",
    )
    parser.add_argument(
        "--per-class-max",
        type=int,
        default=10,
        help="Max images per synset class to ensure diversity (default: 10)",
    )
    return parser.parse_args()


def check_hf_auth():
    """Check if user is logged in to HuggingFace."""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"  Logged in as: {user['name']}")
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ImageNet-1k Downloader (Real Photos)")
    print(f"  Output directory : {output_dir}")
    print(f"  Split            : {args.split}")
    print(f"  Target           : {args.target} images")
    print(f"  Per-class max    : {args.per_class_max}")
    print("=" * 60)

    # Check existing
    existing = sum(1 for f in output_dir.iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"\n  Existing images: {existing}")
    if existing >= args.target:
        print(f"  Already have {existing} images — nothing to do.")
        return

    # Check HuggingFace login
    print("\n  Checking HuggingFace authentication ...")
    if not check_hf_auth():
        print("\n  ERROR: Not logged in to HuggingFace.")
        print("\n  To fix:")
        print("    1. Visit https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("       Click 'Agree and access repository'")
        print("    2. Get token: https://huggingface.co/settings/tokens")
        print("    3. Run: huggingface-cli login")
        print("       (paste your token when prompted)")
        print("\n  Then re-run this script.")
        sys.exit(1)

    # Load dataset
    print(f"\n  Loading ILSVRC/imagenet-1k ({args.split}) in streaming mode ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split=args.split,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\n  ERROR loading dataset: {e}")
        print("\n  Make sure you have accepted the dataset terms at:")
        print("  https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        sys.exit(1)

    saved = 0
    skipped = 0
    class_counts: dict[int, int] = {}

    print(f"  Streaming and saving images (target: {args.target}) ...\n")

    for i, example in enumerate(ds):
        if saved >= args.target:
            break

        try:
            label = example.get("label", -1)

            # Per-class diversity cap
            if class_counts.get(label, 0) >= args.per_class_max:
                skipped += 1
                continue

            img = example["image"]

            # Convert to RGB
            if img.mode not in ("RGB", "RGBA"):
                skipped += 1
                continue
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # Size filter
            if img.width < 256 or img.height < 256:
                skipped += 1
                continue

            out_path = output_dir / f"imagenet_{saved:06d}.jpg"
            if out_path.exists():
                saved += 1
                class_counts[label] = class_counts.get(label, 0) + 1
                continue

            img.save(out_path, "JPEG", quality=92)
            saved += 1
            class_counts[label] = class_counts.get(label, 0) + 1

            if saved % 200 == 0:
                print(f"  {saved}/{args.target}  (scanned {i+1}, skipped {skipped}, "
                      f"classes covered: {len(class_counts)})")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Warning: skipped example {i}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Done.  {saved} images saved to {output_dir}")
    print(f"  Classes covered : {len(class_counts)}")
    print(f"  Skipped         : {skipped}")
    if saved < args.target:
        print(f"\n  WARNING: Only got {saved} (target {args.target}).")
        print(f"  Try re-running with --split train to draw from 1.28M training images.")
    print(f"{'=' * 60}")
    print("\nNext step — run the pipeline:")
    print("  cd dataset_builder && python main.py --config config/real_imagenet_config.yaml")


if __name__ == "__main__":
    main()
