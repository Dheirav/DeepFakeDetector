#!/usr/bin/env python3
"""
Production-grade image upscaling tool for the deepfake detection dataset pipeline.

Upscales low-resolution images (e.g. Tiny-ImageNet 64×64) to a target resolution
suitable for training (256×256 or 512×512).

PRIMARY BACKEND  : Real-ESRGAN (neural super-resolution, research standard)
FALLBACK BACKEND : PIL Lanczos (always available, zero dependencies)

Key features:
  - Auto-detects CUDA GPU; falls back to CPU gracefully
  - Resume support: skips already-upscaled outputs
  - Multiprocessing for Lanczos path; batched GPU for ESRGAN path
  - Lossless PNG output by default (or high-quality JPEG with --format jpeg)
  - Patches export_index.csv and split_index.csv with new dimensions/paths
  - Dry-run mode for validation before committing
  - Structured logging + per-run log file

Usage examples:
  # Upscale all exported ImageNet images (4x: 64 -> 256) using Real-ESRGAN if available
  python tools/upscale_images.py \
      --input-dir train/real val/real test/real \
      --output-dir train_upscaled/real val_upscaled/real test_upscaled/real \
      --scale 4 --format png

  # Lanczos fallback only
  python tools/upscale_images.py \
      --input-dir train/real val/real test/real \
      --output-dir train_upscaled/real val_upscaled/real test_upscaled/real \
      --scale 4 --backend lanczos

  # In-place (overwrites originals, saves a .bak copy)
  python tools/upscale_images.py \
      --input-dir train/real --scale 4 --in-place

  # Patch CSV manifests after upscaling
  python tools/upscale_images.py \
      --patch-csv output/artifacts/imagenet/export_index.csv \
                  output/artifacts/imagenet/split_index.csv \
      --input-dir train/real val/real test/real \
      --output-dir train_upscaled/real val_upscaled/real test_upscaled/real \
      --scale 4
"""
import argparse
import csv
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
ESRGAN_MODEL_NAME_4X = "RealESRGAN_x4plus"
ESRGAN_MODEL_NAME_2X = "realesr-animatesr_x2"   # lighter 2x model
LOG_DIR = Path(__file__).parent.parent / "logs"
SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(log_level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"upscale_{ts}.log"

    logger = logging.getLogger("upscaler")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
def detect_backend(preferred: str, scale: int, logger: logging.Logger) -> str:
    """
    Returns the resolved backend to use: 'realesrgan' or 'lanczos'.
    Only attempts Real-ESRGAN if explicitly preferred or 'auto'.
    """
    if preferred == "lanczos":
        logger.info("Backend: Lanczos (explicitly requested)")
        return "lanczos"

    # Try Real-ESRGAN
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet          # noqa: F401
        from realesrgan import RealESRGANer                     # noqa: F401
        logger.info("Backend: Real-ESRGAN (available)")
        return "realesrgan"
    except ImportError:
        if preferred == "realesrgan":
            logger.error(
                "Real-ESRGAN requested but not installed.\n"
                "  Install with: pip install realesrgan basicsr facexlib gfpgan\n"
                "Falling back to Lanczos."
            )
        else:
            logger.info(
                "Real-ESRGAN not installed — using Lanczos. "
                "For neural super-resolution: pip install realesrgan basicsr"
            )
        return "lanczos"


# ---------------------------------------------------------------------------
# Real-ESRGAN upscaler
# ---------------------------------------------------------------------------
def build_realesrgan_upscaler(scale: int, logger: logging.Logger):
    """
    Instantiates a RealESRGANer with the appropriate model for the scale factor.
    Downloads model weights automatically on first use.
    """
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Real-ESRGAN device: {device}")

    if scale == 4:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=4,
        )
        model_path = SCRIPT_DIR / "weights" / f"{ESRGAN_MODEL_NAME_4X}.pth"
        netscale = 4
    elif scale == 2:
        # Lighter RRDB 2x
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=2,
        )
        model_path = SCRIPT_DIR / "weights" / "RealESRGAN_x2plus.pth"
        netscale = 2
    else:
        raise ValueError(
            f"Real-ESRGAN supports scale=2 or scale=4 only. Got: {scale}. "
            "Use --backend lanczos for arbitrary scales."
        )

    # Download weights if missing
    model_path = _ensure_esrgan_weights(model_path, netscale, logger)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=str(model_path),
        model=model,
        tile=256,          # tile size for large images; avoids OOM on GPU
        tile_pad=10,
        pre_pad=0,
        half=device == "cuda",   # fp16 on GPU for speed, fp32 on CPU for stability
        device=device,
    )
    return upsampler


def _ensure_esrgan_weights(model_path: Path, scale: int, logger: logging.Logger) -> Path:
    """Downloads Real-ESRGAN weights if not already on disk."""
    import urllib.request

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        logger.info(f"Model weights found: {model_path}")
        return model_path

    url_map = {
        4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    }
    url = url_map.get(scale)
    if not url:
        raise RuntimeError(f"No known weight URL for scale={scale}")

    logger.info(f"Downloading Real-ESRGAN weights ({scale}x) from: {url}")
    logger.info(f"Saving to: {model_path}")
    try:
        urllib.request.urlretrieve(url, str(model_path), reporthook=_download_progress)
        print()  # newline after progress
        logger.info("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model weights: {e}") from e
    return model_path


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        print(f"\r  [{'█' * bar}{'░' * (50 - bar)}] {pct:.1f}%", end="", flush=True)


def upscale_realesrgan(
    upsampler,
    img_path: Path,
    out_path: Path,
    out_format: str,
    logger: logging.Logger,
) -> bool:
    """Upscale a single image using Real-ESRGAN. Returns True on success."""
    import cv2
    import numpy as np

    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV could not read: {img_path}")

        output, _ = upsampler.enhance(img, outscale=None)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_format == "png":
            cv2.imwrite(str(out_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:  # jpeg
            cv2.imwrite(str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        logger.warning(f"ESRGAN failed on {img_path.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Lanczos upscaler (worker for multiprocessing)
# ---------------------------------------------------------------------------
def _lanczos_worker(args: Tuple) -> Tuple[str, bool, str]:
    """
    Multiprocessing-safe worker. Returns (out_path_str, success, error_msg).
    """
    img_path_str, out_path_str, scale, target_w, target_h, out_format = args
    try:
        from PIL import Image
        img_path = Path(img_path_str)
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            new_w = target_w if target_w else orig_w * scale
            new_h = target_h if target_h else orig_h * scale
            upscaled = img.resize((new_w, new_h), Image.LANCZOS)

            if out_format == "png":
                upscaled.save(str(out_path), format="PNG", compress_level=1)
            else:
                upscaled.save(str(out_path), format="JPEG", quality=95, subsampling=0)
        return out_path_str, True, ""
    except Exception as e:
        return out_path_str, False, str(e)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------
def collect_image_pairs(
    input_dirs: List[Path],
    output_dirs: List[Path],
    out_ext: str,
    in_place: bool,
    skip_existing: bool,
    logger: logging.Logger,
) -> List[Tuple[Path, Path]]:
    """
    Build (src, dst) pairs. Skips already-upscaled outputs when skip_existing=True.
    """
    pairs = []
    for in_dir, out_dir in zip(input_dirs, output_dirs):
        if not in_dir.exists():
            logger.warning(f"Input dir does not exist, skipping: {in_dir}")
            continue
        for img_file in sorted(in_dir.rglob("*")):
            if img_file.suffix.lower() not in SUPPORTED_EXTS:
                continue
            if in_place:
                dest = img_file.with_suffix(out_ext)
            else:
                # preserve relative sub-structure inside in_dir
                rel = img_file.relative_to(in_dir)
                dest = out_dir / rel.with_suffix(out_ext)
            if skip_existing and dest.exists():
                continue
            pairs.append((img_file, dest))
    logger.info(f"Collected {len(pairs)} images to process.")
    return pairs


# ---------------------------------------------------------------------------
# In-place backup
# ---------------------------------------------------------------------------
def backup_original(img_path: Path, logger: logging.Logger):
    bak = img_path.with_suffix(".bak" + img_path.suffix)
    if not bak.exists():
        shutil.copy2(str(img_path), str(bak))


# ---------------------------------------------------------------------------
# CSV manifest patching
# ---------------------------------------------------------------------------
def patch_csv_manifests(
    csv_paths: List[Path],
    old_to_new: dict,   # {old_export_path_str: new_export_path_str}
    logger: logging.Logger,
):
    """
    Updates 'export_path', 'width', 'height' in CSV manifests to reflect
    upscaled files. Preserves all other columns unchanged.
    """
    from PIL import Image

    dim_cache = {}

    def get_dims(p: str) -> Tuple[int, int]:
        if p not in dim_cache:
            try:
                with Image.open(p) as img:
                    dim_cache[p] = img.size
            except Exception:
                dim_cache[p] = (0, 0)
        return dim_cache[p]

    for csv_path in csv_paths:
        if not csv_path.exists():
            logger.warning(f"CSV not found, skipping patch: {csv_path}")
            continue

        tmp_path = csv_path.with_suffix(".patching.csv")
        patched = 0

        with open(csv_path, "r", newline="") as fin, \
             open(tmp_path, "w", newline="") as fout:
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames or []
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                ep = row.get("export_path", "")
                if ep in old_to_new:
                    new_path = old_to_new[ep]
                    row["export_path"] = new_path
                    w, h = get_dims(new_path)
                    if w and "width" in row:
                        row["width"] = str(w)
                    if h and "height" in row:
                        row["height"] = str(h)
                    # clear the low_resolution flag if now valid
                    if "quality_flag" in row and "low_resolution" in row.get("quality_flag", ""):
                        flags = [f for f in row["quality_flag"].split(",")
                                 if f.strip() != "low_resolution"]
                        row["quality_flag"] = ",".join(flags)
                    if "resolution_ok" in row:
                        row["resolution_ok"] = "True"
                    patched += 1
                writer.writerow(row)

        # Atomic replace
        tmp_path.replace(csv_path)
        logger.info(f"Patched {patched} rows in: {csv_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_upscale(args, logger: logging.Logger):
    start = time.time()

    # --- Resolve scale and output extension ---
    scale = args.scale
    out_ext = ".png" if args.format == "png" else ".jpg"

    # --- Resolve input/output dirs ---
    input_dirs = [Path(d).resolve() for d in args.input_dir]
    if args.in_place:
        output_dirs = input_dirs
    elif args.output_dir:
        if len(args.output_dir) != len(args.input_dir):
            logger.error(
                f"--output-dir count ({len(args.output_dir)}) must match "
                f"--input-dir count ({len(args.input_dir)})"
            )
            sys.exit(1)
        output_dirs = [Path(d).resolve() for d in args.output_dir]
    else:
        # Default: add '_upscaled' suffix to each input dir name
        output_dirs = [d.parent / (d.name + "_upscaled") for d in input_dirs]
        logger.info(f"No --output-dir specified. Using: {output_dirs}")

    # --- Collect pairs ---
    pairs = collect_image_pairs(
        input_dirs, output_dirs,
        out_ext=out_ext,
        in_place=args.in_place,
        skip_existing=not args.overwrite,
        logger=logger,
    )

    if not pairs:
        logger.info("No images to process. All already upscaled or dirs empty.")
        return

    if args.dry_run:
        logger.info(f"[DRY RUN] Would upscale {len(pairs)} images. Exiting.")
        for src, dst in pairs[:5]:
            logger.info(f"  {src} -> {dst}")
        if len(pairs) > 5:
            logger.info(f"  ... and {len(pairs) - 5} more.")
        return

    # --- Backup originals if in-place ---
    if args.in_place and args.backup:
        logger.info("Backing up originals before in-place upscale...")
        for src, _ in pairs:
            backup_original(src, logger)

    # --- Detect and run backend ---
    backend = detect_backend(args.backend, scale, logger)
    old_to_new = {}   # for CSV patching: old_path -> new_path
    failed = []

    if backend == "realesrgan":
        logger.info(f"Running Real-ESRGAN {scale}x upscaling on {len(pairs)} images...")
        upsampler = build_realesrgan_upscaler(scale, logger)
        try:
            from tqdm import tqdm
            iter_pairs = tqdm(pairs, unit="img", desc="Upscaling (ESRGAN)")
        except ImportError:
            iter_pairs = pairs

        for idx, (src, dst) in enumerate(iter_pairs, 1):
            dst_final = dst.with_suffix(out_ext)
            ok = upscale_realesrgan(upsampler, src, dst_final, args.format, logger)
            if ok:
                old_to_new[str(src)] = str(dst_final)
            else:
                failed.append(src)
                # Fall back to Lanczos for this image
                result = _lanczos_worker((
                    str(src), str(dst_final), scale,
                    args.target_width, args.target_height, args.format
                ))
                _, ok2, err = result
                if ok2:
                    old_to_new[str(src)] = str(dst_final)
                    logger.info(f"Lanczos fallback succeeded for: {src.name}")
                else:
                    logger.error(f"Both ESRGAN and Lanczos failed for {src.name}: {err}")

    else:  # lanczos
        logger.info(
            f"Running Lanczos {scale}x upscaling on {len(pairs)} images "
            f"using {args.workers} workers..."
        )
        worker_args = [
            (str(src), str(dst.with_suffix(out_ext)),
             scale, args.target_width, args.target_height, args.format)
            for src, dst in pairs
        ]

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        with mp.Pool(processes=args.workers) as pool:
            if use_tqdm:
                results = list(tqdm(
                    pool.imap_unordered(_lanczos_worker, worker_args),
                    total=len(worker_args), unit="img", desc="Upscaling (Lanczos)"
                ))
            else:
                results = pool.map(_lanczos_worker, worker_args)

        for out_path_str, ok, err in results:
            if ok:
                # Find original src for this dst
                orig = next(
                    (str(s) for s, d in pairs
                     if str(d.with_suffix(out_ext)) == out_path_str),
                    None
                )
                if orig:
                    old_to_new[orig] = out_path_str
            else:
                logger.error(f"Failed: {out_path_str} — {err}")
                failed.append(out_path_str)

    # --- Summary ---
    total = len(pairs)
    succeeded = len(old_to_new)
    n_failed = len(failed)
    elapsed = round(time.time() - start, 2)
    logger.info(
        f"\n=== UPSCALE SUMMARY ==="
        f"\n  Total:     {total}"
        f"\n  Succeeded: {succeeded}"
        f"\n  Failed:    {n_failed}"
        f"\n  Backend:   {backend}"
        f"\n  Scale:     {scale}x"
        f"\n  Format:    {args.format.upper()}"
        f"\n  Runtime:   {elapsed}s"
    )
    if n_failed:
        logger.warning(f"Failed images ({n_failed}):")
        for f in failed[:20]:
            logger.warning(f"  {f}")

    # --- Patch CSV manifests ---
    if args.patch_csv and old_to_new:
        csv_paths = [Path(p).resolve() for p in args.patch_csv]
        logger.info(f"Patching {len(csv_paths)} CSV manifest(s)...")
        patch_csv_manifests(csv_paths, old_to_new, logger)
        logger.info("CSV patching complete.")

    logger.info("Upscaling pipeline complete.")
    return succeeded, n_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production-grade image upscaler for deepfake detection dataset pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input / Output ---
    p.add_argument(
        "--input-dir", nargs="+", required=True,
        help="One or more input directories containing images to upscale."
    )
    p.add_argument(
        "--output-dir", nargs="+", default=None,
        help="One or more output directories (must match --input-dir count). "
             "Defaults to <input_dir>_upscaled."
    )
    p.add_argument(
        "--in-place", action="store_true",
        help="Overwrite originals in-place. Use --backup to keep .bak copies."
    )
    p.add_argument(
        "--backup", action="store_true",
        help="When --in-place is set, save .bak copies of originals before overwriting."
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-process even if the output file already exists. Default: skip."
    )

    # --- Scaling ---
    p.add_argument(
        "--scale", type=int, default=4,
        help="Integer scale factor (default: 4 → 64×64 becomes 256×256). "
             "Real-ESRGAN only supports 2 or 4."
    )
    p.add_argument(
        "--target-width", type=int, default=None,
        help="Explicit target width in pixels (overrides --scale for Lanczos)."
    )
    p.add_argument(
        "--target-height", type=int, default=None,
        help="Explicit target height in pixels (overrides --scale for Lanczos)."
    )

    # --- Backend ---
    p.add_argument(
        "--backend", choices=["auto", "realesrgan", "lanczos"], default="auto",
        help="Upscaling backend. 'auto' tries Real-ESRGAN first, falls back to Lanczos."
    )

    # --- Output format ---
    p.add_argument(
        "--format", choices=["png", "jpeg"], default="png",
        help="Output format. 'png' = lossless. 'jpeg' = Q95, smaller files. (default: png)"
    )

    # --- Parallelism ---
    p.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 2),
        help="Number of parallel workers for Lanczos path. Default: CPU count - 2."
    )

    # --- CSV patching ---
    p.add_argument(
        "--patch-csv", nargs="*", default=None,
        help="One or more CSV manifests (export_index.csv, split_index.csv) to patch "
             "with updated export_path, width, height after upscaling."
    )

    # --- Misc ---
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without actually processing any images."
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity."
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logger(args.log_level)

    logger.info("=== Image Upscaler — Deepfake Detection Pipeline ===")
    logger.info(f"Input dirs  : {args.input_dir}")
    logger.info(f"Output dirs : {args.output_dir or '(auto)'}")
    logger.info(f"Scale       : {args.scale}x")
    logger.info(f"Backend     : {args.backend}")
    logger.info(f"Format      : {args.format}")
    logger.info(f"In-place    : {args.in_place}")
    logger.info(f"Workers     : {args.workers}")
    logger.info(f"Dry run     : {args.dry_run}")

    run_upscale(args, logger)


if __name__ == "__main__":
    main()
