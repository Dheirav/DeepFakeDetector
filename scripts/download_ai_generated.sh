#!/usr/bin/env bash
# =============================================================================
# Download script for AI-Generated dataset sources
# Run from the project root:
#   bash scripts/download_ai_generated.sh
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_GEN_DIR="$PROJECT_ROOT/data_sources/ai_generated"

# Redirect HuggingFace cache away from home dir (avoids filling WSL home partition)
# Cache goes to a temp dir inside the project; deleted after each download.
HF_CACHE_DIR="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_HOME="$HF_CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE"

echo "Project root: $PROJECT_ROOT"
echo "AI generated dir: $AI_GEN_DIR"
echo "HuggingFace cache: $HF_CACHE_DIR (will be cleared after each source)"

# Activate venv if available
if [ -f "$PROJECT_ROOT/venv-linux/bin/activate" ]; then
  source "$PROJECT_ROOT/venv-linux/bin/activate"
fi

# =============================================================================
# SOURCE 1 — Synthbuster (DALL·E 2/3, Midjourney v5, Firefly, SDXL, SD, GLIDE)
# Target:   7,000 sampled  |  Available: 9,000 (1k × 9 generators)
# Size:     12.4 GB
# License:  CC BY-NC-SA 4.0
# URL:      https://zenodo.org/records/10066460
# =============================================================================
download_synthbuster() {
  local OUT_DIR="$AI_GEN_DIR/Synthbuster"
  local TMP_DIR="/tmp/synthbuster_download"
  local ZIP="$TMP_DIR/synthbuster.zip"

  echo ""
  echo "=== Synthbuster ==="
  echo "Downloading 12.4 GB from Zenodo..."

  mkdir -p "$TMP_DIR"
  wget -c --show-progress \
    "https://zenodo.org/records/10066460/files/synthbuster.zip" \
    -O "$ZIP"

  echo "Extracting..."
  unzip -q "$ZIP" -d "$TMP_DIR"

  echo "Flattening into $OUT_DIR ..."
  mkdir -p "$OUT_DIR"
  # Each subdir is named after the generator. Prefix filenames to preserve source.
  find "$TMP_DIR/synthbuster" -name "*.png" | while IFS= read -r f; do
    generator=$(basename "$(dirname "$f")")
    dest="$OUT_DIR/${generator}__$(basename "$f")"
    if [ ! -f "$dest" ]; then
      cp "$f" "$dest"
    fi
  done

  echo "Cleaning temp files..."
  rm -rf "$TMP_DIR"
  echo "Synthbuster: done. Files in $OUT_DIR"
}

# =============================================================================
# SOURCE 2 — DiffusionDB / Stable Diffusion 1.x
# Target:   5,000 sampled  |  Available: 14M+ images
# License:  CC0 1.0 (open)
# URL:      https://huggingface.co/datasets/poloclub/diffusiondb
# =============================================================================
download_diffusiondb() {
  local OUT_DIR="$AI_GEN_DIR/StableDiffusion"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=== DiffusionDB (Stable Diffusion 1.x) ==="
  # Download 2x the pipeline target (10,000 raw → 5,000 sampled after filtering)
  echo "Downloading 10,000 images via direct part-zip download (huggingface_hub)..."

  pip install -q huggingface-hub pillow

  python3 - <<'PYEOF'
import os, sys, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

out_dir = Path(os.environ.get("SD_OUT_DIR", ""))
cache_dir = Path(os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_diffusiondb"))
cache_dir.mkdir(parents=True, exist_ok=True)
raw_target = 10000

existing = len(list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg")))
if existing >= raw_target:
    print(f"Already have {existing} images, skipping.")
    sys.exit(0)

saved = 0
part = 1
while (existing + saved) < raw_target:
    part_name = f"images/part-{part:06d}.zip"
    print(f"  Fetching {part_name} ({existing + saved}/{raw_target})...")
    try:
        zip_path = hf_hub_download(
            repo_id="poloclub/diffusiondb",
            repo_type="dataset",
            filename=part_name,
            cache_dir=str(cache_dir),
        )
    except Exception as e:
        print(f"  Could not fetch part {part}: {e}")
        part += 1
        if part > 20:
            break
        continue
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".png"):
                continue
            dest = out_dir / f"sd1x_p{part:04d}_{Path(name).name}"
            if not dest.exists():
                dest.write_bytes(zf.read(name))
                saved += 1
            if (existing + saved) >= raw_target:
                break
    Path(zip_path).unlink(missing_ok=True)
    part += 1

print(f"Done. Saved {saved} new images ({existing + saved} total).")
PYEOF
}
download_synthbuster
SD_OUT_DIR="$AI_GEN_DIR/StableDiffusion" download_diffusiondb
echo "Clearing HF cache after DiffusionDB..."
rm -rf "$HF_CACHE_DIR/datasets/poloclub___diffusion_db" "$HF_CACHE_DIR/datasets/downloads" 2>/dev/null || true
# SOURCE 3 — JourneyDB / Midjourney v4/v5
# Target:   5,000 sampled  |  Available: 4.4M images
# License:  Custom Terms of Usage (research permitted)
# STEP 1:   Fill the form at https://journeydb.github.io (Downloads section)
# STEP 2:   Accept the HuggingFace dataset terms at:
#           https://huggingface.co/datasets/JourneyDB/JourneyDB
# STEP 3:   Run this script (requires HF login: `huggingface-cli login`)
# =============================================================================
download_journeydb() {
  local OUT_DIR="$AI_GEN_DIR/Midjourney_DALLE"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=== JourneyDB (Midjourney v4/v5) ==="

  existing=$(find "$OUT_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.webp" 2>/dev/null | wc -l)
  if [ "$existing" -ge 5000 ]; then
    echo "Already have $existing images, skipping."
    return
  fi

  # Check HuggingFace login
  if ! python3 -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
    echo ""
    echo "  ⚠  JourneyDB requires HuggingFace login."
    echo "     1. Fill form: https://journeydb.github.io"
    echo "     2. Accept dataset terms: https://huggingface.co/datasets/JourneyDB/JourneyDB"
    echo "     3. Run: huggingface-cli login"
    echo "     4. Then re-run this script."
    echo "  Skipping for now."
    return
  fi

  pip install -q datasets pillow

  python3 - <<'PYEOF'
import os
from pathlib import Path
from datasets import load_dataset

out_dir = Path(os.environ.get("MJ_OUT_DIR", ""))
raw_target = 10000  # 2x pipeline sample target of 5,000
existing = len(list(out_dir.glob("*.*")))
if existing >= raw_target:
    print(f"Already have {existing} images, skipping.")
    exit(0)

print("Loading JourneyDB (streaming)...")
ds = load_dataset("JourneyDB/JourneyDB", split="train", streaming=True)
saved = 0
for i, example in enumerate(ds):
    dest = out_dir / f"mj_{i:07d}.jpg"
    if not dest.exists():
        img = example.get("image") or example.get("img")
        if img:
            img.save(dest, quality=95)
            saved += 1
    if (existing + saved) >= raw_target:
        break
    if i % 500 == 0:
        print(f"  Saved {existing + saved}/{raw_target}...")
print(f"Done. Saved {saved} new images.")
PYEOF
}
MJ_OUT_DIR="$AI_GEN_DIR/Midjourney_DALLE" download_journeydb
echo "Clearing HF cache after JourneyDB..."
rm -rf "$HF_CACHE_DIR/datasets/JourneyDB*" "$HF_CACHE_DIR/datasets/downloads" 2>/dev/null || true
# =============================================================================
# SOURCE 4 — StyleGAN2 fake faces (GAN baseline)
# Target:   2,000 sampled  |  Available: 70,000 images
# License:  CC0 / research use
# URL:      https://huggingface.co/datasets/huggan/fake-faces
# =============================================================================
download_stylegan() {
  local OUT_DIR="$AI_GEN_DIR/StyleGAN"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=== StyleGAN2 fake faces (huggan/fake-faces) ==="

  existing=$(find "$OUT_DIR" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
  if [ "$existing" -ge 2000 ]; then
    echo "Already have $existing images (raw target 4,000), skipping."
    return
  fi

  pip install -q datasets pillow

  python3 - <<'PYEOF'
import os
from pathlib import Path
from datasets import load_dataset

out_dir = Path(os.environ.get("SG_OUT_DIR", ""))
raw_target = 4000  # 2x pipeline sample target of 2,000
existing = len(list(out_dir.glob("*.*")))
if existing >= raw_target:
    exit(0)

print("Loading huggan/fake-faces (streaming)...")
ds = load_dataset("huggan/fake-faces", split="train", streaming=True)
saved = 0
for i, example in enumerate(ds):
    dest = out_dir / f"stylegan2_{i:06d}.jpg"
    if not dest.exists():
        example["image"].save(dest, quality=95)
        saved += 1
    if (existing + saved) >= raw_target:
        break
    if i % 200 == 0:
        print(f"  Saved {existing + saved}/{raw_target}...")
print(f"Done. Saved {saved} new images.")
PYEOF
}
SG_OUT_DIR="$AI_GEN_DIR/StyleGAN" download_stylegan
echo "Clearing HF cache after StyleGAN..."
rm -rf "$HF_CACHE_DIR/datasets/huggan*" "$HF_CACHE_DIR/datasets/downloads" 2>/dev/null || true
# =============================================================================
# SOURCE 5 — FLUX.1 generated images
# Target:   3,000 sampled
# License:  Varies by dataset (check before use)
# Instructions:
#   Search https://huggingface.co/datasets?search=flux+generated for available
#   community datasets. Recommended approach:
#     1. Find a dataset with photorealistic FLUX.1 outputs
#     2. Replace DATASET_NAME below with the actual dataset identifier
#     3. Run this function
# =============================================================================
download_flux() {
  local OUT_DIR="$AI_GEN_DIR/FLUX"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=== FLUX.1 Generated Images ==="

  existing=$(find "$OUT_DIR" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
  if [ "$existing" -ge 6000 ]; then
    echo "Already have $existing images (raw target 6,000), skipping."
    return
  fi

  # ---------------------------------------------------------------------------
  # ACTION REQUIRED: Find a suitable FLUX.1 dataset on HuggingFace first.
  # Search: https://huggingface.co/datasets?search=flux+generated
  # Then set FLUX_DATASET_NAME below before running.
  # ---------------------------------------------------------------------------
  FLUX_DATASET_NAME="${FLUX_DATASET_NAME:-}"

  if [ -z "$FLUX_DATASET_NAME" ]; then
    echo ""
    echo "  ⚠  FLUX dataset not yet configured."
    echo "     1. Search: https://huggingface.co/datasets?search=flux+generated+images"
    echo "     2. Pick a photorealistic dataset (no LoRA stylized art)"
    echo "     3. Set: export FLUX_DATASET_NAME=<owner/dataset-name>"
    echo "     4. Re-run: bash scripts/download_ai_generated.sh"
    echo "  Skipping FLUX for now."
    return
  fi

  pip install -q datasets pillow

  python3 - <<PYEOF
import os
from pathlib import Path
from datasets import load_dataset

out_dir = Path("$OUT_DIR")
dataset_name = "$FLUX_DATASET_NAME"
raw_target = 6000  # 2x pipeline sample target of 3,000
existing = len(list(out_dir.glob("*.*")))
if existing >= raw_target:
    exit(0)

print(f"Loading {dataset_name} (streaming)...")
ds = load_dataset(dataset_name, split="train", streaming=True)
saved = 0
for i, example in enumerate(ds):
    img = example.get("image") or example.get("img") or example.get("pixel_values")
    if img is None:
        continue
    dest = out_dir / f"flux_{i:06d}.png"
    if not dest.exists():
        img.save(dest)
        saved += 1
    if (existing + saved) >= raw_target:
        break
    if i % 300 == 0:
        print(f"  Saved {existing + saved}/{raw_target}...")
print(f"Done. Saved {saved} new images.")
PYEOF
}
download_flux
echo "Clearing HF cache after FLUX..."
rm -rf "$HF_CACHE_DIR/datasets/downloads" 2>/dev/null || true
# =============================================================================
echo ""
echo "============================================================"
echo " Download Summary (raw on disk vs pipeline sample target)"
echo "============================================================"
printf "  %-20s  %8s  %s\n" "Source" "On Disk" "Pipeline Target"
printf "  %-20s  %8s  %s\n" "------" "-------" "---------------"
for src in Synthbuster StableDiffusion Midjourney_DALLE StyleGAN FLUX; do
  count=$(find "$AI_GEN_DIR/$src" -name "*.jpg" -o -name "*.png" -o -name "*.webp" 2>/dev/null | wc -l)
  case "$src" in
    Synthbuster)      target="7,000" ;;
    StableDiffusion)  target="5,000" ;;
    Midjourney_DALLE) target="5,000" ;;
    StyleGAN)         target="2,000" ;;
    FLUX)             target="3,000" ;;
  esac
  printf "  %-20s  %8d  %s\n" "$src" "$count" "$target"
done
echo ""
echo "Next steps after all sources are populated:"
echo "  cd dataset_builder"
echo "  python main.py --config config/ai_generated_synthbuster_config.yaml  --artifacts-dir output/artifacts/synthbuster"
echo "  python main.py --config config/ai_generated_stablediffusion_config.yaml --artifacts-dir output/artifacts/stablediffusion"
echo "  python main.py --config config/ai_generated_midjourney_config.yaml   --artifacts-dir output/artifacts/midjourney"
echo "  python main.py --config config/ai_generated_flux_config.yaml         --artifacts-dir output/artifacts/flux"
echo "  python main.py --config config/ai_generated_stylegan_config.yaml     --artifacts-dir output/artifacts/stylegan"
echo "  # Then merge:"
echo "  python tools/merge_exports.py \\"
echo "    --artifacts-dirs output/artifacts/synthbuster output/artifacts/stablediffusion \\"
echo "    output/artifacts/midjourney output/artifacts/flux output/artifacts/stylegan \\"
echo "    --out-dir output/merged_ai_generated"
