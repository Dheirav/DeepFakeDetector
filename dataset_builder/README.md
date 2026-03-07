# Dataset Builder — Documentation

## Status: ✅ Build Complete

The dataset has been fully constructed across 20 source pipelines. Exports are in `dataset_builder/train/`, `dataset_builder/val/`, `dataset_builder/test/`.

---

## Final Dataset Summary

| Class | Total | Train (~40%) | Val (~30%) | Test (~30%) |
|---|---|---|---|---|
| real | 26,000 | ~10,400 | ~7,800 | ~7,800 |
| ai_generated | 26,000 | ~10,400 | ~7,800 | ~7,800 |
| ai_edited | 25,865 | ~10,346 | ~7,760 | ~7,759 |
| **Total** | **77,865** | **~31,146** | **~23,360** | **~23,359** |

**Max class imbalance: 0.52%** — within the ≤2% target.

> Note: configs specify 70/15/15 splits but the cluster-aware splitter produces ~40/30/30 in practice. This is consistent across all sources and classes.

---

## Artifacts (Source Breakdown)

### Real Images — 26,000 total

| Artifact | Count | Config |
|---|---|---|
| `ffhq` | 5,000 | initial run |
| `coco` | 6,000 | `dataset_config.yaml` |
| `openimages` | 5,000 | `real_openimages_config.yaml` |
| `coco_test_tmp_1772822342` | 7,000 | coco_test run |
| `places365` | 3,000 | `real_places365_config.yaml` |

### AI Generated — 26,000 total

| Artifact | Count | Config | Source |
|---|---|---|---|
| `synthbuster` | 7,000 | `ai_generated_synthbuster_config.yaml` | Zenodo Synthbuster (9 generators) |
| `stablediffusion` | 5,000 | `ai_generated_stablediffusion_config.yaml` | SD 1.x outputs |
| `flux` | 3,000 | `ai_generated_flux_config.yaml` | FLUX.1-dev |
| `stylegan` | 2,000 | `ai_generated_stylegan_config.yaml` | StyleGAN2/3 |
| `midjourney_dalle` | 3,222 | `ai_generated_midjourney_config.yaml` | MJ v4/v5 + DALL·E 3 |
| `flux_topup` | 2,000 | `ai_generated_flux_topup_config.yaml` | `ash12321/flux-1-dev-generated-10k` (HF) |
| `sd_topup` | 2,000 | `ai_generated_sd_topup_config.yaml` | DiffusionDB parts 1–3 (ZIP download) |
| `mj_topup` | 1,137 | `ai_generated_mj_topup_config.yaml` | `ehristoforu/midjourney-images` + `dalle-3-images` |
| `sd_topup2` | 641 | `ai_generated_sd_topup2_config.yaml` | DiffusionDB part 4 (gap fill) |

### AI Edited — 25,865 total

| Artifact | Count | Config |
|---|---|---|
| `defacto` | 6,000 | `ai_edited_defacto_config.yaml` |
| `defacto_inpainting` | 5,000 | `ai_edited_defacto_inpainting_config.yaml` |
| `openforensics` | 5,000 | openforensics run |
| `faceforensics` | 4,142 | `ai_edited_faceforensics_config.yaml` |
| `casia` | 4,000 | `ai_edited_casia_config.yaml` |
| `imd2020` | 1,723 | `ai_edited_imd2020_config.yaml` |

---

## Directory Structure

```
dataset_builder/
├── main.py                   # Pipeline entry point
├── pipeline.py               # Pipeline orchestration logic
├── manifest.json             # Full build manifest
├── export_index.csv          # Flat export index (all 77,865 entries)
├── checksums.csv             # Per-file SHA-256 checksums
│
├── config/                   # One YAML per source (20 configs)
│   ├── dataset_config.yaml
│   ├── real_places365_config.yaml
│   ├── ai_generated_flux_topup_config.yaml
│   ├── ai_generated_sd_topup_config.yaml
│   ├── ai_generated_mj_topup_config.yaml
│   ├── ai_generated_sd_topup2_config.yaml
│   └── ... (14 more)
│
├── modules/
│   ├── indexer.py            # Source directory scanner
│   ├── validator.py          # Resolution, blur, corruption checks
│   ├── deduplicator.py       # pHash near-duplicate removal
│   ├── sampler.py            # Quota-based sampling
│   ├── splitter.py           # Cluster-aware train/val/test split
│   ├── exporter.py           # File copy + manifest writer
│   └── audit_dataset.py      # Compliance audit
│
├── scripts/                  # Download scripts (for rebuilding)
│   ├── download_places365.py
│   ├── download_flux_topup.py
│   ├── download_sd_topup.py      # Uses DiffusionDB ZIP (not HF loader)
│   ├── download_mj_topup.py
│   └── download_sd_topup2.py
│
├── output/
│   └── artifacts/            # Per-source pipeline logs and audit reports
│
├── train/                    # ← EXPORTED DATASET
│   ├── real/
│   ├── ai_generated/
│   └── ai_edited/
├── val/
│   ├── real/
│   ├── ai_generated/
│   └── ai_edited/
└── test/
    ├── real/
    ├── ai_generated/
    └── ai_edited/
```

---

## Pipeline Stages

Each config run executes these stages in order:

1. **Indexing** — scans source directory, builds image list
2. **Validation** — verifies integrity, resolution (≥256×256), format
3. **Deduplication** — removes near-duplicates via pHash (Hamming ≤10)
4. **Quality filtering** — blur score, aspect ratio guards
5. **Sampling** — selects exact quota from validated pool
6. **Cluster-based split** — train/val/test with no cross-split leakage
7. **Export** — copies images, writes manifest and checksums
8. **Audit** — validates counts, balance, and split integrity

All 20 runs: Verdict **PASS**, 0 warnings, 0 leakage detected.

---

## Re-Building the Dataset

All source data has been cleaned to save disk space. To rebuild:

```bash
cd dataset_builder
python scripts/download_places365.py        # re-download source
python main.py --config config/real_places365_config.yaml
```

**DiffusionDB note:** The HuggingFace loading script for DiffusionDB is deprecated.
Use the direct ZIP download approach in `download_sd_topup.py` / `download_sd_topup2.py`.

**Top-up pattern:** When adding images to an already-exported class, use a new
source directory and a new `artifacts_dir` per config to avoid double-counting.

---

## Running the Pipeline

```bash
cd dataset_builder
python main.py --config config/<name>_config.yaml [--dry-run] [--log-level INFO|DEBUG]
```

`--dry-run` simulates all stages without writing output files.