# Dataset Pipeline Setup Instructions

> **⚠️ Note:** The dataset has already been built and exported.
> This document describes the original setup process and is kept for reference.
> For current dataset stats and re-build instructions see [DATASET.md](DATASET.md) and [dataset_builder/README.md](dataset_builder/README.md).
> To start training immediately, see [QUICKSTART.md](QUICKSTART.md).

Complete guide to set up and run the deepfake detection dataset builder pipeline.

---

## Prerequisites

- Python 3.12+ with virtual environment activated
- Sufficient disk space for raw data and processed dataset
- Downloaded dataset sources (see Data Sources section below)

---

## 1. Directory Structure

The following directories have been created:

```
data_sources/
├── real/
│   ├── FFHQ/              # Place FFHQ dataset images here
│   ├── COCO/              # Place MS COCO real images here
│   ├── OpenImages/        # Place Open Images dataset here
│   └── ImageNet/          # Place ImageNet photos here
├── ai_generated/
│   ├── StyleGAN/          # Place StyleGAN outputs here
│   ├── StableDiffusion/   # Place Stable Diffusion outputs here
│   ├── Midjourney_DALLE/  # Place Midjourney/DALL-E images here
│   └── LAION/             # Place LAION diffusion subset here
└── ai_edited/
    ├── FaceForensics/     # Place FaceForensics++ images here
    ├── ForgeryNet/        # Place ForgeryNet dataset here
    ├── CASIA/             # Place CASIA tampering dataset here
    ├── IMD2020/           # Place IMD2020 dataset here
    └── DEFACTO/           # Place DEFACTO dataset here
```

---

## 2. Data Sources

Download and place images from these sources into the corresponding folders:

### CLASS 0 — REAL IMAGES (Target: 22,000)

1. **FFHQ** (5,000 images)
   - Source: https://github.com/NVlabs/ffhq-dataset
   - Place in: `data_sources/real/FFHQ/`

2. **MS COCO** (6,000 images)
   - Source: https://cocodataset.org/
   - Filter: Real photos only (no cartoons/illustrations)
   - Place in: `data_sources/real/COCO/`

3. **Open Images** (5,000 images)
   - Source: https://storage.googleapis.com/openimages/web/index.html
   - Place in: `data_sources/real/OpenImages/`

4. **ImageNet** (4,000 images)
   - Source: https://www.image-net.org/
   - Filter: Photo subset only (no artistic/synthetic)
   - Place in: `data_sources/real/ImageNet/`

### CLASS 1 — AI GENERATED (Target: 22,000)

1. **StyleGAN** (7,000 images)
   - Generate using StyleGAN2/StyleGAN3 or download pre-generated
   - Place in: `data_sources/ai_generated/StyleGAN/`

2. **Stable Diffusion** (7,000 images)
   - Generate realistic images (no anime/stylized art)
   - Place in: `data_sources/ai_generated/StableDiffusion/`

3. **Midjourney/DALL-E** (4,000 images)
   - Research mirrors or generate photorealistic outputs
   - Place in: `data_sources/ai_generated/Midjourney_DALLE/`

4. **LAION** (4,000 images)
   - Source: https://laion.ai/
   - Place in: `data_sources/ai_generated/LAION/`

### CLASS 2 — AI EDITED (Target: 22,000)

1. **FaceForensics++** (6,000 images)
   - Source: https://github.com/ondyari/FaceForensics
   - Place in: `data_sources/ai_edited/FaceForensics/`

2. **ForgeryNet** (5,000 images)
   - Source: https://github.com/yinanhe/forgerynet
   - Place in: `data_sources/ai_edited/ForgeryNet/`

3. **CASIA** (4,000 images)
   - Source: CASIA Image Tampering Detection Dataset
   - Place in: `data_sources/ai_edited/CASIA/`

4. **IMD2020** (4,000 images)
   - Source: Image Manipulation Detection 2020 dataset
   - Place in: `data_sources/ai_edited/IMD2020/`

5. **DEFACTO** (3,000 images)
   - Source: DEFACTO dataset
   - Place in: `data_sources/ai_edited/DEFACTO/`

---

## 3. Configuration

The pipeline configuration has been updated in:
`dataset_builder/config/dataset_config.yaml`

Key settings:
- **Random seed**: 42 (for reproducibility)
- **Class targets**: 22,000 images per class
- **Split ratios**: 70% train, 15% val, 15% test
- **Min image size**: 256x256 pixels
- **Blur threshold**: 80
- **Quality score**: 0.7 minimum

Review and adjust these settings if needed before running the pipeline.

---

## 4. Dependencies

All required packages have been installed:
- ✓ Pillow (image processing)
- ✓ opencv-python (computer vision)
- ✓ imagehash (perceptual hashing)
- ✓ PyYAML (config parsing)
- ✓ tqdm (progress bars)
- ✓ numpy (numerical computing)

---

## 5. Running the Pipeline

### Quick Start

From the project root directory:

```bash
cd dataset_builder
python main.py --config config/dataset_config.yaml --log-level INFO
```

### Options

- `--config`: Path to config file (default: config/dataset_config.yaml)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--dry-run`: Simulate pipeline without writing files

### Dry Run (Recommended First)

Test the configuration without processing:

```bash
python main.py --config config/dataset_config.yaml --dry-run --log-level DEBUG
```

---

## 6. Pipeline Stages

The pipeline executes these stages automatically:

1. **Indexing** - Scans all source directories and catalogs images
2. **Validation** - Checks image integrity, resolution, and quality
3. **Deduplication** - Removes duplicates using perceptual hashing
4. **Sampling & Balancing** - Enforces class balance and quality thresholds
5. **Cluster-Based Split** - Creates train/val/test splits preventing leakage
6. **Export** - Copies images to final dataset structure
7. **Audit** - Generates compliance and quality reports

---

## 7. Output Artifacts

All outputs are written to: `dataset_builder/output/artifacts/`

### Intermediate Files
- `index.csv` - Raw scan of all images
- `validated_index.csv` - Valid images only
- `deduped_index.csv` - After deduplication
- `sampled_index.csv` - Balanced and filtered
- `split_index.csv` - With train/val/test assignments
- `split_report.csv` - Split statistics

### Final Outputs
- `export_index.csv` - Final dataset manifest
- `audit_report.json` - Quality metrics and validation
- `pipeline.log` - Complete execution log
- `conflict_report.csv` - Duplicate image details

### Dataset Files
Final dataset exported to: `data/`
```
data/
├── real/
│   ├── train/
│   ├── val/
│   └── test/
├── ai_generated/
│   ├── train/
│   ├── val/
│   └── test/
└── ai_edited/
    ├── train/
    ├── val/
    └── test/
```

---

## 8. Monitoring Progress

The pipeline provides detailed logging:

```
[1/7] Dataset Indexing...
[2/7] Image Validation...
[3/7] Deduplication...
[4/7] Quality Filtering and Class Balancing...
[5/7] Cluster-Based Train/Val/Test Split...
[6/7] Export and Packaging...
[7/7] Dataset Integrity Audit...
```

Check `dataset_builder/output/artifacts/pipeline.log` for detailed information.

---

## 9. Verification

After pipeline completion:

1. **Check audit report**: `dataset_builder/output/artifacts/audit_report.json`
   - Verdict should be "PASS" or "WARN"
   - Review any warnings or anomalies

2. **Verify class balance**:
   ```bash
   find data/real -type f | wc -l
   find data/ai_generated -type f | wc -l
   find data/ai_edited -type f | wc -l
   ```

3. **Check split ratios**:
   ```bash
   find data/real/train -type f | wc -l
   find data/real/val -type f | wc -l
   find data/real/test -type f | wc -l
   ```

---

## 10. Troubleshooting

### Common Issues

**Issue**: "No images found in source directory"
- **Solution**: Verify images are placed in correct `data_sources/` subdirectories

**Issue**: "Pipeline fails at validation stage"
- **Solution**: Check `pipeline.log` for corrupted images, remove or replace them

**Issue**: "Not enough images after filtering"
- **Solution**: Lower `min_quality_score` in config or add more source images

**Issue**: "Audit verdict: FAIL"
- **Solution**: Review `audit_report.json` for specific failures and adjust config

### Adjusting Configuration

To modify dataset size or quality:
1. Edit `dataset_builder/config/dataset_config.yaml`
2. Adjust `class_targets`, `min_quality_score`, or `image_rules`
3. Re-run the pipeline

### Clean Start

To start fresh:
```bash
rm -rf dataset_builder/output/artifacts/*
rm -rf data/real/* data/ai_generated/* data/ai_edited/*
```

---

## 11. Next Steps

After successful pipeline execution:

1. **Review outputs**: Check audit report and sample images
2. **Train models**: Use `scripts/training/train_full.py`
3. **Evaluate**: Use `scripts/evaluation/evaluate.py`
4. **Document**: Keep records of dataset version and configuration used

---

## Notes

- **Deterministic**: Same input + config = same output (random_seed: 42)
- **Cluster-aware splitting**: Similar images stay together to prevent leakage
- **Quality-first**: Corrupted or low-quality images are automatically filtered
- **Audit trail**: Full logs and manifests for reproducibility

For detailed pipeline documentation, see: `dataset_builder/README.md`

For dataset design rationale, see: `DATASET.md`
