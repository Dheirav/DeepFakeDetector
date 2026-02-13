
# Deepfake Detection Dataset — Design & Builder Pipeline

This document outlines the dataset design specification and builder pipeline for a research-grade, multi-class deepfake detection system.

---

## Project Goals
- Build a dataset with three balanced classes: **Real**, **AI Generated**, and **AI Edited**
- Enforce class balance and prevent dataset bias
- Deduplicate images across all sources using perceptual hashing
- Filter low-quality images based on resolution, blur, and metadata
- Sample exact image quotas per dataset source
- Create deterministic train/validation/test splits
- Produce reproducible manifests and audit logs

**Code Quality Requirements:**
- Modular architecture with clear separation of concerns
- Comprehensive logging and error handling
- Deterministic outputs with reproducible random seeds
- No shortcuts or brittle assumptions

**This project prioritizes correctness over speed.**

---

## Target Dataset Size (High Confidence Setup)
- **Recommended total images:** 60,000 to 90,000
- **Per class:** 20,000 to 30,000 images

This scale is strong enough to:
- Reach 80–90% accuracy with ResNet18 or EfficientNet
- Generalize across unseen AI models
- Learn subtle and localized manipulation artifacts
- Remain stable under compression and resizing

---

## Class Breakdown

### CLASS 0 — REAL IMAGES
- **Target:** 22,000 images
- **Purpose:** Real world distribution, camera noise learning, baseline authenticity

#### Sources & Types
- **FFHQ Real Faces:** 5,000 images
  - High resolution face portraits
  - Varied age, ethnicity, lighting
  - No filters, no edits
  - *Why:* Face realism anchor, matches deepfake face domains
- **MS COCO (Real Only):** 6,000 images
  - People in natural scenes, indoor/outdoor, objects, cluttered scenes
  - Mixed lighting
  - *Avoid:* Cartoon or illustration categories
  - *Why:* Non-face realism, real scene distribution
- **Open Images Dataset:** 5,000 images
  - Vehicles, buildings, street scenes, nature, animals
  - Real camera photos only
  - *Why:* Broad environmental diversity
- **ImageNet (Photo Subset Only):** 4,000 images
  - Natural objects, tools, animals, landscapes
  - *Avoid:* Artistic or synthetic images
  - *Why:* Texture and natural image statistics diversity
- **Optional Real Additions:**
  - Flickr, Unsplash, DSLR camera sets

### CLASS 1 — AI GENERATED
- **Target:** 22,000 images
- **Purpose:** Learn synthetic texture patterns, diffusion artifacts, GAN fingerprints

#### Sources & Types
- **StyleGAN/StyleGAN2/StyleGAN3:** 7,000 images
  - Face portraits, high realism, diverse ages/genders/lighting
  - *Why:* GAN artifact baseline, face generation artifacts
- **Stable Diffusion Outputs:** 7,000 images
  - Faces, landscapes, urban scenes, objects
  - *Must include:* Realistic prompts only
  - *Avoid:* Anime, stylized art
  - *Why:* Diffusion noise pattern learning
- **Midjourney/DALL·E (Research Mirrors):** 4,000 images
  - Ultra-realistic photography style, people, nature, interiors, street scenes
  - *Avoid:* Artistic or painterly styles
  - *Why:* Commercial AI fingerprint learning
- **LAION Diffusion Subset:** 4,000 images
  - Realistic internet-style images, mixed realism levels
  - *Why:* Prevent overfitting to one AI model

### CLASS 2 — AI EDITED (REAL + MANIPULATED)
- **Target:** 22,000 images
- **Purpose:** Learn localized manipulation, blending artifacts, inpainting traces

#### Sources & Types
- **FaceForensics++:** 6,000 images
  - Face swaps, reenactment, neural textures, compressed/uncompressed
  - *Why:* Core facial manipulation training
- **ForgeryNet:** 5,000 images
  - Face edits, object edits, scene edits
  - *Why:* Mixed manipulation diversity
- **CASIA Image Tampering:** 4,000 images
  - Splicing, copy-move, object insertion
  - *Why:* Classical edit learning
- **IMD2020:** 4,000 images
  - Inpainting, object removal, scene editing
  - *Why:* Localized artifact detection
- **DEFACTO:** 3,000 images
  - Semantic edits, AI-assisted object replacement, subtle edits
  - *Why:* Hard case manipulation learning

---

## Balanced Class Summary
| Class        | Target   | Percent |
|--------------|----------|---------|
| Real         | 22,000   | ~33%    |
| AI Generated | 22,000   | ~33%    |
| AI Edited    | 22,000   | ~33%    |
| **Total**    | 66,000   | Balanced|

This meets bias control and fairness constraints.

---

## Image Types to Include Per Class

### Real
- Faces (40%)
- Non-face scenes (60%)
- Indoor and outdoor
- Low-light and daylight
- Different camera qualities

### AI Generated
- GAN faces (30%)
- Diffusion faces (20%)
- Diffusion environments (30%)
- Mixed realism objects (20%)
- **Must avoid:** Anime, paintings, stylized fantasy

### AI Edited
- Face swaps (30%)
- Object insertion/removal (25%)
- Inpainting (20%)
- Background replacement (15%)
- Subtle edits (10%)

---

## Upper Limit for Maximum Success (4 TB SSD)
- **120,000 to 180,000 images total**
- **Per class:** 40,000 to 60,000 images

This scale:
- Rivals industry benchmark datasets
- Allows robust transformer or ConvNeXt training
- Handles unseen generative models better
- Supports paper-grade benchmarking

**Storage estimate at 512×512 JPEG Q85:**
- 150 KB per image
- 150,000 images ≈ 22.5 GB (well below 4 TB limit)

---

## Dataset Builder Pipeline

The `dataset_builder/` module provides a production-grade, deterministic pipeline for constructing this dataset from multiple large-scale sources.

### Pipeline Stages
1. **Indexing**: Scans and indexes all dataset files from source directories
2. **Validation**: Validates image integrity, resolution, and metadata
3. **Deduplication**: Removes duplicate or near-duplicate files using perceptual hashing (pHash)
4. **Quality Filtering**: Filters low-quality or outlier samples based on configurable thresholds
5. **Sampling & Class Balancing**: Ensures balanced class distribution and meets target quotas
6. **Cluster-Based Split**: Splits data into train/val/test sets while keeping similar images together
7. **Export & Packaging**: Exports the final dataset structure and index manifests
8. **Audit**: Runs integrity and compliance checks with detailed reporting

### Usage
Run the pipeline from the `dataset_builder` directory:

```bash
python main.py --config config/dataset_config.yaml [--dry-run] [--log-level INFO]
```

### Configuration Example
```yaml
random_seed: 42
artifacts_dir: output/artifacts
export_root: data/
strict_mode: true
image_rules:
  min_width: 256
  min_height: 256
class_targets:
  - real
  - ai_generated
  - ai_edited
split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15
```

### Key Features
- **Modular and extensible**: Each stage is independent and can be customized
- **Config-driven**: All parameters controlled via YAML configuration
- **Deterministic**: Fixed random seeds ensure reproducible results
- **Dry-run mode**: Test pipeline without writing files
- **Comprehensive logging**: Detailed logs and audit reports for every run
- **Cluster-aware splitting**: Prevents similar images from leaking across train/test splits

### Artifacts
All intermediate and final artifacts are written to the configured `artifacts_dir`:
- `index.csv`: Raw dataset index
- `validated_index.csv`: Validated images
- `deduped_index.csv`: Deduplicated images
- `filtered_index.csv`: Quality-filtered images
- `sampled_index.csv`: Class-balanced sample
- `split_index.csv`: Train/val/test split with cluster assignments
- `export_index.csv`: Final exported dataset index
- `audit_report.json`: Compliance and quality audit report
- `pipeline.log`: Full execution log

See [dataset_builder/README.md](dataset_builder/README.md) for complete documentation.

---

## Recommended Workflow

1. **Download source datasets** (FFHQ, COCO, StyleGAN outputs, FaceForensics++, etc.)
2. **Configure the pipeline** in `dataset_builder/config/dataset_config.yaml`:
   - Set source directories for each class
   - Define target counts per source
   - Configure quality thresholds
3. **Run the dataset builder**:
   ```bash
   cd dataset_builder
   python main.py --config config/dataset_config.yaml
   ```
4. **Verify the output** in `data/real/`, `data/ai_generated/`, `data/ai_edited/`
5. **Proceed to model training** using the scripts in `scripts/training/`
