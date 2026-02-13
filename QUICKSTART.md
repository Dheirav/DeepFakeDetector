# Quick Start Guide - Dataset Pipeline

## TL;DR - What You Need to Do

### 1. Place Your Images
Copy downloaded dataset images into these folders:

```
data_sources/
├── real/FFHQ/              ← FFHQ dataset (5k images)
├── real/COCO/              ← MS COCO real images (6k images)
├── real/OpenImages/        ← Open Images (5k images)
├── real/ImageNet/          ← ImageNet photos (4k images)
├── ai_generated/StyleGAN/  ← StyleGAN outputs (7k images)
├── ai_generated/StableDiffusion/ ← Stable Diffusion (7k images)
├── ai_generated/Midjourney_DALLE/ ← Midjourney/DALL-E (4k images)
├── ai_generated/LAION/     ← LAION subset (4k images)
├── ai_edited/FaceForensics/ ← FaceForensics++ (6k images)
├── ai_edited/ForgeryNet/   ← ForgeryNet (5k images)
├── ai_edited/CASIA/        ← CASIA tampering (4k images)
├── ai_edited/IMD2020/      ← IMD2020 (4k images)
└── ai_edited/DEFACTO/      ← DEFACTO (3k images)
```

### 2. Run the Pipeline

```bash
cd dataset_builder
python main.py --config config/dataset_config.yaml --log-level INFO
```

### 3. Check Output

Final dataset will be in `data/real/`, `data/ai_generated/`, `data/ai_edited/`

---

## What's Been Done

✅ **Directory structure created** - All 13 source folders ready  
✅ **Configuration updated** - `dataset_config.yaml` has all required settings  
✅ **Pipeline fixed** - Module calls corrected for proper execution  
✅ **Dependencies installed** - All required packages ready  
✅ **Documentation created** - Complete setup guide available  

---

## What Happens When You Run It

1. **Indexing** - Scans all images in `data_sources/`
2. **Validation** - Checks quality, resolution, corruption
3. **Deduplication** - Removes duplicates via perceptual hashing
4. **Sampling** - Balances classes (22k images each)
5. **Splitting** - Creates 70/15/15 train/val/test split
6. **Export** - Copies images to `data/` folder
7. **Audit** - Generates quality report

---

## Key Configuration Settings

```yaml
class_targets:
  real: 22000
  ai_generated: 22000
  ai_edited: 22000

split_ratios:
  train: 0.7   # 70% training
  val: 0.15    # 15% validation
  test: 0.15   # 15% test

image_rules:
  min_width: 256
  min_height: 256
  blur_threshold: 80
  min_quality_score: 0.7
```

---

## Outputs

**Artifacts** (in `dataset_builder/output/artifacts/`):
- `pipeline.log` - Full execution log
- `audit_report.json` - Quality metrics
- `export_index.csv` - Final dataset manifest

**Dataset** (in `data/`):
```
data/
├── real/{train,val,test}/
├── ai_generated/{train,val,test}/
└── ai_edited/{train,val,test}/
```

---

## Testing Before Full Run

Dry run to test without processing:
```bash
python main.py --config config/dataset_config.yaml --dry-run --log-level DEBUG
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No images found | Check images are in correct `data_sources/` folders |
| Pipeline fails | Check `pipeline.log` for errors |
| Not enough images | Lower `min_quality_score` or add more sources |
| Audit fails | Review `audit_report.json` for issues |

---

## For Full Details

See **SETUP_INSTRUCTIONS.md** for complete documentation including:
- Data source download links
- Detailed configuration options
- Stage-by-stage pipeline explanation
- Verification procedures

---

## Support

- Pipeline documentation: `dataset_builder/README.md`
- Dataset design rationale: `DATASET.md`
- Training guide: `TRAINING_GUIDE.md`
