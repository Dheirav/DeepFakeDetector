# Quick Start Guide

## Status: ✅ Dataset Built — Ready for Training

The dataset is fully constructed and exported. **No pipeline runs are required to start training.**

---

## Dataset Summary

| Class | Count |
|---|---|
| Real | 26,000 |
| AI Generated | 26,000 |
| AI Edited | 25,865 |
| **Total** | **77,865** |

Max class imbalance: **0.52%** (within the ≤2% target)

---

## Dataset Location

```
dataset_builder/
├── train/
│   ├── real/          (~10,400 images)
│   ├── ai_generated/  (~10,400 images)
│   └── ai_edited/     (~10,346 images)
├── val/
│   ├── real/          (~7,800 images)
│   ├── ai_generated/  (~7,800 images)
│   └── ai_edited/     (~7,760 images)
└── test/
    ├── real/          (~7,800 images)
    ├── ai_generated/  (~7,800 images)
    └── ai_edited/     (~7,760 images)
```

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train a Model

**Baseline (ResNet18, minimal setup):**
```bash
python scripts/training/train_baseline.py \
    --data_dir dataset_builder \
    --epochs 20 \
    --batch_size 32 \
    --device cuda
```

**Advanced (config-driven, TensorBoard, checkpoints):**
```bash
python scripts/training/train_full.py --config scripts/training/train_config.yaml
```

Set `data_dir: dataset_builder` in `train_config.yaml`.

---

## 3. Evaluate

```bash
python scripts/evaluation/evaluate.py \
    --model_path models/best_resnet18.pth \
    --data_dir dataset_builder
```

---

## 4. Launch the UI

```bash
streamlit run frontend/app.py
```

Upload an image to get a prediction and Grad-CAM heatmap.

---

## Hardware Configurations

| Hardware | Batch Size | Workers |
|---|---|---|
| Entry-level (integrated GPU, 8GB RAM) | 8–16 | 1 |
| Mid-range (GTX 1650/3050, 16GB RAM) | 16–32 | 2 |
| High-end (RTX 4060/4070, 16GB+ RAM) | 64 | 2–4 |

Monitor GPU: `watch -n 1 nvidia-smi`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `batch_size` |
| Import errors | Run from project root with venv active |
| Pipeline audit fails | Check `dataset_builder/output/artifacts/*/pipeline.log` |

---

## References

- Dataset design and sources: [DATASET.md](DATASET.md)
- Pipeline internals: [dataset_builder/README.md](dataset_builder/README.md)
- Training guide: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Full project docs: [README.md](README.md)
