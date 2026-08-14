# File Cleanup Guide — Redundant Files & Superior Alternatives

This document identifies redundant files in your project and recommends which ones to delete and which superior alternatives to use instead.

---

## ❌ Files to DELETE (Have Superior Counterparts)

### 1. **`scripts/evaluation/plot_confusion.py`**
**Status:** ❌ DELETE  
**Superior alternative:** `scripts/evaluation/plot_confusion_matrix.py`

**Why delete?**
- Both files do the exact same thing (plot confusion matrices)
- `plot_confusion_matrix.py` has better code structure with a reusable `plot_cm()` function
- Keeping both creates confusion and maintenance burden

**Recommendation:**
```bash
rm scripts/evaluation/plot_confusion.py
```

**Use instead:**
```bash
python scripts/evaluation/plot_confusion_matrix.py
```

---

### 2. **`scripts/explainability/grad_cam.py`**
**Status:** ❌ DELETE  
**Superior alternative:** `frontend/gradcam.py`

**Why delete?**
- Both implement Grad-CAM for model explainability
- `frontend/gradcam.py` is more modular, reusable, and cleaner
- `scripts/explainability/grad_cam.py` is a standalone script with hardcoded paths
- `frontend/gradcam.py` is used by the Streamlit UI and can be imported anywhere

**Key differences:**
| Feature | `scripts/explainability/grad_cam.py` | `frontend/gradcam.py` ✅ |
|---------|--------------------------------------|--------------------------|
| **Modularity** | ❌ Standalone script | ✅ Importable class |
| **Reusability** | ❌ Hardcoded paths | ✅ Configurable |
| **UI Integration** | ❌ No | ✅ Used by Streamlit |
| **Code Quality** | ⚠️ Basic | ✅ Production-ready |

**Recommendation:**
```bash
rm scripts/explainability/grad_cam.py
```

**Use instead:**
```python
from frontend.gradcam import GradCAM, overlay_heatmap
from frontend.inference import load_model, preprocess_image
from PIL import Image

model = load_model("models/best_model.pth")
cam = GradCAM(model)

image = Image.open("sample.jpg")
tensor = preprocess_image(image)
heatmap = cam(tensor, class_idx=1)
overlay = overlay_heatmap(image, heatmap, alpha=0.5)
overlay.save("output.png")
```

---

### 3. **`ui/` Directory (Empty)**
**Status:** ❌ DELETE  
**Superior alternative:** `frontend/`

**Why delete?**
- `ui/components/` is empty
- `ui/utils/` is empty
- `frontend/` already contains the complete Streamlit UI
- No reason to keep an empty parallel directory

**Recommendation:**
```bash
rm -rf ui/
```

**Use instead:**
```bash
streamlit run frontend/app.py
```

---

### 4. **`scripts/data/split_data.py`** (Conditional)
**Status:** ⚠️ DELETE IF using `dataset_builder`  
**Superior alternative:** `dataset_builder/modules/splitter.py`

**Why delete?**
- `split_data.py` does basic random train/val split
- `dataset_builder/modules/splitter.py` does intelligent cluster-based splitting
- Cluster-based splitting prevents similar images from leaking across train/test sets

**Key differences:**
| Feature | `scripts/data/split_data.py` | `dataset_builder/modules/splitter.py` ✅ |
|---------|------------------------------|------------------------------------------|
| **Split method** | ❌ Random | ✅ Cluster-based (prevents leakage) |
| **Configurability** | ⚠️ CLI args only | ✅ YAML config |
| **Deduplication** | ❌ No | ✅ Yes, via pHash |
| **Audit trail** | ❌ Basic CSV log | ✅ Detailed reports |
| **Best for** | Small, pre-organized data | Large-scale dataset building |

**Recommendation:**
```bash
# If you're using dataset_builder pipeline:
rm scripts/data/split_data.py

# If you have small, pre-organized data and don't need dataset_builder:
# Keep scripts/data/split_data.py
```

**Use dataset_builder instead:**
```bash
cd dataset_builder
python main.py --config config/dataset_config.yaml
```

---

## ✅ Files to KEEP (No Better Alternative)

### Data Utilities
- ✅ **`scripts/data/clean_dataset.py`** — Removes corrupted images (unique utility)
- ✅ **`scripts/data/dataset_stats.py`** — Quick dataset statistics (useful standalone tool)

### Preprocessing
- ✅ **`scripts/preprocessing/preprocessing.py`** — Data augmentation pipeline (used by training)
- ✅ **`scripts/preprocessing/visualize_augmentations.py`** — Visualize augmentation effects

### Dataset Loading
- ✅ **`scripts/dataloader/dataset.py`** — PyTorch Dataset class (required for training)
- ✅ **`scripts/dataloader/dataset_loader.py`** — DataLoader creation (required for training)

### Training
- ✅ **`scripts/training/train_baseline.py`** — Quick baseline training (simple, beginner-friendly)
- ✅ **`scripts/training/train_full.py`** — Advanced training (production features)
- ✅ **`scripts/training/train_config.yaml`** — Training configuration

### Evaluation
- ✅ **`scripts/evaluation/evaluate.py`** — Compute metrics (primary evaluation script)
- ✅ **`scripts/evaluation/evaluation_matrices.py`** — Additional metrics computation
- ✅ **`scripts/evaluation/plot_confusion_matrix.py`** — Confusion matrix visualization

### Frontend
- ✅ **`frontend/app.py`** — Streamlit UI (main application)
- ✅ **`frontend/config.py`** — UI configuration
- ✅ **`frontend/inference.py`** — Inference utilities
- ✅ **`frontend/gradcam.py`** — Grad-CAM implementation (best version)

### Dataset Builder
- ✅ **`dataset_builder/main.py`** — Pipeline orchestrator
- ✅ **`dataset_builder/pipeline.py`** — Pipeline logic
- ✅ **All modules in `dataset_builder/modules/`** — Core pipeline components

---

## 🧹 Recommended Cleanup Commands

### Safe Cleanup (Removes only confirmed redundant files)
```bash
cd /home/dheirav/Code/Deepfake_Detection/deepfake-project

# Delete redundant confusion matrix script
rm scripts/evaluation/plot_confusion.py

# Delete redundant grad-cam script
rm scripts/explainability/grad_cam.py

# Delete empty UI directory
rm -rf ui/

# Optional: Delete simple split script if using dataset_builder
# rm scripts/data/split_data.py

echo "Cleanup complete!"
```

### Verify Cleanup
```bash
# Check what was deleted
git status

# If you want to undo (before committing):
git restore scripts/evaluation/plot_confusion.py
git restore scripts/explainability/grad_cam.py
```

---

## 📊 Summary Table

| File | Status | Reason | Superior Alternative |
|------|--------|--------|---------------------|
| `scripts/evaluation/plot_confusion.py` | ❌ DELETE | Duplicate functionality | `plot_confusion_matrix.py` |
| `scripts/explainability/grad_cam.py` | ❌ DELETE | Less modular, hardcoded | `frontend/gradcam.py` |
| `ui/` directory | ❌ DELETE | Empty, unused | `frontend/` |
| `scripts/data/split_data.py` | ⚠️ CONDITIONAL | Basic random split | `dataset_builder/modules/splitter.py` |

**Total files to delete:** 3-4 files/directories

---

## 🎯 Benefits of Cleanup

After cleanup, you'll have:
- ✅ **Clearer project structure** — No confusion about which file to use
- ✅ **Less maintenance burden** — Fewer files to update and test
- ✅ **Better code quality** — Using superior implementations only
- ✅ **Easier onboarding** — New contributors won't be confused by duplicates

---

## ⚠️ Important Notes

### Before Deleting
1. **Check if any custom code depends on these files**
   ```bash
   grep -r "plot_confusion.py" .
   grep -r "grad_cam.py" .
   grep -r "ui/" .
   ```

2. **Commit your current state first**
   ```bash
   git add .
   git commit -m "Checkpoint before cleanup"
   ```

3. **Run tests (if you have any)**
   ```bash
   pytest  # or your test command
   ```

### After Deleting
1. **Update any documentation that references deleted files**
2. **Update import statements if needed**
3. **Test that training and evaluation still work**
   ```bash
   python scripts/training/train_baseline.py --data_dir data --epochs 1
   python scripts/evaluation/evaluate.py --model_path models/best_model.pth --data_dir data
   streamlit run frontend/app.py
   ```

---

## 🔄 Migration Guide

### If you were using `plot_confusion.py`:
**Old:**
```bash
python scripts/evaluation/plot_confusion.py
```

**New:**
```bash
python scripts/evaluation/plot_confusion_matrix.py
```

### If you were using `scripts/explainability/grad_cam.py`:
**Old:**
```bash
python scripts/explainability/grad_cam.py
```

**New (Option 1 - UI):**
```bash
streamlit run frontend/app.py
# Upload image in UI to see Grad-CAM
```

**New (Option 2 - Programmatic):**
```python
from frontend.gradcam import GradCAM, overlay_heatmap
from frontend.inference import load_model, preprocess_image
from PIL import Image

model = load_model("models/best_model.pth")
cam = GradCAM(model)
image = Image.open("sample.jpg")
tensor = preprocess_image(image)
heatmap = cam(tensor, class_idx=1)
overlay = overlay_heatmap(image, heatmap, alpha=0.5)
overlay.save("gradcam_output.png")
```

### If you were using `scripts/data/split_data.py`:
**Old:**
```bash
python scripts/data/split_data.py --data_dir data --test_size 0.2
```

**New (Recommended):**
```bash
cd dataset_builder
python main.py --config config/dataset_config.yaml
```

**New (Alternative - if you need simple splitting):**
```python
from sklearn.model_selection import train_test_split
# Use train_test_split directly in your training script
```

---

## 📞 Questions?

If you're unsure about deleting any file:
1. Check if it's imported anywhere: `grep -r "filename" .`
2. Look for similar functionality in other files
3. Test your workflow after deletion
4. Keep a backup or use git to track changes

Remember: You can always restore deleted files from git history!
```bash
git log --all --full-history -- path/to/deleted/file
git restore --source=<commit-hash> path/to/deleted/file
```
