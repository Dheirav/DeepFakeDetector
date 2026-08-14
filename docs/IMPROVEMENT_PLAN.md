# Improvement Plan — Deepfake Detection Model

**Date:** March 8, 2026  
**Current best:** `convnext_srm_focal` — 86.85% test accuracy, F1 macro 0.868  
**Primary target:** Reduce Real ↔ AI-Edited boundary confusion

---

## The Core Problem

The model's critical weakness is the Real/AI-Edited boundary:

| Error type | Count | Rate |
|---|---|---|
| Real predicted as AI-Edited | 1,293 | 16.6% of Real samples |
| AI-Edited predicted as Real | 1,124 | 14.5% of AI-Edited samples |
| **Total boundary errors** | **2,417** | **10.36% of test set** |

By contrast, AI-Generated errors are almost solved (2.4% miss rate).  
AI-Edited images are locally manipulated real photos — the manipulation region is
often small relative to the whole image, so in low-texture images the global feature
vector looks nearly identical to Real. This is an **information bottleneck problem**,
not a regularisation problem.

---

## Priority Ranking (Impact vs Effort)

| # | Step | Expected Δ Acc | Effort | Targets | Status |
|---|---|---|---|---|---|
| 1a | ~~Class weights [2.0,1.0,2.0]~~ | ~~+0.5–1.0%~~ | — | — | ✅ Done — sweep showed **[1.5,1.0,1.5] best**, higher weights hurt |
| 1b | Focal loss (γ = 3.0) | +0.3–0.7% | ~30 min | Real→AIEdit, AIEdit→Real | ✅ Done — `convnext_gamma3` |
| 2 | Compression + noise augmentation | +0.5–1.0% | ~1 hr | Both directions | ✅ Done — `convnext_augv2`, `convnext_augv3_light`, `convnext_augv4*` |
| 3 | Cascade specialist model | +1.5–3.0% | 4–6 hr | Both directions | ⬜ Not tried |
| 4 | Attention / patch-level features (GeM pooling) | +1.0–2.0% | 3–4 hr | AIEdit→Real | ⬜ Not tried |
| 5 | ~~ConvNeXt-Tiny backbone~~ | ~~+3–4%~~ | — | — | ✅ Done |
| 5b | ConvNeXt-Small backbone | +0.5–1.5% | ~2 hr | General | ✅ Done — `convnext_small_augv1_light` |
| 6 | Mixup / CutMix between Real and AI-Edited | +0.5–1.0% | ~1 hr | Both directions | ⬜ Not tried |

## Step 1 — Boundary-Aware Loss Tuning

### 1a — Class weights ✅ ALREADY DONE
Full weight sweep was run across 8 configurations. `[1.5, 1.0, 1.5]` won.  
Pushing Real/AI-Edited weights above 1.5 consistently degraded performance — do not retry.

### 1b — Focal gamma=3.0 (weights stay at [1.5, 1.0, 1.5]) 🔲
**Why:** The weight sweep used `gamma=2.0` throughout. A higher gamma doesn't change  
which class gets emphasised — it changes *how hard* the model is forced to focus on  
difficult boundary samples within each class. With gamma=3, a sample that's only 70%  
confident contributes 3× more loss than with gamma=2. This could help without the  
disadvantage of over-weighting a class.

**Change needed:** Expose `--focal-gamma` arg in `train_full.py` (1 line in `build_criterion` call).

**Run to try:**
```bash
python scripts/training/train_full.py \
  --backbone convnext_tiny \
  --run-name convnext_gamma3 \
  --focal-gamma 3.0
```

**Success criteria:** AI-Edited F1 > 0.850, Real F1 > 0.820, total boundary errors < 2,100

---

## Step 2 — Augmentation: Compression + Local Noise

**Why:** AI-Edited images often survive JPEG compression and noise — their manipulation
artefacts are low-frequency. Real images are more sensitive. Exposing the model to
degraded real images forces it to learn texture-independent Real features rather than
relying on compression quality as a shortcut.

**Changes to `scripts/preprocessing/preprocessing.py`:**

```python
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    
    # NEW: compression artefact simulation (especially important for Real class)
    A.ImageCompression(quality_range=(20, 95), p=0.5),   # was (30,100) p=0.4
    
    # NEW: sensor / transmission noise
    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.4),
    
    # NEW: slight blur to prevent texture overfitting
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.3),
    
    A.Normalize(),
    ToTensorV2()
])
```

**Success criteria:** Train/val gap shrinks below 10 pts, AI-Edited recall improves

---

## Step 3 — Cascade Specialist Model (Highest Expected Impact)

**Why:** A single 3-class model must simultaneously handle the easy case (Real vs
AI-Generated) and the hard case (Real vs AI-Edited). The hard case needs much more
capacity and task-specific features. A cascade separates them.

**Architecture:**
```
Stage 1 — "Is it AI-Generated?"  (existing convnext_srm_focal, frozen)
    ├── If AI-Generated (confidence > 0.90): output class 1, done
    └── Otherwise → Stage 2

Stage 2 — "Is it Real or AI-Edited?" (new specialist, trained only on those 2 classes)
    ├── Uses same ConvNeXt-Tiny backbone
    ├── Trained ONLY on Real + AI-Edited samples
    ├── Can use patch-level attention or higher-resolution input (320×320)
    └── Output: class 0 or class 2
```

**New files to create:**
- `scripts/training/train_specialist.py` — binary Real/AI-Edited trainer:
  - Higher resolution input: 320×320
  - Stronger augmentation focused on local texture
  - Class-balanced sampling (equal Real and AI-Edited per batch)
  - Longer training: 60 epochs
- `scripts/evaluation/evaluate_cascade.py` — runs both models, applies cascade logic
- `scripts/inference/predict_cascade.py` — for deployment / demo

**Specialist model training command (to create):**
```bash
python scripts/training/train_specialist.py \
  --run-name specialist_real_vs_aiedit_v1 \
  --backbone convnext_tiny \
  --input-size 320 \
  --epochs 60
```

**Expected gains:** +1.5–3.0% overall test accuracy by removing 40–60% of boundary errors

---


## Step 4 — Patch-Level Attention (Localise the Edit Region)

### Add-on/Extendable Attention Head Methodology

To make patch-level attention (e.g., GeM pooling, CBAM) an add-on, extendable feature:

1. **Modularize the Classifier Head**
    - Add an argument (e.g., `attention_head`) to `_build_backbone` in `train_full.py`.
    - Allow dynamic selection of the classifier head (standard, GeM, CBAM, etc.).

2. **Implement Attention Heads as Separate Modules**
    - Create a file `modules/attention_heads.py`.
    - Implement classes like `GeMMPoolingHead`, `CBAMHead`.
    - Import and use them in `_build_backbone` based on the selected type.

3. **Add Command-Line Argument**
    - Add `--attention-head` to your training script to select the head type.

4. **Example Integration (GeM Pooling):**
    - In `train_full.py`, after parsing args, pass `args.attention_head` to `_build_backbone`.
    - In `_build_backbone`, use:
      ```python
      if attention_head == "gem":
            from modules.attention_heads import GeMMPoolingHead
            m.classifier[2] = GeMMPoolingHead(m.classifier[2].in_features, num_classes, dropout_p)
      ```

5. **Extendable for Other Attention Modules**
    - Add more heads (e.g., CBAM) in `modules/attention_heads.py`.
    - Update `_build_backbone` to support them.

This approach allows easy experimentation and extension with new attention/pooling heads.

**Why:** ConvNeXt's global average pooling collapses spatial information. An AI-Edited
image where only 10% of pixels were manipulated looks identical to Real at the global
feature level. Adding spatial attention forces the model to find the edited region.

**Implementation options (choose one):**

### Option A — CBAM (Convolutional Block Attention Module), easiest
Insert a CBAM block after the last ConvNeXt stage before the classifier.  
`pip install timm` — CBAM available in `timm.models.layers`.

### Option B — Class Activation Mapping supervision (Grad-CAM guided training)
During training, check if the Grad-CAM activation falls on a plausible manipulated
region.  
Requires manipulation masks in the dataset — only possible with FaceForensics++  
and CASIA subsets which include masks.

### Option C — Multi-scale feature pooling
Replace global average pool with `GeM` (Generalised Mean Pooling) which preserves
more spatial structure, or concat global average + global max + spatial max features.

**Recommended starting point:** Option C — 2-hour change, no new dependencies.

**Changes to `_build_backbone()` in `train_full.py`:**
```python
# Replace model.classifier with:
class GeMMPoolingHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout_p, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):  # x: [B, C, H, W] last feature map
        x = F.avg_pool2d(x.clamp(min=1e-6).pow(self.p), x.shape[-2:]).pow(1/self.p)
        return self.fc(self.drop(x.flatten(1)))
```

---

## Step 5 — Larger Backbone (ConvNeXt-Small) ✅ ConvNeXt-Tiny done → now go bigger

**ConvNeXt-Tiny (28M) is the current best at 86.85%. Already done.**

**Why go bigger:** ConvNeXt-Small has 50M params. The extra capacity specifically helps  
with fine-grained texture discrimination (exactly the Real/AI-Edit case).

**Prerequisite:** Steps 1b–3 first. A bigger model on the same pipeline without addressing
augmentation and cascade may just overfit harder.

**ConvNeXt-Small is already in torchvision:**
```python
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
# Add to _build_backbone():
elif backbone_name == "convnext_small":
    m = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    m.classifier[2] = make_head(m.classifier[2].in_features)  # 768 → num_classes
    return m
```

**Memory note:** ConvNeXt-Small at batch_size=64 needs ~7.5 GB GPU. With 8.2 GB available,
reduce batch to 48 if OOM.

---

## Step 6 — Mixup Between Real and AI-Edited

**Why:** Linear interpolation between Real and AI-Edited training images in both input
and label space forces the model to build a smooth, well-calibrated decision boundary
rather than a sharp heuristic.

**Implementation in training loop (`train_full.py`):**
```python
def mixup_batch(x, y, alpha=0.4, boundary_only=True):
    """Mixup only between Real (0) and AI-Edited (2) samples."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    if boundary_only:
        # Only mix Real↔AIEdit pairs
        mask_real = (y == 0)
        mask_edit = (y == 2)
        if mask_real.sum() > 0 and mask_edit.sum() > 0:
            idx_real = torch.where(mask_real)[0]
            idx_edit = torch.where(mask_edit)[0]
            n = min(len(idx_real), len(idx_edit))
            x[idx_real[:n]] = lam * x[idx_real[:n]] + (1-lam) * x[idx_edit[:n]]
            # Soft labels: e.g. lam=0.7 Real → [0.7, 0, 0.3]
            # Requires soft-label-aware loss
    return x, y
```

Add `--mixup-alpha` flag (default 0.0, try 0.2–0.4).

---


## Updated Recommended Execution Order

1. Cascade specialist model (train_specialist.py, evaluate_cascade.py)
2. Patch-level attention (GeM pooling or CBAM)
3. ConvNeXt-Small backbone (if more capacity needed)
4. Mixup Real↔AI-Edited (add --mixup-alpha)
5. Final ensemble evaluation

All steps above should be measured using the checklist below after each run.

---

## Measurement Checklist

After each run, check these specifically (not just overall accuracy):

- [ ] AI-Edited recall > 0.860 (currently 0.844)
- [ ] Real recall > 0.820 (currently 0.796)  
- [ ] Boundary errors (Real↔AIEdit) < 2,000 (currently 2,417)
- [ ] Train/val gap < 10 pts (currently 12.7)
- [ ] AI-Generated F1 stays above 0.950 (don't regress the easy class)

Run `python scripts/compare_runs.py --best-only --name <run_name>` after each experiment
to auto-generate the comparison plots.
