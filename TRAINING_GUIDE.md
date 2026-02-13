# Model Training Guide — Deepfake Detection

This guide covers everything you need to train a deepfake detection model from scratch using the project's training scripts.

---

## 📋 Prerequisites

### 1. Dataset Preparation
Ensure your dataset is organized in the following structure:
```
data/
├── real/           # Real images (class 0)
├── ai_generated/   # AI-generated images (class 1)
└── ai_edited/      # AI-edited images (class 2)
```

**Recommended dataset size:**
- **Minimum:** 1,000 images per class (3,000 total)
- **Good:** 5,000-10,000 images per class (15,000-30,000 total)
- **Excellent:** 20,000+ images per class (60,000+ total)

### 2. Environment Setup
```bash
# Activate your virtual environment
source venv-linux/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Training Options

You have **two training scripts** to choose from:

### Option 1: `train_baseline.py` — Quick & Simple
✅ Best for: Beginners, quick experiments, baseline results  
✅ Features: Minimal configuration, fast setup, good defaults

### Option 2: `train_full.py` — Advanced & Production-Ready
✅ Best for: Research, production, experiment tracking  
✅ Features: Config files, TensorBoard, checkpoints, advanced logging

---

## 🚀 Option 1: Baseline Training (Recommended for Getting Started)

### Basic Usage
```bash
python scripts/training/train_baseline.py \
    --data_dir data \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001
```

### All Available Arguments
```bash
python scripts/training/train_baseline.py \
    --data_dir data \                    # Path to dataset root
    --epochs 20 \                        # Number of training epochs
    --batch_size 32 \                    # Batch size
    --learning_rate 0.001 \              # Learning rate
    --num_workers 4 \                    # DataLoader workers
    --device cuda \                      # Device: cuda or cpu
    --seed 42 \                          # Random seed for reproducibility
    --model_save_path models/best_model.pth  # Where to save best model
```

### Hardware-Specific Configurations

#### **Entry-Level (Integrated GPU, 8GB RAM)**
```bash
python scripts/training/train_baseline.py \
    --data_dir data \
    --epochs 10 \
    --batch_size 8 \
    --num_workers 1 \
    --device cpu
```

#### **Mid-Range (GTX 1650/3050, 16GB RAM)**
```bash
python scripts/training/train_baseline.py \
    --data_dir data \
    --epochs 20 \
    --batch_size 32 \
    --num_workers 2 \
    --device cuda
```

#### **High-End (RTX 4060/4070, 16GB+ RAM)**
```bash
python scripts/training/train_baseline.py \
    --data_dir data \
    --epochs 30 \
    --batch_size 64 \
    --num_workers 4 \
    --device cuda
```

### What Happens During Training?
1. Loads and splits dataset (80% train, 20% validation)
2. Initializes ResNet18 with pretrained ImageNet weights
3. Replaces final layer for 3-class classification
4. Trains with CrossEntropyLoss and Adam optimizer
5. Validates after each epoch
6. Saves best model based on validation accuracy
7. Prints training metrics to console

### Output
```
models/
└── best_model.pth           # Best model checkpoint
```

---

## 🔬 Option 2: Advanced Training (For Research & Production)

### Configuration-Driven Training

#### Step 1: Edit Configuration File
Edit `scripts/training/train_config.yaml`:

```yaml
# Dataset
data_dir: data
train_val_split: 0.8
seed: 42

# Model
model_name: resnet18
num_classes: 3
pretrained: true

# Training hyperparameters
epochs: 30
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
optimizer: adam              # adam, sgd, adamw

# Learning rate scheduler
lr_scheduler: step           # step, cosine, plateau
lr_step_size: 10             # For step scheduler
lr_gamma: 0.1                # For step scheduler

# Hardware
device: cuda
num_workers: 4
pin_memory: true

# Checkpointing
checkpoint_dir: models/checkpoints
save_every_n_epochs: 5
keep_best_only: false

# Logging
log_dir: logs
tensorboard_dir: results/tensorboard
save_plots: true
plot_dir: results/plots

# Early stopping (optional)
early_stopping: true
patience: 5                  # Stop if no improvement for N epochs
```

#### Step 2: Run Training
```bash
python scripts/training/train_full.py --config scripts/training/train_config.yaml
```

### Advanced Features

#### **1. TensorBoard Monitoring**
Monitor training in real-time:
```bash
# In a separate terminal
tensorboard --logdir results/tensorboard

# Open browser to: http://localhost:6006
```

**Metrics tracked:**
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Learning rate changes

#### **2. Checkpointing**
Models are saved periodically:
```
models/checkpoints/
├── epoch_5.pth
├── epoch_10.pth
├── epoch_15.pth
├── best_model.pth           # Best validation accuracy
└── last_model.pth           # Most recent epoch
```

Resume training from checkpoint:
```bash
python scripts/training/train_full.py \
    --config scripts/training/train_config.yaml \
    --resume models/checkpoints/epoch_15.pth
```

#### **3. Learning Rate Scheduling**

**Step Decay:**
```yaml
lr_scheduler: step
lr_step_size: 10      # Reduce LR every 10 epochs
lr_gamma: 0.1         # Multiply LR by 0.1
```

**Cosine Annealing:**
```yaml
lr_scheduler: cosine
epochs: 50            # Will smoothly decay LR over 50 epochs
```

**ReduceLROnPlateau:**
```yaml
lr_scheduler: plateau
patience: 3           # Reduce if no improvement for 3 epochs
```

#### **4. Early Stopping**
Automatically stop if validation accuracy plateaus:
```yaml
early_stopping: true
patience: 5           # Stop after 5 epochs without improvement
```

---

## 📊 Monitoring Training

### During Training
Watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

Watch CPU/RAM:
```bash
htop
```

### Training Output Example
```
Epoch 1/30
----------
Train Loss: 0.8234, Train Acc: 65.32%
Val Loss: 0.6891, Val Acc: 72.45%
Best model saved!

Epoch 2/30
----------
Train Loss: 0.5123, Train Acc: 78.91%
Val Loss: 0.4987, Val Acc: 81.23%
Best model saved!

...

Training complete in 45m 23s
Best validation accuracy: 87.65%
```

---

## 🔧 Troubleshooting

### Problem 1: CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
--batch_size 16  # or 8

# Reduce image resolution in preprocessing.py
# Change to 128x128 instead of 224x224

# Use gradient accumulation (if supported)
--gradient_accumulation_steps 2
```

### Problem 2: Training Too Slow
**Solutions:**
```bash
# Increase num_workers
--num_workers 4

# Enable pin_memory (in config)
pin_memory: true

# Use mixed precision (FP16) - add to train_full.py if needed
```

### Problem 3: Poor Accuracy
**Solutions:**
1. **More data:** Use dataset_builder to sample more images
2. **More epochs:** Train for 30-50 epochs instead of 10-20
3. **Data augmentation:** Ensure `train_transform` is enabled in preprocessing
4. **Learning rate:** Try 0.0001 or 0.0005 instead of 0.001
5. **Model architecture:** Try ResNet50 or EfficientNet instead of ResNet18

### Problem 4: Overfitting (Train Acc >> Val Acc)
**Solutions:**
1. **More data:** Increase dataset size
2. **Stronger augmentation:** Add more transforms in `preprocessing.py`
3. **Regularization:** Increase `weight_decay` to 0.001 or 0.01
4. **Dropout:** Add dropout layers to model
5. **Early stopping:** Enable early stopping in config

### Problem 5: Validation Accuracy Fluctuates
**Solutions:**
1. **Larger validation set:** Use 0.8 train/val split instead of 0.9
2. **Stable learning rate:** Use lower LR like 0.0001
3. **Batch normalization:** Ensure model uses BatchNorm layers
4. **More epochs:** Train longer for smoother convergence

---

## 📈 After Training

### 1. Evaluate Your Model
```bash
python scripts/evaluation/evaluate.py \
    --model_path models/best_model.pth \
    --data_dir data
```

**Output:**
```
Overall Accuracy: 87.65%

Per-Class Metrics:
Class 0 (Real):        Precision: 89.2%, Recall: 86.7%, F1: 87.9%
Class 1 (AI Generated): Precision: 85.1%, Recall: 88.3%, F1: 86.7%
Class 2 (AI Edited):   Precision: 88.9%, Recall: 87.9%, F1: 88.4%

Confusion Matrix saved to: results/confusion_matrix.png
```

### 2. Visualize Confusion Matrix
```bash
python scripts/evaluation/plot_confusion_matrix.py \
    --model_path models/best_model.pth \
    --data_dir data
```

### 3. Generate Grad-CAM Explanations
```bash
streamlit run frontend/app.py
```
Upload test images to see predictions with heatmap overlays.

---

## 🎯 Recommended Training Workflow

### For Beginners (First Time)
```bash
# 1. Use baseline training with small epochs
python scripts/training/train_baseline.py \
    --data_dir data \
    --epochs 10 \
    --batch_size 16

# 2. Evaluate
python scripts/evaluation/evaluate.py \
    --model_path models/best_model.pth \
    --data_dir data

# 3. If results are good, train longer
python scripts/training/train_baseline.py \
    --data_dir data \
    --epochs 30 \
    --batch_size 32
```

### For Research/Production
```bash
# 1. Configure train_config.yaml with your settings

# 2. Start TensorBoard
tensorboard --logdir results/tensorboard &

# 3. Train with full monitoring
python scripts/training/train_full.py \
    --config scripts/training/train_config.yaml

# 4. Monitor progress in browser (localhost:6006)

# 5. Evaluate best checkpoint
python scripts/evaluation/evaluate.py \
    --model_path models/checkpoints/best_model.pth \
    --data_dir data
```

---

## 🔬 Advanced: Experiment Tracking

### Compare Multiple Runs
```bash
# Run 1: Baseline
python scripts/training/train_full.py \
    --config configs/baseline.yaml \
    --experiment_name baseline_resnet18

# Run 2: Higher LR
python scripts/training/train_full.py \
    --config configs/high_lr.yaml \
    --experiment_name high_lr_resnet18

# Run 3: More augmentation
python scripts/training/train_full.py \
    --config configs/strong_aug.yaml \
    --experiment_name strong_aug_resnet18

# Compare in TensorBoard
tensorboard --logdir results/tensorboard
```

### Hyperparameter Search
Try different combinations:
- Learning rates: 0.0001, 0.0005, 0.001, 0.005
- Batch sizes: 16, 32, 64
- Weight decay: 0, 0.0001, 0.001
- Optimizers: Adam, AdamW, SGD

---

## 📚 Label Mapping Reference
Always remember:
- **Class 0:** Real images
- **Class 1:** AI Generated images
- **Class 2:** AI Edited images

This mapping is used consistently across all scripts.

---

## 🚀 Quick Reference Commands

### Train baseline model (CPU)
```bash
python scripts/training/train_baseline.py --data_dir data --epochs 10 --device cpu
```

### Train baseline model (GPU)
```bash
python scripts/training/train_baseline.py --data_dir data --epochs 20 --batch_size 32 --device cuda
```

### Train advanced model with config
```bash
python scripts/training/train_full.py --config scripts/training/train_config.yaml
```

### Monitor with TensorBoard
```bash
tensorboard --logdir results/tensorboard
```

### Evaluate trained model
```bash
python scripts/evaluation/evaluate.py --model_path models/best_model.pth --data_dir data
```

### Launch inference UI
```bash
streamlit run frontend/app.py
```

---

## 💡 Tips for Best Results

1. **Start small, then scale up**
   - Train 5-10 epochs first to verify everything works
   - Then train full 30-50 epochs

2. **Use GPU if available**
   - Training is 10-50x faster on GPU
   - Check with: `nvidia-smi`

3. **Monitor overfitting**
   - If train accuracy >> validation accuracy, you're overfitting
   - Add more data or increase regularization

4. **Save your experiments**
   - Use descriptive names for model checkpoints
   - Document your hyperparameters

5. **Use version control**
   - Commit code before major training runs
   - Tag successful experiments

---

## 📞 Need Help?

- Check [README.md](README.md) for project overview
- Check [DATASET.md](DATASET.md) for dataset preparation
- Check [Troubleshooting](#-troubleshooting) section above
- Review TensorBoard logs for training insights
- Check `logs/` directory for detailed error messages
