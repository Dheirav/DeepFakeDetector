

# Deepfake Detection Training Pipeline — Documentation

This directory contains robust, research-grade scripts and configuration for training deepfake detection models using PyTorch.

---

## 1. Prerequisites

- Python 3.8+
- Install dependencies:
	```bash
	pip install -r ../../requirements.txt
	```
- Ensure your data is organized as:
	```
	data/
		real/
		ai_generated/
		ai_edited/
	```

---

## 2. Baseline Training Script (`train_baseline.py`)

### New Features (Research-Grade)
- **Data Loading:** Uses the DeepfakeDataset and standard transforms for robust input handling.
- **Device Selection:** Automatically uses GPU if available.
- **Training & Validation Loop:** Includes both training and validation phases each epoch, with accuracy and loss logging.
- **Best Model Saving:** Automatically saves the best model based on validation accuracy.
- **Reproducibility:** Sets all random seeds for deterministic results.
- **Flexible Hyperparameters:** All key settings (epochs, batch size, learning rate, etc.) are configurable via command-line arguments.
- **Progress Bar:** Uses tqdm for real-time progress updates.

### Usage
```bash
python train_baseline.py --data_dir data --epochs 5 --batch_size 32 --lr 0.0001 --val_split 0.2 --checkpoint_dir models --seed 42
```
All arguments are optional and have sensible defaults.

#### Arguments
| Argument           | Description                        | Default         |
|--------------------|------------------------------------|-----------------|
| --data_dir         | Path to dataset root               | "data"          |
| --epochs           | Number of training epochs          | 5               |
| --batch_size       | Batch size                         | 32              |
| --lr               | Learning rate                      | 0.0001          |
| --val_split        | Validation split ratio             | 0.2             |
| --checkpoint_dir   | Directory for model checkpoints    | "models"        |
| --seed             | Random seed                        | 42              |

#### Outputs
- Best model: `models/best_baseline_resnet18.pth`

---

## 3. Advanced Training Script (`train_full.py`)

See the next section for advanced features: experiment tracking, config files, TensorBoard, and more.

---

## 4. Configuration for Advanced Training

All experiment settings for `train_full.py` can be managed in `train_config.yaml`:

```yaml
data_dir: "data"           # Path to dataset root
epochs: 20                 # Number of training epochs
batch_size: 32             # Batch size
lr: 0.0001                 # Learning rate
optimizer: "adam"          # "adam" or "sgd"
val_split: 0.2             # Fraction of data for validation
checkpoint_dir: "models"   # Directory for model checkpoints
plot_dir: "results"        # Directory for plots and logs
seed: 42                   # Random seed for reproducibility
```

You can override any config value via command-line arguments.

---

## 5. Running Advanced Training

### A. Using the Config File
```bash
python train_full.py --config train_config.yaml
```

### B. Overriding Config with CLI Arguments
```bash
python train_full.py --epochs 30 --batch_size 64 --lr 0.00005
```

### C. All Available Arguments

| Argument           | Description                        | Default         |
|--------------------|------------------------------------|-----------------|
| --data_dir         | Path to dataset root               | "data"          |
| --epochs           | Number of training epochs          | 10              |
| --batch_size       | Batch size                         | 32              |
| --lr               | Learning rate                      | 0.0001          |
| --optimizer        | Optimizer (adam or sgd)            | "adam"          |
| --val_split        | Validation split ratio             | 0.2             |
| --checkpoint_dir   | Directory for model checkpoints    | "models"        |
| --plot_dir         | Directory for plots and logs       | "results"       |
| --seed             | Random seed                        | 42              |
| --config           | Path to YAML config file           | None            |

---

## 6. Outputs & Artifacts (Advanced)

- **Best model:** `models/best_resnet18.pth`
- **Epoch checkpoints:** `models/resnet18_epoch{N}.pth`
- **Plots:** `results/loss_curve.png`, `results/accuracy_curve.png`
- **TensorBoard logs:** `results/tensorboard/`

---

## 7. Monitoring & Visualization

### View Training Progress in TensorBoard
```bash
tensorboard --logdir results/tensorboard/
```
Open the provided URL in your browser to see live metrics, loss/accuracy curves, and more.

---

## 8. Best Practices & Tips

- **Reproducibility:** All runs are seeded. For full reproducibility, log your environment (Python, CUDA, library versions).
- **Experiment Tracking:** Use TensorBoard for comparing runs. For advanced tracking, consider integrating Weights & Biases or MLflow.
- **Hyperparameter Tuning:** Adjust `train_config.yaml` or use CLI overrides for systematic experiments.
- **Checkpoints:** Always use the best model for evaluation.
- **Validation:** Monitor both training and validation metrics to avoid overfitting.
- **Scaling Up:** For large datasets or multi-GPU, adapt DataLoader and training loop as needed.

---

## 9. Troubleshooting

- **CUDA Out of Memory:** Lower batch size or use a smaller model.
- **Slow Training:** Ensure you are using a GPU. Check DataLoader workers.
- **No Plots/Logs:** Ensure `results/` and `models/` directories exist and are writable.
- **Reproducibility Issues:** Double-check random seed and environment versions.

---

## 10. Example Workflow

1. Prepare your dataset in the required folder structure.
2. Edit `train_config.yaml` for your experiment (for advanced training).
3. Run training:
	 ```bash
	 python train_baseline.py --data_dir data
	 # or for advanced
	 python train_full.py --config train_config.yaml
	 ```
4. Monitor progress with TensorBoard (for advanced training).
5. Use the best model and plots for evaluation and reporting.

---

For more details, see the main project documentation or contact the maintainers.
