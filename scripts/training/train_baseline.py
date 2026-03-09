
import csv
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import argparse
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import train_transform, val_transform
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm
from training.losses import build_criterion
from tqdm import tqdm


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # deterministic=True forces a safe-but-slow cuDNN path that fires
    # thousands of cudaFuncGetAttributes calls per step on laptop GPUs.
    # benchmark=True lets cuDNN auto-tune once on the first batch (input
    # size is fixed at 224x224), then reuses the fastest kernel every step.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_data_loaders(data_dir, batch_size, val_split=0.2, seed=42):
    # Build train and val datasets with their *correct* transforms.
    # Previously both splits used train_transform (with augmentations) which
    # made val metrics noisy and wasted augmentation compute on validation.
    full_dataset = DeepfakeDataset(data_dir, transform=None)
    val_size  = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Wrap each subset with its own transform so augmentations only hit train.
    train_subset.dataset = DeepfakeDataset(data_dir, transform=train_transform)
    val_subset.dataset   = DeepfakeDataset(data_dir, transform=val_transform)

    # persistent_workers=True keeps worker processes alive between epochs
    # instead of spawning/joining them on every epoch — saves several hundred
    # milliseconds of fork overhead per epoch on a laptop.
    # prefetch_factor=2 means each worker pre-fetches 2 batches ahead so the
    # GPU never sits idle waiting for the next batch to arrive.
    num_workers = 2
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    print(f"Train: {train_size} samples | Val: {val_size} samples")
    return train_loader, val_loader


def train(
    data_dir="dataset_builder/train",
    epochs=5,
    batch_size=32,
    lr=1e-4,
    val_split=0.2,
    checkpoint_dir="models",
    plot_dir="results",
    seed=42,
    early_stop_patience=5,
    loss_type="weighted_focal",
    use_srm=False,
    label_smoothing=0.0,
    run_name=None,
):
    # Create a timestamped run subfolder so each run is self-contained
    run_id = run_name if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_dir, run_id)
    plot_dir       = os.path.join(plot_dir, run_id)
    print(f"Run ID: {run_id}")
    print(f"  Checkpoints -> {checkpoint_dir}")
    print(f"  Plots/logs  -> {plot_dir}")

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"Loss: {loss_type} | SRM: {use_srm} | Label smoothing: {label_smoothing}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(data_dir, batch_size, val_split, seed)

    # Use ResNet18_Weights.DEFAULT instead of the deprecated pretrained=True flag.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 3)

    # ── SRM Filter Layer ────────────────────────────────────────────────────
    # When enabled, prepend a fixed high-pass residual layer before ResNet's
    # first conv.  This gives the network a direct view of manipulation
    # artefacts (edit boundary noise, blending seam frequencies) that are
    # suppressed by standard ImageNet normalisation.
    #
    # The first conv is replaced with a 6-channel version (3 RGB + 3 residual).
    # Pretrained RGB weights are preserved; residual weights start at 0.1× so
    # the model warm-starts from ImageNet features and gradually learns to
    # weight the residual channels.
    if use_srm:
        srm_layer = SRMLayer().to(device)
        model.conv1 = adapt_conv1_for_srm(model.conv1)
        # Wrap model so SRM runs as the first operation in forward()
        class SRMResNet(nn.Module):
            def __init__(self, srm, backbone):
                super().__init__()
                self.srm = srm
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(self.srm(x))
        model = SRMResNet(srm_layer, model)
        print("SRM layer enabled (6-channel input: 3 RGB + 3 residuals)")

    model.to(device)

    # torch.compile fuses element-wise ops and eliminates redundant kernel
    # launches — typically 10-30% faster on PyTorch >= 2.0 with no code changes.
    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    # ── Loss Function ────────────────────────────────────────────────────────
    # weighted_focal (default): per-class weights [1.5, 1.0, 1.5] + focal
    # scaling — forces the optimizer to focus on the hard Real/AI Edited
    # boundary instead of coasting on easy AI Generated examples.
    criterion = build_criterion(loss_type, device, label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ReduceLROnPlateau halves LR when val_loss stops improving, preventing
    # the optimizer from overshooting and oscillating at the bottom of the loss
    # landscape — removes wasted epochs running at too high a LR.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # AMP GradScaler: enables float16 CUDA kernels for conv/matmul ops.
    # Roughly halves memory bandwidth pressure and doubles throughput on the
    # tensor cores present in most NVIDIA laptop GPUs (Turing+).
    # Falls back gracefully to no-op on CPU.
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(checkpoint_dir, "best_baseline_resnet18.pth")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Open CSV for per-epoch metrics
    metrics_csv_path = os.path.join(plot_dir, "metrics.csv")
    metrics_csv_file = open(metrics_csv_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_csv_file)
    metrics_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                              'val_f1_macro', 'f1_real', 'f1_ai_gen', 'f1_ai_edit'])

    for epoch in range(epochs):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # set_to_none=True frees gradient buffers entirely instead of
            # filling them with zeros — cheaper on memory bandwidth.
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct, val_total = 0, 0
        val_all_preds, val_all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total
        val_f1_macro    = f1_score(val_all_labels, val_all_preds, average='macro')
        val_f1_per_class = f1_score(val_all_labels, val_all_preds, average=None, labels=[0, 1, 2])

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1_macro:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Step scheduler on val loss so LR drops when the model plateaus.
        scheduler.step(val_loss)

        # Log to CSV
        metrics_writer.writerow([
            epoch + 1,
            round(train_loss, 6), round(train_acc, 6),
            round(val_loss, 6), round(val_acc, 6),
            round(val_f1_macro, 6),
            round(float(val_f1_per_class[0]), 6),
            round(float(val_f1_per_class[1]), 6),
            round(float(val_f1_per_class[2]), 6),
        ])
        metrics_csv_file.flush()

        # ── Checkpoint ──────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f}, val_f1={val_f1_macro:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs).")
                break

    metrics_csv_file.close()
    print(f"Per-epoch metrics saved to: {metrics_csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    actual_epochs = len(train_losses)
    plt.figure()
    plt.plot(range(1, actual_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, actual_epochs + 1), val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, actual_epochs + 1), train_accs, label="Train Acc")
    plt.plot(range(1, actual_epochs + 1), val_accs,   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy Curve")
    plt.savefig(os.path.join(plot_dir, "accuracy_curve.png"))
    plt.close()

    # ── Training summary JSON ────────────────────────────────────────────────
    summary = {
        "run_id": run_id,
        "best_val_acc":    round(float(best_val_acc), 4),
        "best_val_loss":   round(float(best_val_loss), 4),
        "epochs_trained":  actual_epochs,
        "final_train_acc": round(float(train_accs[-1]), 4),
        "final_val_acc":   round(float(val_accs[-1]), 4),
        "config": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "loss_type": loss_type, "use_srm": use_srm,
            "label_smoothing": label_smoothing, "seed": seed,
        },
    }
    summary_path = os.path.join(plot_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}")
    print(f"Training complete. Best val acc: {best_val_acc:.4f}. Model: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline ResNet18 for Deepfake Detection")
    parser.add_argument('--data_dir',            type=str,   default="dataset_builder/train",   help='Dataset directory (expects real/, ai_generated/, ai_edited/ subfolders)')
    parser.add_argument('--epochs',              type=int,   default=5,        help='Max epochs')
    parser.add_argument('--batch_size',          type=int,   default=32,       help='Batch size (try 64 if VRAM allows)')
    parser.add_argument('--lr',                  type=float, default=1e-4,     help='Initial learning rate')
    parser.add_argument('--val_split',           type=float, default=0.2,      help='Validation split ratio')
    parser.add_argument('--checkpoint_dir',      type=str,   default="models", help='Checkpoint save directory')
    parser.add_argument('--seed',                type=int,   default=42,       help='Random seed')
    parser.add_argument('--early_stop_patience', type=int,   default=5,        help='Early stopping patience (epochs)')
    parser.add_argument('--loss', type=str, default='weighted_focal',
                        choices=['ce', 'weighted', 'focal', 'weighted_focal'],
                        help=(
                            'Loss function: '
                            'ce=standard CrossEntropy (original baseline), '
                            'weighted=CE with class weights [1.5,1.0,1.5], '
                            'focal=FocalLoss(gamma=2) no weights, '
                            'weighted_focal=FocalLoss+weights (recommended)'
                        ))
    parser.add_argument('--use_srm', action='store_true',
                        help=(
                            'Enable SRM (Steganalysis Rich Model) high-pass filter layer. '
                            'Extracts 3 manipulation-residual channels and concatenates with RGB '
                            '(6-channel input). Targets high-frequency artefacts left by AI editing.'
                        ))
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.1 is a common default)')
    parser.add_argument('--plot_dir', type=str, default='results',
                        help='Directory to save plots and logs')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this run (subfolder, e.g. "exp1_srm"). '
                             'Defaults to a timestamp like run_20260307_153045.')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        loss_type=args.loss,
        use_srm=args.use_srm,
        label_smoothing=args.label_smoothing,
        run_name=args.run_name,
    )
