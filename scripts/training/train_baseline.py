
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import argparse
import os
import sys
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import train_transform, val_transform
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
    seed=42,
    early_stop_patience=5,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(data_dir, batch_size, val_split, seed)

    # Use ResNet18_Weights.DEFAULT instead of the deprecated pretrained=True flag.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 3)
    model.to(device)

    # torch.compile fuses element-wise ops and eliminates redundant kernel
    # launches — typically 10-30% faster on PyTorch >= 2.0 with no code changes.
    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ReduceLROnPlateau halves LR when val_loss stops improving, preventing
    # the optimizer from overshooting and oscillating at the bottom of the loss
    # landscape — removes wasted epochs running at too high a LR.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # AMP GradScaler: enables float16 CUDA kernels for conv/matmul ops.
    # Roughly halves memory bandwidth pressure and doubles throughput on the
    # tensor cores present in most NVIDIA laptop GPUs (Turing+).
    # Falls back gracefully to no-op on CPU.
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(checkpoint_dir, "best_baseline_resnet18.pth")

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

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Step scheduler on val loss so LR drops when the model plateaus.
        scheduler.step(val_loss)

        # ── Checkpoint ──────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs).")
                break

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
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
    )
