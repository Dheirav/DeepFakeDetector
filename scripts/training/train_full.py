
# === Standard Library Imports ===
import os
import csv
import json
import random
import argparse
import platform
import yaml
from datetime import datetime

# === Third-Party Imports ===
import numpy as np
import torch
import torch.nn as nn
import torch.profiler
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    EfficientNet_B3_Weights,
    ViT_B_16_Weights
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# === Albumentations (Augmentations) ===
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === System Monitoring (Optional) ===
try:
    import psutil
except ImportError:
    psutil = None
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
    pynvml_available = True
except ImportError:
    pynvml_available = False

# === Project Imports ===
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import train_transform, val_transform, get_train_transform
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm
from training.losses import build_criterion

def _build_backbone(backbone_name, num_classes, dropout_p):
    """Build a pretrained backbone with a custom classification head."""
    def make_head(in_features):
        if dropout_p > 0:
            return nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
        return nn.Linear(in_features, num_classes)

    if backbone_name == "resnet18":
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = make_head(m.fc.in_features)  # 512 → num_classes
        return m
    elif backbone_name == "convnext_tiny":
        m = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = make_head(m.classifier[2].in_features)  # 768 → num_classes
        return m
    elif backbone_name == "resnet50":
        m = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        m.fc = make_head(m.fc.in_features)  # 2048 → num_classes
        return m
    elif backbone_name == "convnext_small":
        m = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        m.classifier[2] = make_head(m.classifier[2].in_features)  # 768 → num_classes
        return m
    elif backbone_name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        m.classifier[1] = make_head(m.classifier[1].in_features)  # 1536 → num_classes
        return m
    elif backbone_name == "vit_b_16":
        m = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        m.heads.head = make_head(m.heads.head.in_features)  # 768 → num_classes
        return m
    else:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose: resnet18, convnext_tiny")


def get_data_loaders(train_dir, batch_size, val_dir=None, val_split=0.2, seed=42, augment="standard"):
    t_transform = get_train_transform(augment)

    train_dataset = DeepfakeDataset(train_dir, transform=t_transform)

    if val_dir and os.path.isdir(val_dir):
        # Use the pre-built validation split
        val_dataset = DeepfakeDataset(val_dir, transform=val_transform)
        print(f"Using pre-built val dir: {val_dir} ({len(val_dataset)} samples)")
    else:
        # Fall back to random split from training data
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        print(f"No val_dir found — using random {val_split*100:.0f}% val split")

    # On a laptop, 4 workers compete with the training process.
    # 2 workers + persistent_workers avoids respawning overhead each epoch.
    # prefetch_factor=2 keeps the next 2 batches ready in the background.
    num_workers = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # deterministic=True + benchmark=False disables cuDNN auto-tuning and causes
    # thousands of cudaFuncGetAttributes calls per step. Since our input size is
    # fixed (224x224), benchmark=True lets cuDNN cache the fastest kernel once
    # and reuse it every step — eliminates most of that overhead.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def train(
    data_dir="dataset_builder/train",
    val_dir="dataset_builder/val",
    epochs=50,
    batch_size=64,
    lr=1e-4,
    optimizer_name="adam",
    weight_decay=1e-4,
    val_split=0.2,
    checkpoint_dir="models",
    plot_dir="results",
    seed=42,
    config_path=None,
    light_augment=False,  # kept for YAML backward compat — prefer augment=
    augment="standard",
    run_name=None,
    use_srm=False,
    loss_type="weighted_focal",
    label_smoothing=0.1,
    early_stop_patience=7,
    enable_profiler=False,
    class_weights=None,
    dropout_p=0.4,
    lr_schedule="cosine",
    backbone="resnet18",
    focal_gamma=2.0,
):
    # Load config if provided (single read)
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_dir       = config.get('data_dir', data_dir)
        val_dir        = config.get('val_dir', val_dir)
        epochs         = config.get('epochs', epochs)
        batch_size     = config.get('batch_size', batch_size)
        lr             = config.get('lr', lr)
        optimizer_name = config.get('optimizer', optimizer_name)
        weight_decay   = config.get('weight_decay', weight_decay)
        val_split      = config.get('val_split', val_split)
        checkpoint_dir = config.get('checkpoint_dir', checkpoint_dir)
        plot_dir       = config.get('plot_dir', plot_dir)
        seed           = config.get('seed', seed)
        light_augment  = config.get('light_augment', light_augment)
        augment        = config.get('augment', augment)
        run_name       = config.get('run_name', run_name)
        use_srm        = config.get('use_srm', use_srm)
        loss_type      = config.get('loss_type', loss_type)
        label_smoothing = config.get('label_smoothing', label_smoothing)
        early_stop_patience = config.get('early_stop_patience', early_stop_patience)
        enable_profiler  = config.get('enable_profiler', enable_profiler)
        class_weights    = config.get('class_weights', class_weights)
        dropout_p        = config.get('dropout_p', dropout_p)
        lr_schedule      = config.get('lr_schedule', lr_schedule)
        focal_gamma      = config.get('focal_gamma', focal_gamma)
        backbone         = config.get('backbone', backbone)

    # light_augment flag is a legacy alias — map it into the augment string
    if light_augment:
        augment = "light"

    run_id = run_name if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_dir, run_id)
    plot_dir       = os.path.join(plot_dir, run_id)
    print(f"Run ID: {run_id}")
    print(f"  Checkpoints -> {checkpoint_dir}")
    print(f"  Plots/logs  -> {plot_dir}")

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"Loss: {loss_type} | SRM: {use_srm} | Label smoothing: {label_smoothing} | WD: {weight_decay}")
    print(f"Class weights: {class_weights if class_weights else '[1.5, 1.0, 1.5] (default)'}")
    print(f"Backbone: {backbone} | Dropout (FC head): {dropout_p} | LR schedule: {lr_schedule} | Focal gamma: {focal_gamma} | Augment: {augment}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Early stopping state
    epochs_no_improve   = 0

    # Initialize GPU monitoring if available
    if pynvml_available:
        nvmlInit()
        gpu_handle = nvmlDeviceGetHandleByIndex(0)

    writer = SummaryWriter(log_dir=os.path.join(plot_dir, "tensorboard"))

    train_loader, val_loader = get_data_loaders(
        train_dir=data_dir,
        batch_size=batch_size,
        val_dir=val_dir,
        val_split=val_split,
        seed=seed,
        augment=augment
    )

    model = _build_backbone(backbone, num_classes=3, dropout_p=dropout_p)

    # ── SRM Filter Layer ──────────────────────────────────────────────────
    if use_srm:
        srm_layer = SRMLayer().to(device)
        if backbone == "resnet18":
            model.conv1 = adapt_conv1_for_srm(model.conv1)
        elif backbone == "convnext_tiny":
            model.features[0][0] = adapt_conv1_for_srm(model.features[0][0])
        class SRMNet(nn.Module):
            def __init__(self, srm, bb):
                super().__init__()
                self.srm = srm
                self.backbone = bb
            def forward(self, x):
                return self.backbone(self.srm(x))
        model = SRMNet(srm_layer, model)
        print(f"SRM layer enabled on {backbone} (6-channel input: 3 RGB + 3 residuals)")

    model.to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    criterion = build_criterion(loss_type, device, label_smoothing, class_weights, gamma=focal_gamma)
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

    if lr_schedule == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Open CSV for per-epoch metrics logging
    metrics_csv_path = os.path.join(plot_dir, "metrics.csv")
    os.makedirs(plot_dir, exist_ok=True)
    metrics_csv_file = open(metrics_csv_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_csv_file)
    metrics_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                              'val_f1_macro', 'f1_real', 'f1_ai_gen', 'f1_ai_edit'])

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        # --- Resource Monitoring ---
        sys_stats = {}
        if psutil:
            sys_stats['cpu_percent'] = psutil.cpu_percent(interval=0.5)
            sys_stats['ram_percent'] = psutil.virtual_memory().percent
        if pynvml_available:
            meminfo = nvmlDeviceGetMemoryInfo(gpu_handle)
            util = nvmlDeviceGetUtilizationRates(gpu_handle)
            sys_stats['gpu_mem_used_MB'] = meminfo.used // (1024*1024)
            sys_stats['gpu_mem_total_MB'] = meminfo.total // (1024*1024)
            sys_stats['gpu_util_percent'] = util.gpu
        # Log to console and TensorBoard
        print(f"[Resource] Epoch {epoch+1}: {sys_stats}")
        for k, v in sys_stats.items():
            writer.add_scalar(f"Resource/{k}", v, epoch+1)

        if epoch == 0 and enable_profiler:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(plot_dir, "profiler")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for images, labels in loop:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    prof.step()
            print("\n[PyTorch Profiler] First epoch summary:")
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        else:
            for images, labels in loop:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_acc, epoch+1)

        # Validation
        model.eval()
        val_loss = 0
        val_correct, val_total = 0, 0
        val_all_preds, val_all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1_macro = f1_score(val_all_labels, val_all_preds, average='macro')
        val_f1_per_class = f1_score(val_all_labels, val_all_preds, average=None, labels=[0,1,2])
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/val', val_acc, epoch+1)
        writer.add_scalar('F1_macro/val', val_f1_macro, epoch+1)
        writer.add_scalar('F1/real',         val_f1_per_class[0], epoch+1)
        writer.add_scalar('F1/ai_generated',  val_f1_per_class[1], epoch+1)
        writer.add_scalar('F1/ai_edited',     val_f1_per_class[2], epoch+1)

        if lr_schedule == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1_macro:.4f} "
              f"[real={val_f1_per_class[0]:.3f} ai_gen={val_f1_per_class[1]:.3f} ai_edit={val_f1_per_class[2]:.3f}]")

        # Log per-epoch metrics to CSV
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

        # Save checkpoint, keep only last checkpoint_path = os.path.join(checkpoint_dir, f"resnet18_epoch{epoch+1}.pth")3 to save disk
        
        torch.save(model.state_dict(), checkpoint_path)
        old_ckpt = os.path.join(checkpoint_dir,f"{backbone}_epoch{epoch+1}.pth")
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✔ Best model saved (epoch {epoch+1}, val acc {val_acc:.4f}, val F1 {val_f1_macro:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    metrics_csv_file.close()
    print(f"Per-epoch metrics saved to: {metrics_csv_path}")

    # Plot loss and accuracy curves
    actual_epochs = len(train_losses)
    plt.figure()
    plt.plot(range(1, actual_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, actual_epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))

    plt.figure()
    plt.plot(range(1, actual_epochs+1), train_accs, label="Train Acc")
    plt.plot(range(1, actual_epochs+1), val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(plot_dir, "accuracy_curve.png"))

    writer.close()
    if pynvml_available:
        nvmlShutdown()

    # Save training summary
    summary = {
        "best_val_acc":   round(float(best_val_acc), 4),
        "epochs_trained": len(train_losses),
        "final_train_acc": round(float(train_accs[-1]), 4),
        "final_val_acc":  round(float(val_accs[-1]), 4),
        "config": {
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "optimizer": optimizer_name, "weight_decay": weight_decay,
            "loss_type": loss_type, "use_srm": use_srm,
            "label_smoothing": label_smoothing, "seed": seed,
            "class_weights": class_weights if class_weights else [1.5, 1.0, 1.5],
            "dropout_p": dropout_p, "lr_schedule": lr_schedule, "backbone": backbone,
            "focal_gamma": focal_gamma, "augment": augment,
        }
    }
    summary_path = os.path.join(plot_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}")
    print("Training complete. Best model saved at:", best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 for Deepfake Detection")
    parser.add_argument('--data_dir', type=str, default="dataset_builder/train", help='Training data directory')
    parser.add_argument('--val_dir', type=str, default="dataset_builder/val", help='Validation data directory (optional, uses val_split if not set)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer: adam or sgd')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (ignored if --val_dir set)')
    parser.add_argument('--checkpoint_dir', type=str, default="models", help='Directory to save checkpoints')
    parser.add_argument('--plot_dir', type=str, default="results", help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--light_augment', action='store_true',
                        help='[Legacy] Alias for --augment light')
    parser.add_argument('--augment', type=str, default='standard',
                        choices=['light', 'standard', 'strong'],
                        help='Augmentation level: light (flip only), standard (default), '
                             'strong (wider JPEG + noise + blur, targets Real/AI-Edit boundary)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this run (used as subfolder, e.g. "exp1_lr1e4"). '
                             'Defaults to a timestamp like run_20260307_153045.')
    parser.add_argument('--use_srm', action='store_true',
                        help='Enable SRM high-pass filter layer (6-channel input: 3 RGB + 3 residuals)')
    parser.add_argument('--loss', type=str, default='weighted_focal',
                        choices=['ce', 'weighted', 'focal', 'weighted_focal'],
                        help='Loss function (default: weighted_focal)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Optimizer weight decay (default: 1e-4)')
    parser.add_argument('--early_stop_patience', type=int, default=7,
                        help='Early stopping patience in epochs (default: 7)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable PyTorch profiler on epoch 1 (slow — use for debugging only)')
    parser.add_argument('--class_weights', type=float, nargs=3,
                        metavar=('W_REAL', 'W_AIGEN', 'W_AIEDIT'),
                        default=None,
                        help='Per-class loss weights [real, ai_gen, ai_edit]. '
                             'Default: 1.5 1.0 1.5. Example: --class_weights 2.0 1.0 1.5')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout probability before FC head (default: 0.4, set 0.0 to disable)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'convnext_tiny', 'convnext_small', 'efficientnet_b3', 'vit_b_16'],
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='LR scheduler: cosine (CosineAnnealingWarmRestarts) or plateau (ReduceLROnPlateau)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma focusing parameter (default: 2.0, try 3.0 for harder boundary focus)')
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        seed=args.seed,
        config_path=args.config,
        light_augment=args.light_augment,
        augment=args.augment,
        run_name=args.run_name,
        use_srm=args.use_srm,
        loss_type=args.loss,
        label_smoothing=args.label_smoothing,
        early_stop_patience=args.early_stop_patience,
        enable_profiler=args.profile,
        class_weights=args.class_weights,
        dropout_p=args.dropout,
        lr_schedule=args.lr_schedule,
        backbone=args.backbone,
        focal_gamma=args.focal_gamma,
    )
