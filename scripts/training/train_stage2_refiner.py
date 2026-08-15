"""Train a Stage-2 binary refiner: Real (0) vs AI-Edited (1).

This script reuses the training conventions of `train_full.py` while
limiting the dataset to the two classes required by the refiner. It is
designed to be a near-drop-in parity implementation with options for
SRM/FFT preprocessing, attention heads, schedulers, and logging.

Checkpoints are saved under `models/stage2_refiner/<run_name>/best_model.pth`.
"""
import os
import csv
import json
import random
import argparse
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import get_train_transform, val_transform
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm
from preprocessing.fft import FFTLayer
from training.losses import build_criterion
from modules.attention_heads import GeM, CBAMBlock
from training.train_full import _build_backbone, set_seed


class BinaryWrapperDataset(torch.utils.data.Dataset):
    """Wrap DeepfakeDataset and remap labels: real->0, ai_edited->1.

    The underlying dataset should only contain the two classes (real, ai_edited).
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # original mapping: 0=real,1=ai_generated,2=ai_edited
        if int(label) == 0:
            return img, torch.tensor(0)
        if int(label) == 2:
            return img, torch.tensor(1)
        # Shouldn't happen if dataset filtered correctly
        raise RuntimeError(f"Unexpected label for binary refiner: {label}")


def train(
    data_dir="dataset_builder/train",
    val_dir="dataset_builder/val",
    epochs=50,
    batch_size=64,
    lr=1e-4,
    optimizer_name="adam",
    weight_decay=1e-4,
    val_split=0.2,
    seed=42,
    config_path=None,
    augment="standard",
    run_name=None,
    use_srm=False,
    use_fft=False,
    loss_type="weighted_focal",
    label_smoothing=0.0,
    early_stop_patience=7,
    enable_profiler=False,
    class_weights=None,
    dropout_p=0.4,
    lr_schedule="cosine",
    backbone="resnet18",
    focal_gamma=3.0,
    attention_head="none",
    gem_p=3.0,
    gem_learnable=False,
    cbam_reduction=16,
    cbam_kernel=7,
):
    # Load config if provided (single read)
    if config_path:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        data_dir = cfg.get('data_dir', data_dir)
        val_dir = cfg.get('val_dir', val_dir)
        epochs = cfg.get('epochs', epochs)
        batch_size = cfg.get('batch_size', batch_size)
        lr = cfg.get('lr', lr)
        optimizer_name = cfg.get('optimizer', optimizer_name)
        weight_decay = cfg.get('weight_decay', weight_decay)
        val_split = cfg.get('val_split', val_split)
        seed = cfg.get('seed', seed)
        augment = cfg.get('augment', augment)
        run_name = cfg.get('run_name', run_name)
        use_srm = cfg.get('use_srm', use_srm)
        use_fft = cfg.get('use_fft', use_fft)
        loss_type = cfg.get('loss_type', loss_type)
        label_smoothing = cfg.get('label_smoothing', label_smoothing)
        early_stop_patience = cfg.get('early_stop_patience', early_stop_patience)
        class_weights = cfg.get('class_weights', class_weights)
        dropout_p = cfg.get('dropout_p', dropout_p)
        lr_schedule = cfg.get('lr_schedule', lr_schedule)
        focal_gamma = cfg.get('focal_gamma', focal_gamma)
        backbone = cfg.get('backbone', backbone)
        attention_head = cfg.get('attention_head', attention_head)
        gem_p = cfg.get('gem_p', gem_p)
        gem_learnable = cfg.get('gem_learnable', gem_learnable)
        cbam_reduction = cfg.get('cbam_reduction', cbam_reduction)
        cbam_kernel = cfg.get('cbam_kernel', cbam_kernel)

    run_id = run_name if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("models", "stage2_refiner", run_id)
    plot_dir = os.path.join("results", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Run ID: {run_id}")
    print(f"  Checkpoints -> {checkpoint_dir}")
    print(f"  Plots/logs  -> {plot_dir}")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # Transforms
    t_transform = get_train_transform(augment)
    v_transform = val_transform

    # Datasets (filtered to real & ai_edited)
    train_dataset = DeepfakeDataset(data_dir, transform=t_transform, include_classes=["real", "ai_edited"]) 
    val_dataset = DeepfakeDataset(val_dir, transform=v_transform, include_classes=["real", "ai_edited"]) if val_dir and os.path.isdir(val_dir) else None

    train_dataset = BinaryWrapperDataset(train_dataset)
    if val_dataset is not None:
        val_dataset = BinaryWrapperDataset(val_dataset)

    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2) if val_dataset is not None else None

    # Build model
    model = _build_backbone(
        backbone,
        num_classes=2,
        dropout_p=dropout_p,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )

    # SRM / FFT preprocessing wrappers
    srm_layer = None
    fft_layer = None

    class PreprocessNet(nn.Module):
        def __init__(self, srm, fft, backbone):
            super().__init__()
            self.srm = srm
            self.fft = fft
            self.backbone = backbone

        def forward(self, x):
            rgb = x
            features = [rgb]
            if self.srm is not None:
                srm_out = self.srm(rgb)
                if srm_out.shape[1] == rgb.shape[1] * 2:
                    residuals = srm_out[:, rgb.shape[1]:, ...]
                    features.append(residuals)
                else:
                    features.append(srm_out)
            if self.fft is not None:
                features.append(self.fft(rgb))
            x = torch.cat(features, dim=1)
            return self.backbone(x)

    if use_srm:
        srm_layer = SRMLayer().to(device)
    if use_fft:
        fft_layer = FFTLayer().to(device)

    if use_srm or use_fft:
        # adapt first conv channels for srm/fft if necessary
        in_ch = 3 + (3 if use_srm else 0) + (1 if use_fft else 0)
        try:
            if backbone in ["resnet18", "resnet50"]:
                model.conv1 = adapt_conv1_for_srm(model.conv1, in_ch)
            else:
                # attempt replace first conv in ConvNeXt
                stem = model.features[0]
                def _replace_first_conv(module, target_in_channels):
                    for name, child in module.named_children():
                        if isinstance(child, nn.Conv2d):
                            setattr(module, name, adapt_conv1_for_srm(child, target_in_channels))
                            return True
                        if _replace_first_conv(child, target_in_channels):
                            return True
                    return False
                _replace_first_conv(stem, in_ch)
        except Exception:
            pass
        model = PreprocessNet(srm_layer, fft_layer, model)

    model.to(device)

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass

    # Class weights
    if class_weights is None:
        binary_weights = [1.5, 1.5]
    else:
        if len(class_weights) == 3:
            binary_weights = [class_weights[0], class_weights[2]]
        elif len(class_weights) == 2:
            binary_weights = class_weights
        else:
            raise ValueError("class_weights must be length 2 or 3")

    criterion = build_criterion(loss_type, device, label_smoothing, binary_weights, gamma=focal_gamma)

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

    # Logging
    writer = SummaryWriter(log_dir=os.path.join(plot_dir, "tensorboard"))
    metrics_csv_path = os.path.join(plot_dir, "metrics.csv")
    metrics_csv_file = open(metrics_csv_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_csv_file)
    metrics_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    best_val = 0.0
    best_path = os.path.join(checkpoint_dir, "best_model.pth")

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', enabled=(device=='cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total if running_total else 0.0
        train_acc = running_correct / running_total if running_total else 0.0

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            v_running_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    v_running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
            val_loss = v_running_loss / v_total if v_total else 0.0
            val_acc = v_correct / v_total if v_total else 0.0

        # Scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            try:
                scheduler.step()
            except Exception:
                pass

        # Logging
        metrics_writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
        metrics_csv_file.flush()
        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Train/Acc', train_acc, epoch+1)
        writer.add_scalar('Val/Loss', val_loss, epoch+1)
        writer.add_scalar('Val/Acc', val_acc, epoch+1)

        print(f"Epoch {epoch+1}: Train acc: {train_acc*100:.2f}% | Val acc: {val_acc*100:.2f}%")

        # Checkpoint
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model ({best_val*100:.2f}%) to {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping: no improvement in {early_stop_patience} epochs")
                break

    # finalize
    metrics_csv_file.close()
    writer.close()

    # write training summary
    summary = {
        'run_id': run_id,
        'config': {
            'data_dir': data_dir,
            'val_dir': val_dir,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer_name,
            'weight_decay': weight_decay,
            'backbone': backbone,
            'dropout_p': dropout_p,
            'loss_type': loss_type,
            'label_smoothing': label_smoothing,
            'class_weights': binary_weights,
            'attention_head': attention_head,
            'use_srm': use_srm,
            'use_fft': use_fft,
        }
    }
    with open(os.path.join(plot_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Stage-2 refiner (Real vs AI-Edited)")
    parser.add_argument('--data_dir', type=str, default='dataset_builder/train')
    parser.add_argument('--val_dir', type=str, default='dataset_builder/val')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--augment', type=str, default='standard')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--use_srm', action='store_true')
    parser.add_argument('--use_fft', action='store_true')
    parser.add_argument('--loss_type', type=str, default='weighted_focal')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--early_stop_patience', type=int, default=7)
    parser.add_argument('--enable_profiler', action='store_true')
    parser.add_argument('--class_weights', type=float, nargs='*', default=None)
    parser.add_argument('--dropout_p', type=float, default=0.4)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--attention_head', type=str, default='none', choices=['none','gem','cbam'])
    parser.add_argument('--gem_p', type=float, default=3.0)
    parser.add_argument('--gem_learnable', action='store_true')
    parser.add_argument('--cbam_reduction', type=int, default=16)
    parser.add_argument('--cbam_kernel', type=int, default=7)

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
        seed=args.seed,
        config_path=args.config_path,
        augment=args.augment,
        run_name=args.run_name,
        use_srm=args.use_srm,
        use_fft=args.use_fft,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        early_stop_patience=args.early_stop_patience,
        enable_profiler=args.enable_profiler,
        class_weights=args.class_weights,
        dropout_p=args.dropout_p,
        lr_schedule=args.lr_schedule,
        backbone=args.backbone,
        focal_gamma=args.focal_gamma,
        attention_head=args.attention_head,
        gem_p=args.gem_p,
        gem_learnable=args.gem_learnable,
        cbam_reduction=args.cbam_reduction,
        cbam_kernel=args.cbam_kernel,
    )
