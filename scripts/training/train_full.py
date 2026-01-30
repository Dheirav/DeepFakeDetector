

import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import argparse
import random
import numpy as np
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import train_transform, val_transform
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml

def get_data_loaders(data_dir, batch_size, val_split=0.2, seed=42):
    dataset = DeepfakeDataset(data_dir, transform=train_transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    data_dir="data",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    optimizer_name="adam",
    val_split=0.2,
    checkpoint_dir="models",
    plot_dir="results",
    seed=42,
    config_path=None
):
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_dir = config.get('data_dir', data_dir)
        epochs = config.get('epochs', epochs)
        batch_size = config.get('batch_size', batch_size)
        lr = config.get('lr', lr)
        optimizer_name = config.get('optimizer', optimizer_name)
        val_split = config.get('val_split', val_split)
        checkpoint_dir = config.get('checkpoint_dir', checkpoint_dir)
        plot_dir = config.get('plot_dir', plot_dir)
        seed = config.get('seed', seed)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(plot_dir, "tensorboard"))

    train_loader, val_loader = get_data_loaders(data_dir, batch_size, val_split, seed)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_resnet18.pth")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/val', val_acc, epoch+1)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"resnet18_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val acc {val_acc:.4f}")

    # Plot loss and accuracy curves
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))

    plt.figure()
    plt.plot(range(1, epochs+1), train_accs, label="Train Acc")
    plt.plot(range(1, epochs+1), val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(plot_dir, "accuracy_curve.png"))

    writer.close()
    print("Training complete. Best model saved at:", best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 for Deepfake Detection")
    parser.add_argument('--data_dir', type=str, default="data", help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer: adam or sgd')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_dir', type=str, default="models", help='Directory to save checkpoints')
    parser.add_argument('--plot_dir', type=str, default="results", help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer_name=args.optimizer,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        seed=args.seed,
        config_path=args.config
    )
