
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import train_transform, val_transform
from tqdm import tqdm

def set_seed(seed=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def get_data_loaders(data_dir, batch_size, val_split=0.2, seed=42):
	dataset = DeepfakeDataset(data_dir, transform=train_transform)
	val_size = int(len(dataset) * val_split)
	train_size = len(dataset) - val_size
	generator = torch.Generator().manual_seed(seed)
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
	return train_loader, val_loader

def train(
	data_dir="data",
	epochs=5,
	batch_size=32,
	lr=1e-4,
	val_split=0.2,
	checkpoint_dir="models",
	seed=42
):
	set_seed(seed)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	os.makedirs(checkpoint_dir, exist_ok=True)

	train_loader, val_loader = get_data_loaders(data_dir, batch_size, val_split, seed)

	model = models.resnet18(pretrained=True)
	model.fc = nn.Linear(512, 3)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	best_val_acc = 0.0
	best_model_path = os.path.join(checkpoint_dir, "best_baseline_resnet18.pth")

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

		print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

		# Save best model
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), best_model_path)
			print(f"Best model saved at epoch {epoch+1} with val acc {val_acc:.4f}")

	print("Training complete. Best model saved at:", best_model_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train Baseline ResNet18 for Deepfake Detection")
	parser.add_argument('--data_dir', type=str, default="data", help='Dataset directory')
	parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
	parser.add_argument('--checkpoint_dir', type=str, default="models", help='Directory to save checkpoints')
	parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
	args = parser.parse_args()
	train(
		data_dir=args.data_dir,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		val_split=args.val_split,
		checkpoint_dir=args.checkpoint_dir,
		seed=args.seed
	)
