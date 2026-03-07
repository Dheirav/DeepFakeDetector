import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import val_transform

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

# Project root = two levels above this script (scripts/evaluation/ → root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _derive_save_dir(model_path: str) -> str:
    """
    Derive the results directory from the model checkpoint path.

    If the path contains a run folder matching run_YYYYMMDD_HHMMSS (e.g.
    models/run_20260307_063053/best_resnet18.pth), the results are saved to
    results/run_20260307_063053/ so all artefacts from the same run stay together.

    Falls back to results/ at the project root if no run ID is found.
    """
    match = re.search(r"(run_\d{8}_\d{6})", os.path.abspath(model_path))
    if match:
        return os.path.join(_PROJECT_ROOT, "results", match.group(1))
    return os.path.join(_PROJECT_ROOT, "results")

def load_model(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_evaluation(model_path, data_dir, batch_size=64, save_dir="../results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)

    dataset = DeepfakeDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(dataset)} | Batches: {len(loader)}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Only report on classes that are actually present in the dataset so
    # classification_report doesn't fail when some splits are missing.
    present_labels = sorted(set(y_true))
    present_names  = [CLASS_NAMES[i] for i in present_labels]

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, digits=4))

    print("CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred, labels=present_labels))

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
    print(f"\nSaved y_true.npy and y_pred.npy to {save_dir}")
    print(f"Overall accuracy: {(y_true == y_pred).mean()*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model on test set")
    parser.add_argument('--model_path',  type=str, required=True,
                        help='Path to model .pth checkpoint')
    parser.add_argument('--data_dir',    type=str, default="dataset_builder/test",
                        help='Test data directory (with real/, ai_generated/, ai_edited/ subfolders)')
    parser.add_argument('--batch_size',  type=int, default=64, help='Batch size')
    parser.add_argument('--save_dir',    type=str, default=None,
                        help=(
                            'Directory to save y_true.npy and y_pred.npy. '
                            'Defaults to results/<run_id>/ derived from --model_path, '
                            'or results/ if no run ID is found in the checkpoint path.'
                        ))
    args = parser.parse_args()

    # Auto-derive save_dir from the checkpoint path when not explicitly set.
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else _derive_save_dir(args.model_path)
    print(f"Results will be saved to: {save_dir}")

    run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        save_dir=save_dir,
    )
