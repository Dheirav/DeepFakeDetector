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
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

# Project root = two levels above this script (scripts/evaluation/ → root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _derive_save_dir(model_path: str) -> str:
    """
    Derive the results directory from the model checkpoint path.

    Priority:
    1. If the path contains a timestamped run folder (run_YYYYMMDD_HHMMSS),
       mirror it under results/ so all artefacts from the same training run
       stay together (e.g. results/run_20260307_063053/).
    2. If the model lives inside a named subfolder (e.g. models/my_exp/best.pth),
       use that folder name (e.g. results/my_exp/).
    3. Otherwise fall back to results/<checkpoint_stem>/ derived from the
       filename itself (e.g. models/best_resnet18.pth → results/best_resnet18/).
    """
    abs_path = os.path.abspath(model_path)

    # 1. Timestamped run folder
    match = re.search(r"(run_\d{8}_\d{6})", abs_path)
    if match:
        return os.path.join(_PROJECT_ROOT, "results", match.group(1))

    # 2. Named parent folder (anything that isn't the bare models/ root)
    parent = os.path.basename(os.path.dirname(abs_path))
    models_root = os.path.basename(
        os.path.abspath(os.path.join(_PROJECT_ROOT, "models"))
    )
    if parent and parent != models_root:
        return os.path.join(_PROJECT_ROOT, "results", parent)

    # 3. Checkpoint filename stem
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    return os.path.join(_PROJECT_ROOT, "results", stem)

def _build_srm_net(device, sequential_fc=False, is_convnext=False):
    """Reconstruct the SRM-wrapped model used during training."""
    def make_fc(in_features):
        if sequential_fc:
            return nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, 3))
        return nn.Linear(in_features, 3)

    if is_convnext:
        from torchvision.models import ConvNeXt_Tiny_Weights
        backbone = models.convnext_tiny(weights=None)
        backbone.classifier[2] = make_fc(backbone.classifier[2].in_features)
        srm_layer = SRMLayer().to(device)
        backbone.features[0][0] = adapt_conv1_for_srm(backbone.features[0][0])
    else:
        backbone = models.resnet18(weights=None)
        backbone.fc = make_fc(512)
        srm_layer = SRMLayer().to(device)
        backbone.conv1 = adapt_conv1_for_srm(backbone.conv1)

    class SRMNet(nn.Module):
        def __init__(self, srm, bb):
            super().__init__()
            self.srm = srm
            self.backbone = bb
        def forward(self, x):
            return self.backbone(self.srm(x))

    return SRMNet(srm_layer, backbone)


def load_model(model_path, device):
    raw_sd = torch.load(model_path, map_location=device)

    # torch.compile wraps every key with "_orig_mod." — strip it.
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}

    # Detect architecture from state dict keys
    has_srm      = any(k.startswith("backbone.") for k in raw_sd)
    is_convnext  = any(k.startswith("backbone.features.") for k in raw_sd)
    # Detect Sequential FC head: key ends with "fc.1.weight" (ResNet) or "classifier.2.1.weight" (ConvNeXt)
    sequential_fc = any(k.endswith("fc.1.weight") or k.endswith("classifier.2.1.weight") for k in raw_sd)

    if has_srm:
        arch = "ConvNeXt-Tiny" if is_convnext else "ResNet-18"
        fc_label = "Sequential(Dropout+Linear)" if sequential_fc else "Linear"
        print(f"Detected SRM checkpoint — rebuilding SRMNet ({arch}, FC: {fc_label}).")
        model = _build_srm_net(device, sequential_fc=sequential_fc, is_convnext=is_convnext)
    elif is_convnext or any(k.startswith("features.") for k in raw_sd):
        from torchvision.models import ConvNeXt_Tiny_Weights
        print("Detected plain ConvNeXt-Tiny checkpoint.")
        model = models.convnext_tiny(weights=None)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_feat, 3)) if sequential_fc else nn.Linear(in_feat, 3)
    else:
        model = models.resnet18(weights=None)
        if sequential_fc:
            model.fc = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(512, 3))
        else:
            model.fc = nn.Linear(512, 3)

    model.load_state_dict(raw_sd)
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
