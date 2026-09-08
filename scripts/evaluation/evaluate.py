import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import val_transform
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm
from preprocessing.fft import FFTLayer
from modules.attention_heads import GeM, CBAMBlock
from modules.cascade_classifier import CascadeClassifier

import sys
# Ensure the `scripts/` directory is first on sys.path so imports like
# `modules.*` resolve to `scripts/modules/*`. Add the repo root after it
# so top-level packages do not shadow the `scripts` module namespace.
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
if repo_root not in sys.path:
    # place repo_root after scripts_dir to avoid shadowing
    sys.path.insert(1, repo_root)


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

def load_model(
    model_path,
    device,
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
):
    """Load a checkpoint into the architecture its config describes.

    Delegates to ``modules.model_builder``. The previous implementation inferred
    the architecture from state-dict key names and loaded with ``strict=False``
    behind a bare ``except: pass``. Both silent-failure modes it produced are
    documented in ``results/README.md``:

    * ``results/20`` -- FFT is undetectable by key name because ``FFTLayer`` has
      no parameters, so a bare backbone was built and **zero** weights loaded.
      Reported 34.91%, i.e. chance, against an 88.37% training log.
    * ``results/18`` -- CBAM re-nests ``features``, so the ConvNeXt-Small probe
      missed and Small loaded as Tiny. Identical channel widths meant no size
      mismatch, so 162 surplus tensors were dropped without a word. Reported
      70.94% against 83.56%.

    The ``attention_head`` and ``gem_*``/``cbam_*`` arguments are retained for
    CLI compatibility but are no longer consulted -- the checkpoint's own config
    is authoritative, which is what makes forgetting ``--attention_head cbam``
    harmless instead of quietly wrong.
    """
    from modules.model_builder import load_model as _load

    model, cfg = _load(model_path, device=device)
    print(
        f"Loaded {cfg.get('backbone')} "
        f"(srm={bool(cfg.get('use_srm'))}, fft={bool(cfg.get('use_fft'))}, "
        f"attention={cfg.get('attention_head') or 'none'}, "
        f"classes={cfg.get('num_classes', 3)}) -- config-driven, strict=True"
    )
    return model



def run_evaluation(
    model_path,
    data_dir,
    batch_size=64,
    save_dir="../results",
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Attention head: {attention_head}")
    model = load_model(
        model_path,
        device,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )

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


def run_cascade_evaluation(
    stage1_model_path,
    stage2_model_path,
    data_dir,
    batch_size=64,
    save_dir="../results",
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
    cascade_threshold: float = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading Stage-1 model from: {stage1_model_path}")
    stage1 = load_model(
        stage1_model_path,
        device,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )

    print(f"Loading Stage-2 model from: {stage2_model_path}")
    stage2 = load_model(
        stage2_model_path,
        device,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )

    dataset = DeepfakeDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(dataset)} | Batches: {len(loader)}")

    cascade = CascadeClassifier(stage1, stage2, device=device, cascade_threshold=cascade_threshold)
    out = cascade.run_on_dataloader(loader)

    # ensure save dir exists
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_true.npy"), out['y_true'])
    np.save(os.path.join(save_dir, "y_pred.npy"), out['y_pred'])
    np.save(os.path.join(save_dir, "stage1_preds.npy"), out['stage1_preds'])
    np.save(os.path.join(save_dir, "stage2_preds.npy"), out['stage2_preds'])

    # write stats
    stats_path = os.path.join(save_dir, "cascade_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(out['stats'], f, indent=2)

    print(f"Saved cascade outputs to: {save_dir}")
    print(f"Cascade stats: {out['stats']}")

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
    parser.add_argument(
        '--attention_head',
        type=str,
        default=None,
        choices=['none', 'gem', 'cbam'],
        help='Attention head used during training. '
             'If omitted, will be inferred from training_summary.json when available.',
    )
    parser.add_argument('--gem_p', type=float, default=None,
                        help='GeM pooling exponent p (overrides summary if set)')
    parser.add_argument('--gem_learnable', action='store_true',
                        help='Use learnable GeM exponent p (overrides summary if set)')
    parser.add_argument('--cbam_reduction', type=int, default=None,
                        help='CBAM channel reduction ratio (overrides summary if set)')
    parser.add_argument('--cbam_kernel', type=int, default=None,
                        help='CBAM spatial attention kernel size, 3 or 7 (overrides summary if set)')
    parser.add_argument('--cascade', action='store_true', help='Enable two-stage cascade evaluation')
    parser.add_argument('--stage2_model_path', type=str, default=None, help='Path to Stage-2 refiner model (.pth)')
    parser.add_argument('--cascade_threshold', type=float, default=None, help='Optional threshold to gate Stage-2 when |P(real)-P(ai_edited)| <= threshold')

    args = parser.parse_args()

    # Auto-derive save_dir from the checkpoint path when not explicitly set.
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else _derive_save_dir(args.model_path)
    print(f"Results will be saved to: {save_dir}")

    # Try to infer attention configuration from training summary when not provided.
    summary_path = os.path.join(save_dir, "training_summary.json")
    summary_cfg = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            summary_cfg = data.get("config", {})
            print(f"Loaded attention config from {summary_path}")
        except Exception as e:
            print(f"Warning: failed to read {summary_path}: {e}")

    def _cfg_or(default, cli_value, key):
        if cli_value is not None:
            return cli_value
        return summary_cfg.get(key, default)

    attention_head = args.attention_head if args.attention_head is not None else summary_cfg.get("attention_head", "none")
    gem_p = _cfg_or(3.0, args.gem_p, "gem_p")
    # gem_learnable stored as bool; CLI flag only enables it.
    gem_learnable = summary_cfg.get("gem_learnable", False) or args.gem_learnable
    cbam_reduction = _cfg_or(16, args.cbam_reduction, "cbam_reduction")
    cbam_kernel = _cfg_or(7, args.cbam_kernel, "cbam_kernel")

    if args.cascade:
        if not args.stage2_model_path:
            raise RuntimeError("--cascade requires --stage2_model_path to be set")
        run_cascade_evaluation(
            stage1_model_path=args.model_path,
            stage2_model_path=args.stage2_model_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            save_dir=save_dir,
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
            cascade_threshold=args.cascade_threshold,
        )
    else:
        run_evaluation(
            model_path=args.model_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            save_dir=save_dir,
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
        )
