"""
Stage-2 sweep utility

Runs multiple Stage-2 refiner trainings with different hyperparameters
and evaluates each on the binary Real vs AI_Edited validation set.

Usage:
  python3 scripts/training/stage2_sweep.py

This mirrors the pattern used by `weight_sweep.py` for Stage-1.
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Sweep configs: tuples (run_name, config_kwargs)
# Here we sweep binary class weights and learning rates as an example.
# Baseline: model 21__convnext-small__light__0.4__cosine__focal__srm
# Architecture: ConvNeXt-Small, augmentation=light, dropout=0.4, scheduler=cosine,
# loss=weighted_focal, preprocessing=SRM
SWEEP_CONFIGS = [
    ("21_base_w1.0_lr1e-4", {"class_weights": [1.0, 1.0], "lr": 1e-4}),
    ("21_base_w1.5_lr1e-4", {"class_weights": [1.5, 1.0], "lr": 1e-4}),
    ("21_base_w2.0_lr1e-4", {"class_weights": [2.0, 1.0], "lr": 1e-4}),
    ("21_base_w1.5_lr5e-5", {"class_weights": [1.5, 1.0], "lr": 5e-5}),
]

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_stage2_refiner.py"
EVAL_SCRIPT  = ROOT / "scripts" / "evaluation" / "evaluate_binary.py"
CM_SCRIPT    = ROOT / "scripts" / "evaluation" / "plot_confusion_matrix.py"


def run_training(run_name, cfg, epochs, skip_existing):
    model_dir = ROOT / "models" / "stage2_refiner" / run_name
    if skip_existing and model_dir.exists():
        print(f"  [SKIP TRAIN] {run_name} — folder exists")
        return True

        # Use baseline settings derived from model 21 (ConvNeXt-Small, light augment, dropout 0.4, cosine, SRM)
        cmd = [sys.executable, str(TRAIN_SCRIPT),
            "--run_name", run_name,
            "--epochs", str(epochs),
            "--batch_size", "64",
            "--lr", str(cfg.get("lr", 1e-4)),
            "--backbone", "convnext_small",
            "--augment", "light",
            "--dropout_p", "0.4",
            "--use_srm",
            "--lr_schedule", "cosine",
            "--loss_type", "weighted_focal",
        ]
    # class weights (binary) — pass as two floats
    if "class_weights" in cfg:
        cmd += ["--class_weights", str(cfg["class_weights"][0]), str(cfg["class_weights"][1])]

    print(f"\nTRAIN: {run_name} cfg={cfg}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0


def run_evaluation(run_name):
    # checkpoint path depends on train script saving under models/stage2_refiner/<run_name>/best_model.pth
    ckpt = ROOT / "models" / "stage2_refiner" / run_name / "best_model.pth"
    if not ckpt.exists():
        print(f"  [SKIP EVAL] No checkpoint found for {run_name}")
        return False

    results_dir = ROOT / "results" / run_name
    if (results_dir / "y_true.npy").exists() and (results_dir / "y_pred.npy").exists():
        print(f"  [SKIP EVAL] y_true/y_pred already exist in {results_dir.name}")
        return True

    cmd = [sys.executable, str(EVAL_SCRIPT),
           "--model_path", str(ckpt),
           "--data_dir", "dataset_builder/val",
           "--save_dir", str(results_dir),
    ]
    print(f"\nEVAL: {run_name}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0


def run_confusion_matrix(run_name):
    results_dir = ROOT / "results" / run_name
    y_true_path = results_dir / "y_true.npy"
    y_pred_path = results_dir / "y_pred.npy"
    save_path   = results_dir / "confusion_matrix.png"

    if not y_true_path.exists() or not y_pred_path.exists():
        print(f"  [SKIP CM] No y_true/y_pred in {results_dir.name}")
        return

    if save_path.exists():
        print(f"  [SKIP CM] confusion_matrix.png already exists in {results_dir.name}")
        return

    cmd = [sys.executable, str(CM_SCRIPT),
           "--run_dir", str(results_dir),
           "--y_true_path", str(y_true_path),
           "--y_pred_path", str(y_pred_path),
           "--save_path", str(save_path),
    ]
    print(f"  CM: {run_name}")
    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep Stage-2 hyperparameters")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--configs', nargs='+', default=None)
    args = parser.parse_args()

    configs = SWEEP_CONFIGS
    if args.configs:
        configs = [c for c in SWEEP_CONFIGS if c[0] in args.configs]

    results = []
    for run_name, cfg in configs:
        ok = run_training(run_name, cfg, args.epochs, args.skip_existing)
        if ok:
            eval_ok = run_evaluation(run_name)
            if eval_ok:
                run_confusion_matrix(run_name)

        # try to load summary metrics if present
        summary_path = ROOT / "results" / run_name / "training_summary.json"
        best_val_acc = None
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            best_val_acc = data.get('best_val_acc')
        results.append({"run_name": run_name, "cfg": cfg, "best_val_acc": best_val_acc})

    print("\nSweep finished. Results summary:")
    for r in results:
        print(r)
