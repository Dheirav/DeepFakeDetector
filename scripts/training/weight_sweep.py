"""
Weight Sweep — find the best class weights for WeightedFocalLoss.

Runs a short training (configurable epochs) for each weight combination,
evaluates on the test set, plots a confusion matrix, then prints a ranked
comparison table at the end.

Target: reduce Real ↔ AI-Edited confusion (the main error source).
Current best: [1.5, 1.0, 1.5] — 82.94% test acc, F1 Real 0.765

Usage
-----
# Full sweep (all configs, 10 epochs each):
python3 scripts/training/weight_sweep.py

# Fewer epochs for a quick comparison:
python3 scripts/training/weight_sweep.py --epochs 6

# Skip configs whose run folder already exists (resume interrupted sweep):
python3 scripts/training/weight_sweep.py --skip_existing

# Run only specific configs:
python3 scripts/training/weight_sweep.py --configs sweep_w200_100_150 sweep_w250_100_150
"""

import os
import sys
import json
import argparse
import subprocess
import csv
from pathlib import Path

# ── Sweep configs ─────────────────────────────────────────────────────────────
# Each entry: (run_name, [w_real, w_ai_gen, w_ai_edit])
# Rationale:
#   AI Generated already achieves F1~0.92 — it needs no extra emphasis.
#   Real ↔ AI-Edited is the hard boundary; bump those weights.
SWEEP_CONFIGS = [
    ("sweep_w150_100_150", [1.5, 1.0, 1.5]),   # current best — reference point
    ("sweep_w200_100_150", [2.0, 1.0, 1.5]),   # boost Real
    ("sweep_w250_100_150", [2.5, 1.0, 1.5]),   # boost Real more
    ("sweep_w200_100_200", [2.0, 1.0, 2.0]),   # boost both boundary classes equally
    ("sweep_w200_080_150", [2.0, 0.8, 1.5]),   # boost Real, reduce AI-Gen
    ("sweep_w200_080_200", [2.0, 0.8, 2.0]),   # boost Real+AI-Edit, reduce AI-Gen
    ("sweep_w300_100_150", [3.0, 1.0, 1.5]),   # aggressive Real boost
    ("sweep_w150_100_200", [1.5, 1.0, 2.0]),   # boost AI-Edited only
]

_ROOT        = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _ROOT / "scripts" / "training" / "train_full.py"
_EVAL_SCRIPT  = _ROOT / "scripts" / "evaluation" / "evaluate.py"
_CM_SCRIPT    = _ROOT / "scripts" / "evaluation" / "plot_confusion_matrix.py"


def run_training(run_name, weights, epochs, skip_existing):
    model_dir = _ROOT / "models" / run_name
    if skip_existing and model_dir.exists():
        print(f"  [SKIP TRAIN] {run_name} — folder already exists")
        return True

    cmd = [
        sys.executable, str(_TRAIN_SCRIPT),
        "--use_srm",
        "--loss", "weighted_focal",
        "--label_smoothing", "0.1",
        "--weight_decay", "1e-4",
        "--epochs", str(epochs),
        "--early_stop_patience", "3",
        "--batch_size", "64",
        "--run_name", run_name,
        "--class_weights", *[str(w) for w in weights],
    ]
    print(f"\n{'='*60}")
    print(f"  TRAIN: {run_name}  weights={weights}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return result.returncode == 0


def run_evaluation(run_name):
    """Run evaluate.py — saves y_true.npy and y_pred.npy to results/<run_name>/"""
    ckpt = _ROOT / "models" / run_name / "best_model.pth"
    if not ckpt.exists():
        # fallback for older runs that used the old name
        ckpt = _ROOT / "models" / run_name / "best_resnet18.pth"
    if not ckpt.exists():
        print(f"  [SKIP EVAL] No checkpoint found: {ckpt}")
        return False

    results_dir = _ROOT / "results" / run_name
    if (results_dir / "y_true.npy").exists() and (results_dir / "y_pred.npy").exists():
        print(f"  [SKIP EVAL] y_true/y_pred already exist in {results_dir.name}")
        return True

    cmd = [
        sys.executable, str(_EVAL_SCRIPT),
        "--model_path", str(ckpt),
        "--data_dir", "dataset_builder/test",
        "--save_dir", str(results_dir),
    ]
    print(f"\n  EVAL: {run_name}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return result.returncode == 0


def run_confusion_matrix(run_name):
    """Run plot_confusion_matrix.py — reads npy files, saves confusion_matrix.png"""
    results_dir = _ROOT / "results" / run_name
    y_true_path = results_dir / "y_true.npy"
    y_pred_path = results_dir / "y_pred.npy"
    save_path   = results_dir / "confusion_matrix.png"

    if not y_true_path.exists() or not y_pred_path.exists():
        print(f"  [SKIP CM] No y_true/y_pred in {results_dir.name}")
        return

    if save_path.exists():
        print(f"  [SKIP CM] confusion_matrix.png already exists in {results_dir.name}")
        return

    cmd = [
        sys.executable, str(_CM_SCRIPT),
        "--run_dir",     str(results_dir),
        "--y_true_path", str(y_true_path),
        "--y_pred_path", str(y_pred_path),
        "--save_path",   str(save_path),
    ]
    print(f"  CM: {run_name}")
    subprocess.run(cmd, cwd=str(_ROOT))


def load_summary(run_name):
    """Return (best_val_acc, best_val_f1, f1_real, f1_ai_gen, f1_ai_edit, weights)."""
    summary_path = _ROOT / "results" / run_name / "training_summary.json"
    metrics_path = _ROOT / "results" / run_name / "metrics.csv"

    best_val_acc = best_val_f1 = f1_real = f1_ai_gen = f1_ai_edit = None
    weights = None

    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        best_val_acc = data.get("best_val_acc")
        weights = data.get("config", {}).get("class_weights")

    if metrics_path.exists():
        rows = list(csv.DictReader(metrics_path.open()))
        if rows:
            best_row = max(rows, key=lambda r: float(r["val_acc"]))
            best_val_f1  = float(best_row["val_f1_macro"])
            f1_real      = float(best_row["f1_real"])
            f1_ai_gen    = float(best_row["f1_ai_gen"])
            f1_ai_edit   = float(best_row["f1_ai_edit"])

    return best_val_acc, best_val_f1, f1_real, f1_ai_gen, f1_ai_edit, weights


def print_table(results):
    results.sort(key=lambda r: r["best_val_acc"] or 0, reverse=True)

    col_w = [28, 18, 10, 10, 10, 10, 10]
    headers = ["run_name", "weights", "val_acc", "f1_macro", "f1_real", "f1_aigen", "f1_aiedit"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f"{{:{w}}}" for w in col_w) + "|"

    print("\n" + "=" * 60)
    print("WEIGHT SWEEP RESULTS (ranked by val_acc)")
    print("=" * 60)
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for rank, r in enumerate(results, 1):
        w = r["weights"]
        w_str = f"[{w[0]},{w[1]},{w[2]}]" if w else "?"
        prefix = "* " if rank == 1 else "  "
        print(fmt.format(
            (prefix + r["run_name"])[:28],
            w_str[:18],
            f"{r['best_val_acc']:.4f}" if r["best_val_acc"] else "—",
            f"{r['f1_macro']:.4f}"    if r["f1_macro"]    else "—",
            f"{r['f1_real']:.4f}"     if r["f1_real"]     else "—",
            f"{r['f1_ai_gen']:.4f}"   if r["f1_ai_gen"]   else "—",
            f"{r['f1_ai_edit']:.4f}"  if r["f1_ai_edit"]  else "—",
        ))
    print(sep)

    if results and results[0]["best_val_acc"]:
        best = results[0]
        print(f"\n* Best config : {best['run_name']}")
        print(f"  Weights     : {best['weights']}")
        print(f"  Val acc     : {best['best_val_acc']:.4f}")
        print(f"  F1 Real     : {best['f1_real']:.4f}")
        print(f"  Confusion matrices saved to results/<run_name>/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep class weights for WeightedFocalLoss")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs per config (default: 10)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip training/eval steps whose outputs already exist")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Run only specific config names")
    args = parser.parse_args()

    configs = SWEEP_CONFIGS
    if args.configs:
        configs = [(n, w) for n, w in SWEEP_CONFIGS if n in args.configs]
        if not configs:
            print("No matching configs found. Available:")
            for n, w in SWEEP_CONFIGS:
                print(f"  {n}  {w}")
            sys.exit(1)

    print(f"\nWeight sweep: {len(configs)} configs × {args.epochs} epochs each")
    print("Pipeline per config: train → evaluate → confusion matrix\n")
    print("Configs:")
    for name, w in configs:
        print(f"  {name:30s}  {w}")

    results = []
    for run_name, weights in configs:
        train_ok = run_training(run_name, weights, args.epochs, args.skip_existing)
        if train_ok:
            eval_ok = run_evaluation(run_name)
            if eval_ok:
                run_confusion_matrix(run_name)
        best_val_acc, best_val_f1, f1_real, f1_ai_gen, f1_ai_edit, saved_weights = load_summary(run_name)
        results.append({
            "run_name": run_name,
            "weights": saved_weights or weights,
            "best_val_acc": best_val_acc,
            "f1_macro": best_val_f1,
            "f1_real": f1_real,
            "f1_ai_gen": f1_ai_gen,
            "f1_ai_edit": f1_ai_edit,
        })

    print_table(results)

