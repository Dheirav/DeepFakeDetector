"""
compare_runs.py — Model Comparison Dashboard
=============================================
Auto-discovers every folder under results/ that contains a metrics.csv,
then generates a multi-panel comparison plot saved to results/comparison/.

Usage:
    python scripts/compare_runs.py                    # compare all runs
    python scripts/compare_runs.py --runs sweep_w150_100_150 convnext_srm_focal
    python scripts/compare_runs.py --exclude resnet18_ce_baseline
    python scripts/compare_runs.py --best-only        # highlight best, shade rest
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

matplotlib.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    }
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Real", "AI-Generated", "AI-Edited"]
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "comparison"

# Runs that are purely reference / not architecture experiments — dim them
BASELINE_RUNS = {
    "resnet18_ce_baseline",
    "resnet18_srm_focal_5ep",
    "run_20260307_063053",  # old tensorboard run
}

# Friendly display names (auto-fallback to folder name)
DISPLAY_NAMES = {
    "resnet18_srm_focal_wd": "ResNet-18 SRM+Focal",
    "sweep_w150_100_150": "ResNet-18 [1.5,1.0,1.5] (best sweep)",
    "resnet18_dropout_cosine": "ResNet-18 dropout=0.4 cosine",
    "resnet18_dropout05_plateau": "ResNet-18 dropout=0.5 plateau",
    "convnext_srm_focal": "ConvNeXt-Tiny ",
}

# ── Data loading ───────────────────────────────────────────────────────────────


def discover_runs(results_dir: Path) -> list[Path]:
    """Return all result folders that contain a metrics.csv."""
    return sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and (p / "metrics.csv").exists()]
    )


def load_run(run_dir: Path) -> dict | None:
    """Load all available data for a single run. Returns None if metrics.csv missing."""
    if not (run_dir / "metrics.csv").exists():
        return None

    data = {"name": run_dir.name, "dir": run_dir}

    # Training curves
    try:
        data["metrics"] = pd.read_csv(run_dir / "metrics.csv")
    except Exception:
        return None

    # Training summary
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data["summary"] = json.load(f)
    else:
        data["summary"] = {}

    # Test predictions
    y_true_path = run_dir / "y_true.npy"
    y_pred_path = run_dir / "y_pred.npy"
    if y_true_path.exists() and y_pred_path.exists():
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        data["y_true"] = y_true
        data["y_pred"] = y_pred
        data["test_acc"] = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
        data["f1_per_class"] = [report[c]["f1-score"] for c in CLASS_NAMES]
        data["f1_macro"] = report["macro avg"]["f1-score"]
        data["cm"] = confusion_matrix(y_true, y_pred)
    else:
        data["test_acc"] = None
        data["f1_per_class"] = None
        data["f1_macro"] = None
        data["cm"] = None

    return data


# ── Colour palette (auto-assign, stable across runs) ──────────────────────────


def build_palette(run_names: list[str]) -> dict[str, str]:
    colours = [
        "#2196F3",  # blue
        "#FF9800",  # orange
        "#4CAF50",  # green
        "#F44336",  # red
        "#9C27B0",  # purple
        "#00BCD4",  # cyan
        "#FF5722",  # deep orange
        "#607D8B",  # grey-blue
        "#8BC34A",  # light green
        "#795548",  # brown

        "#3F51B5",  # indigo
        "#E91E63",  # pink
        "#009688",  # teal
        "#CDDC39",  # lime
        "#FFC107",  # amber
        "#673AB7",  # deep purple
        "#03A9F4",  # light blue
        "#A1887F",  # light brown
        "#AED581",  # pastel green
        "#FF7043",  # coral

        "#1E88E5",  # strong blue
        "#D81B60",  # strong pink
        "#43A047",  # strong green
        "#FB8C00",  # strong orange
        "#8E24AA",  # strong purple
        "#26C6DA",  # bright cyan
        "#7CB342",  # olive green
        "#546E7A",  # slate
        "#EF5350",  # soft red
        "#AB47BC",  # lavender purple
    ]

    return {name: colours[i % len(colours)] for i, name in enumerate(run_names)}

# ── Individual plot panels ─────────────────────────────────────────────────────


def plot_val_accuracy_curves(ax, runs, palette):
    """Line chart: validation accuracy over epochs for every run."""
    for r in runs:
        m = r["metrics"]
        if "val_acc" not in m.columns:
            continue
        name = DISPLAY_NAMES.get(r["name"], r["name"])
        alpha = 0.35 if r["name"] in BASELINE_RUNS else 1.0
        lw = 2.5 if r["name"] not in BASELINE_RUNS else 1.2
        ax.plot(
            m["epoch"],
            m["val_acc"] * 100,
            label=name,
            color=palette[r["name"]],
            alpha=alpha,
            linewidth=lw,
        )
    ax.set_title("Validation Accuracy per Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=7, loc="lower right")


def plot_train_val_curves(ax, runs, palette):
    """Line chart: train vs val accuracy for all runs (dashed = train)."""
    for r in runs:
        m = r["metrics"]
        if "train_acc" not in m.columns or "val_acc" not in m.columns:
            continue
        name = DISPLAY_NAMES.get(r["name"], r["name"])
        alpha = 0.35 if r["name"] in BASELINE_RUNS else 1.0
        c = palette[r["name"]]
        ax.plot(m["epoch"], m["train_acc"] * 100, linestyle="--", color=c, alpha=alpha * 0.6, linewidth=1)
        ax.plot(m["epoch"], m["val_acc"] * 100, color=c, alpha=alpha, linewidth=2, label=name)
    ax.set_title("Train (dashed) vs Val Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=7, loc="lower right")


def plot_test_accuracy_bar(ax, runs, palette):
    """Bar chart: test accuracy per model (sorted descending)."""
    evaluated = [r for r in runs if r["test_acc"] is not None]
    evaluated_sorted = sorted(evaluated, key=lambda r: r["test_acc"])
    names = [DISPLAY_NAMES.get(r["name"], r["name"]) for r in evaluated_sorted]
    accs = [r["test_acc"] * 100 for r in evaluated_sorted]
    colours = [palette[r["name"]] for r in evaluated_sorted]
    alphas = [0.45 if r["name"] in BASELINE_RUNS else 1.0 for r in evaluated_sorted]

    bars = ax.barh(names, accs, color=colours, alpha=0.9)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            va="center",
            fontsize=8,
        )

    ax.set_title("Test Accuracy", fontweight="bold")
    ax.set_xlabel("Accuracy (%)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    # x-axis starts near the minimum for readability
    if accs:
        ax.set_xlim(max(0, min(accs) - 3), min(100, max(accs) + 3))


def plot_f1_per_class_bar(ax, runs, palette):
    """Grouped bar chart: F1 per class for each model."""
    evaluated = [r for r in runs if r["f1_per_class"] is not None]
    if not evaluated:
        ax.set_visible(False)
        return

    n_runs = len(evaluated)
    n_classes = len(CLASS_NAMES)
    x = np.arange(n_classes)
    width = 0.8 / n_runs

    for i, r in enumerate(evaluated):
        name = DISPLAY_NAMES.get(r["name"], r["name"])
        alpha = 0.45 if r["name"] in BASELINE_RUNS else 1.0
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            r["f1_per_class"],
            width,
            label=name,
            color=palette[r["name"]],
            alpha=alpha,
        )

    ax.set_title("Per-Class F1 Score", fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")


def plot_overfitting_gap(ax, runs, palette):
    """Bar chart: final train acc − final val acc (overfitting gap)."""
    evaluated = [
        r
        for r in runs
        if "train_acc" in r["metrics"].columns and "val_acc" in r["metrics"].columns
    ]
    evaluated_sorted = sorted(
        evaluated,
        key=lambda r: (r["metrics"]["train_acc"].iloc[-1] - r["metrics"]["val_acc"].iloc[-1]),
        reverse=True,
    )
    names = [DISPLAY_NAMES.get(r["name"], r["name"]) for r in evaluated_sorted]
    gaps = [
        (r["metrics"]["train_acc"].iloc[-1] - r["metrics"]["val_acc"].iloc[-1]) * 100
        for r in evaluated_sorted
    ]
    colours = [palette[r["name"]] for r in evaluated_sorted]
    alphas = [0.45 if r["name"] in BASELINE_RUNS else 1.0 for r in evaluated_sorted]

    bars = ax.barh(names, gaps, color=colours)
    for bar, a, gap in zip(bars, alphas, gaps):
        bar.set_alpha(a)
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{gap:.1f}pt",
            va="center",
            fontsize=8,
        )

    ax.set_title("Train−Val Gap (final epoch)", fontweight="bold")
    ax.set_xlabel("Gap (percentage points)")
    ax.axvline(0, color="black", linewidth=0.8)


def plot_confusion_matrix(ax, run, title_suffix=""):
    """Heatmap confusion matrix for one run."""
    if run["cm"] is None:
        ax.set_visible(False)
        return
    cm = run["cm"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)

    name = DISPLAY_NAMES.get(run["name"], run["name"])
    ax.set_title(f"Confusion Matrix\n{name}{title_suffix}", fontweight="bold", fontsize=9)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            colour = "white" if cm_pct[i, j] > 55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=7.5,
                color=colour,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Row %", fontsize=7)


# ── Summary table ──────────────────────────────────────────────────────────────


def print_summary_table(runs):
    rows = []
    for r in runs:
        summary = r.get("summary", {})
        m = r["metrics"]
        best_val = summary.get("best_val_acc", m["val_acc"].max())
        epochs = summary.get("epochs_trained", len(m))
        config = summary.get("config", {})
        backbone = config.get("backbone", "resnet18")
        dropout = config.get("dropout_p", 0.0)
        lr_sched = config.get("lr_schedule", "—")
        test_acc = f"{r['test_acc']*100:.2f}%" if r["test_acc"] else "—"
        f1_macro = f"{r['f1_macro']:.4f}" if r["f1_macro"] else "—"
        rows.append(
            {
                "Run": DISPLAY_NAMES.get(r["name"], r["name"]),
                "Backbone": backbone,
                "Dropout": dropout,
                "LR Schedule": lr_sched,
                "Epochs": epochs,
                "Best Val": f"{best_val*100:.2f}%",
                "Test Acc": test_acc,
                "F1 Macro": f1_macro,
            }
        )

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("TRAINING RUN COMPARISON")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compare training runs visually.")
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run names to include (default: all)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Run names to skip",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Only show non-baseline runs (filters BASELINE_RUNS set)",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Named subfolder under the output dir, e.g. --name after_convnext",
    )
    args = parser.parse_args()

    output_dir = Path(args.out) / args.name if args.name else Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover or filter runs
    if args.runs:
        run_dirs = [RESULTS_DIR / name for name in args.runs]
    else:
        run_dirs = discover_runs(RESULTS_DIR)

    # Apply exclusions
    excluded = set(args.exclude)
    if args.best_only:
        excluded |= BASELINE_RUNS
    run_dirs = [d for d in run_dirs if d.name not in excluded]

    runs = []
    for d in run_dirs:
        r = load_run(d)
        if r:
            runs.append(r)
        else:
            print(f"  [skip] {d.name} — no metrics.csv")

    if not runs:
        print("No runs found. Check that results/ contains folders with metrics.csv.")
        sys.exit(1)

    print(f"\nLoaded {len(runs)} run(s): {', '.join(r['name'] for r in runs)}")
    palette = build_palette([r["name"] for r in runs])

    # ── Figure 1: Training Overview ───────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig1.suptitle("Model Training Comparison", fontsize=14, fontweight="bold", y=1.01)

    plot_val_accuracy_curves(axes[0, 0], runs, palette)
    plot_train_val_curves(axes[0, 1], runs, palette)
    plot_test_accuracy_bar(axes[1, 0], runs, palette)
    plot_overfitting_gap(axes[1, 1], runs, palette)

    fig1.tight_layout()
    fig1_path = output_dir / "training_overview.png"
    fig1.savefig(fig1_path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {fig1_path}")

    # ── Figure 2: Performance Detail ──────────────────────────────────────────
    evaluated = [r for r in runs if r["f1_per_class"] is not None]
    n_best = min(2, len(evaluated))
    best_runs = sorted(evaluated, key=lambda r: r.get("test_acc") or 0, reverse=True)[:n_best]

    fig2_cols = 1 + n_best
    fig2, axes2 = plt.subplots(1, fig2_cols, figsize=(7 * fig2_cols, 6))
    if fig2_cols == 1:
        axes2 = [axes2]
    fig2.suptitle("Performance Detail", fontsize=14, fontweight="bold")

    plot_f1_per_class_bar(axes2[0], runs, palette)
    for i, r in enumerate(best_runs):
        label = " (best)" if i == 0 else ""
        plot_confusion_matrix(axes2[1 + i], r, title_suffix=label)

    fig2.tight_layout()
    fig2_path = output_dir / "performance_detail.png"
    fig2.savefig(fig2_path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {fig2_path}")

    # ── Figure 3: Loss curves ──────────────────────────────────────────────────
    runs_with_loss = [r for r in runs if "train_loss" in r["metrics"].columns]
    if runs_with_loss:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for r in runs_with_loss:
            m = r["metrics"]
            name = DISPLAY_NAMES.get(r["name"], r["name"])
            alpha = 0.35 if r["name"] in BASELINE_RUNS else 1.0
            ax3.plot(m["epoch"], m["train_loss"], linestyle="--", color=palette[r["name"]], alpha=alpha * 0.5, linewidth=1)
            ax3.plot(m["epoch"], m["val_loss"], color=palette[r["name"]], alpha=alpha, linewidth=2, label=name)
        ax3.set_title("Train (dashed) vs Val Loss", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend(fontsize=7)
        fig3.tight_layout()
        fig3_path = output_dir / "loss_curves.png"
        fig3.savefig(fig3_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fig3_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print_summary_table(runs)

    plt.close("all")
    print(f"All plots saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
