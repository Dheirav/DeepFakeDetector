"""
Standalone script to regenerate training plots from a saved metrics.csv.

Usage:
    python scripts/training/plot_metrics.py --csv results/metrics.csv --out results/
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt


def load_csv(csv_path):
    epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []
    f1_macro, f1_real, f1_ai_gen, f1_ai_edit = [], [], [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            train_accs.append(float(row['train_acc']))
            val_accs.append(float(row['val_acc']))
            f1_macro.append(float(row['val_f1_macro']))
            f1_real.append(float(row['f1_real']))
            f1_ai_gen.append(float(row['f1_ai_gen']))
            f1_ai_edit.append(float(row['f1_ai_edit']))

    return dict(
        epochs=epochs,
        train_losses=train_losses, val_losses=val_losses,
        train_accs=train_accs, val_accs=val_accs,
        f1_macro=f1_macro, f1_real=f1_real,
        f1_ai_gen=f1_ai_gen, f1_ai_edit=f1_ai_edit,
    )


def plot_loss(data, out_dir):
    plt.figure()
    plt.plot(data['epochs'], data['train_losses'], label='Train Loss')
    plt.plot(data['epochs'], data['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy(data, out_dir):
    plt.figure()
    plt.plot(data['epochs'], data['train_accs'], label='Train Acc')
    plt.plot(data['epochs'], data['val_accs'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    path = os.path.join(out_dir, 'accuracy_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_f1(data, out_dir):
    plt.figure()
    plt.plot(data['epochs'], data['f1_macro'],   label='Val F1 (macro)')
    plt.plot(data['epochs'], data['f1_real'],    label='F1 real',     linestyle='--')
    plt.plot(data['epochs'], data['f1_ai_gen'],  label='F1 ai_gen',   linestyle='--')
    plt.plot(data['epochs'], data['f1_ai_edit'], label='F1 ai_edit',  linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Curves')
    plt.legend()
    path = os.path.join(out_dir, 'f1_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV")
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to metrics.csv produced by train_full.py')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory for plots (defaults to same dir as CSV)')
    parser.add_argument('--plots', nargs='+', default=['loss', 'accuracy', 'f1'],
                        choices=['loss', 'accuracy', 'f1'],
                        help='Which plots to generate (default: all)')
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    data = load_csv(args.csv)
    print(f"Loaded {len(data['epochs'])} epochs from {args.csv}")

    if 'loss' in args.plots:
        plot_loss(data, out_dir)
    if 'accuracy' in args.plots:
        plot_accuracy(data, out_dir)
    if 'f1' in args.plots:
        plot_f1(data, out_dir)


if __name__ == '__main__':
    main()
