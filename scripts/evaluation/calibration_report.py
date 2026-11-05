"""Calibration, confusion matrix and bootstrap intervals from a run's saved arrays.

Reads probs.npy / y_true.npy / y_pred.npy (and heldout_probs_*.npy when present)
from a mask-head run directory. No model is loaded and no GPU is used, so it is
safe to run while something else owns the machine.

What it does NOT do: fit a temperature or any recalibration. Every number is
measured on the same 4,050-image test split the best epoch was selected on
(train_mask_head.py keeps the epoch with the best accuracy on that split), so
accuracy and ECE here are mildly optimistic. Held-out generator cells were never
used for selection and are the cleaner numbers.
"""
import argparse
import glob
import json
import os

import numpy as np

CLASSES = ["real", "ai_generated", "ai_edited"]
N_BINS = 15


def ece_and_bins(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    n = np.bincount(idx, minlength=n_bins)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    sa = np.bincount(idx, weights=correct, minlength=n_bins)
    ok = n > 0
    mc, ma = np.zeros(n_bins), np.zeros(n_bins)
    mc[ok], ma[ok] = sc[ok] / n[ok], sa[ok] / n[ok]
    return float((n / n.sum() * np.abs(ma - mc)).sum()), n, mc, ma


def f1_per_class(t, y):
    out = []
    for c in range(3):
        tp = np.sum((t == c) & (y == c))
        p = tp / max(np.sum(y == c), 1)
        r = tp / max(np.sum(t == c), 1)
        out.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return np.array(out)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#2a78d6", "#143e73"])
    rn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rn[i, j] * 100:.1f}%\n{cm[i, j]}", ha="center", va="center",
                    color="white" if rn[i, j] > 0.5 else ink, fontsize=10)
    ax.set_xticks(range(3), CLASSES, color=ink2)
    ax.set_yticks(range(3), CLASSES, color=ink2)
    ax.set_xlabel("predicted", color=ink2)
    ax.set_ylabel("true", color=ink2)
    ax.set_title("Confusion matrix, balanced test split (rows sum to 100%)", color=ink, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def plot_reliability(n, mc, ma, ece, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, line = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 5.6), facecolor=surface,
                                 gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ok = n > 0
    a1.plot([0, 1], [0, 1], color=ink2, lw=1, ls="--", label="perfectly calibrated")
    a1.plot(mc[ok], ma[ok], color=line, lw=2, marker="o", ms=8,
            markeredgecolor=surface, markeredgewidth=2, label="model (top-label)")
    a1.set_ylabel("accuracy in bin", color=ink2)
    a1.set_ylim(0, 1.02)
    a1.set_title(f"Reliability, balanced test split (ECE {ece:.3f}, {N_BINS} bins)", color=ink, fontsize=10)
    a1.legend(frameon=False, labelcolor=ink2, loc="upper left", fontsize=9)
    a2.bar(mc[ok], n[ok], width=0.04, color=line)
    a2.set_ylabel("images", color=ink2)
    a2.set_xlabel("mean top-class probability in bin", color=ink2)
    for a in (a1, a2):
        a.set_facecolor(surface)
        a.grid(axis="y", color=grid, lw=0.8)
        a.tick_params(colors=ink2)
        for k in ("top", "right", "left", "bottom"):
            a.spines[k].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="results/mask_head_clip448_balanced")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--abstain", type=float, default=0.90)
    a = ap.parse_args()

    p = np.load(os.path.join(a.run, "probs.npy")).astype(np.float64)
    t = np.load(os.path.join(a.run, "y_true.npy"))
    y = np.load(os.path.join(a.run, "y_pred.npy"))
    assert (p.argmax(1) == y).all(), "probs.npy and y_pred.npy disagree"
    n = len(t)
    conf, correct = p.max(1), (y == t).astype(float)

    cm = np.zeros((3, 3), int)
    np.add.at(cm, (t, y), 1)
    ece, nb, mc, ma = ece_and_bins(conf, correct)
    nll = float(-np.log(np.clip(p[np.arange(n), t], 1e-12, None)).mean())
    brier = float(((p - np.eye(3)[t]) ** 2).sum(1).mean())

    rng = np.random.default_rng(a.seed)
    acc_b, f1_b, ece_b = [], [], []
    for _ in range(a.boot):
        i = rng.integers(0, n, n)
        acc_b.append(correct[i].mean())
        f1_b.append(f1_per_class(t[i], y[i]))
        ece_b.append(ece_and_bins(conf[i], correct[i])[0])
    f1_b = np.array(f1_b)
    f1 = f1_per_class(t, y)

    ans = conf >= a.abstain
    out = {
        "run": a.run, "n_test": n, "classes": CLASSES,
        "caveat": "same split the best epoch was selected on; no recalibration fitted",
        "bootstrap": {"resamples": a.boot, "seed": a.seed},
        "accuracy": {"value": float(correct.mean()), "ci95": ci(acc_b)},
        "f1": {c: {"value": float(f1[k]), "ci95": ci(f1_b[:, k])} for k, c in enumerate(CLASSES)},
        "confusion_counts_true_by_pred": cm.tolist(),
        "ece": {"value": ece, "bins": N_BINS, "ci95": ci(ece_b)},
        "nll": nll, "brier_multiclass": brier,
        "mean_confidence": float(conf.mean()),
        "reliability_bins": [{"n": int(nb[k]), "mean_conf": float(mc[k]), "acc": float(ma[k])}
                             for k in range(N_BINS) if nb[k]],
        "abstain": {"threshold": a.abstain, "coverage": float(ans.mean()),
                    "accuracy_when_answered": float(correct[ans].mean())},
    }

    cells = {}
    for f in sorted(glob.glob(os.path.join(a.run, "heldout_probs_*.npy"))):
        stem = os.path.basename(f)[len("heldout_probs_"):-4]
        for c in CLASSES:
            if stem.endswith("_" + c):
                gen, cls = stem[: -len(c) - 1], c
        hp = np.load(f)
        hits = (hp.argmax(1) == CLASSES.index(cls)).astype(float)
        lo, hi = wilson(int(hits.sum()), len(hits))
        cells[f"{gen}/{cls}"] = {"n": len(hits), "recall": float(hits.mean()),
                                 "wilson95": [lo, hi], "_hits": hits}
    if cells:
        means = []
        for _ in range(a.boot):
            means.append(np.mean([c["_hits"][rng.integers(0, len(c["_hits"]), len(c["_hits"]))].mean()
                                  for c in cells.values()]))
        out["heldout"] = {"mean_recall": float(np.mean([c["recall"] for c in cells.values()])),
                          "mean_recall_ci95": ci(means),
                          "cells": {k: {kk: vv for kk, vv in v.items() if kk != "_hits"}
                                    for k, v in cells.items()}}
        ref = os.path.join(a.run, "heldout_generators.json")
        if os.path.exists(ref):
            r = json.load(open(ref))
            diffs = [abs(cells[f"{g}/{c}"]["recall"] - v["recall"])
                     for g, d in r.items() for c, v in d.items() if f"{g}/{c}" in cells]
            out["heldout"]["max_abs_diff_vs_heldout_generators_json"] = float(max(diffs))

    with open(os.path.join(a.run, "calibration.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    plot_confusion(cm, os.path.join(a.run, "calibration_confusion.png"))
    plot_reliability(nb, mc, ma, ece, os.path.join(a.run, "calibration_reliability.png"))

    print(f"n={n}  acc={out['accuracy']['value']:.4f} CI {out['accuracy']['ci95'][0]:.4f}-{out['accuracy']['ci95'][1]:.4f}")
    for c in CLASSES:
        print(f"  F1 {c:13s} {out['f1'][c]['value']:.3f}  CI {out['f1'][c]['ci95'][0]:.3f}-{out['f1'][c]['ci95'][1]:.3f}")
    print(f"ECE={ece:.4f} CI {out['ece']['ci95'][0]:.4f}-{out['ece']['ci95'][1]:.4f}  NLL={nll:.4f}  Brier={brier:.4f}  mean conf={conf.mean():.4f}")
    print(f"abstain@{a.abstain}: coverage {out['abstain']['coverage']:.3f}, accuracy when answered {out['abstain']['accuracy_when_answered']:.3f}")
    if cells:
        h = out["heldout"]
        print(f"held-out mean recall {h['mean_recall']:.4f} CI {h['mean_recall_ci95'][0]:.4f}-{h['mean_recall_ci95'][1]:.4f}  "
              f"(max diff vs heldout_generators.json {h.get('max_abs_diff_vs_heldout_generators_json', float('nan')):.2e})")
        for k, v in h["cells"].items():
            print(f"  {k:22s} n={v['n']} recall {v['recall']:.3f}  Wilson {v['wilson95'][0]:.3f}-{v['wilson95'][1]:.3f}")


if __name__ == "__main__":
    main()
