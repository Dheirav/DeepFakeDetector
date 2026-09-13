#!/usr/bin/env python3
"""What an abstain threshold costs and buys, from saved probabilities.

The frontend used to force one of three answers. The first real photo it was
shown came back "ai_edited" at 0.82 with the mask on a tablecloth. A tool that
helps someone decide whether to trust an image should be allowed to say
"cannot tell", and the question is where to put the line.

Rule: predict argmax if max prob >= tau, else abstain. For each tau this
prints coverage (fraction answered), accuracy on the answered subset, and the
per-class precision on the answered subset, on the in-distribution test split
and on the held-out generators. Precision of "real" on the answered subset is
the number a user cares about: when it says real, how often is it right. And
the real-class *recall* among answered images is how often a real photo gets
a wrong confident answer, which is the failure this is meant to catch.

Writes <run>/decision_rule.json with the chosen tau so the frontend can read
it, unless --dry-run.
"""

import argparse, glob, json, os
import numpy as np

CLASSES = ["real", "ai_generated", "ai_edited"]


def summarise(P, y, tau):
    conf, pred = P.max(1), P.argmax(1)
    ans = conf >= tau
    n = len(y); cov = ans.mean()
    if ans.sum() == 0:
        return cov, float("nan"), [float("nan")] * 3, [float("nan")] * 3
    acc = (pred[ans] == y[ans]).mean()
    prec, rec = [], []
    for c in range(3):
        pc = (pred[ans] == c); tc = (y[ans] == c)
        prec.append((pc & tc).sum() / pc.sum() if pc.sum() else float("nan"))
        rec.append((pc & tc).sum() / tc.sum() if tc.sum() else float("nan"))
    return cov, acc, prec, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/mask_head_clip448_balanced")
    ap.add_argument("--taus", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98")
    ap.add_argument("--choose", type=float, default=None,
                    help="tau to write into decision_rule.json (default: none written)")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]

    P = np.load(os.path.join(args.run, "probs.npy")); y = np.load(os.path.join(args.run, "y_true.npy"))
    print(f"  in-distribution test split, n={len(y)}")
    print(f"  {'tau':>5}{'coverage':>10}{'acc|ans':>9}   precision real/gen/edit    recall real/gen/edit")
    for t in taus:
        cov, acc, pr, rc = summarise(P, y, t)
        print(f"  {t:>5.2f}{cov:>10.3f}{acc:>9.3f}   {pr[0]:.3f} {pr[1]:.3f} {pr[2]:.3f}          {rc[0]:.3f} {rc[1]:.3f} {rc[2]:.3f}")

    # held-out generators: one file per (generator, class), pooled
    hp, hy = [], []
    for f in sorted(glob.glob(os.path.join(args.run, "heldout_probs_*.npy"))):
        cls = os.path.basename(f)[len("heldout_probs_"):-4].split("_", 1)[1]
        p = np.load(f); hp.append(p); hy.append(np.full(len(p), CLASSES.index(cls)))
    if hp:
        HP, HY = np.concatenate(hp), np.concatenate(hy)
        print(f"\n  held-out generators pooled, n={len(HY)} (real only from sd2; mostly fake)")
        print(f"  {'tau':>5}{'coverage':>10}{'acc|ans':>9}   precision real/gen/edit    recall real/gen/edit")
        for t in taus:
            cov, acc, pr, rc = summarise(HP, HY, t)
            print(f"  {t:>5.2f}{cov:>10.3f}{acc:>9.3f}   {pr[0]:.3f} {pr[1]:.3f} {pr[2]:.3f}          {rc[0]:.3f} {rc[1]:.3f} {rc[2]:.3f}")
        # the failure this is for: a real image getting a confident wrong answer
        real = HY == 0
        print(f"\n  sd2 real images (n={real.sum()}): confident-wrong rate by tau")
        for t in taus:
            conf, pred = HP[real].max(1), HP[real].argmax(1)
            cw = ((pred != 0) & (conf >= t)).mean(); ab = (conf < t).mean()
            print(f"  {t:>5.2f}   wrong and answered {cw:.3f}   abstained {ab:.3f}")

    if args.choose is not None:
        cov, acc, pr, rc = summarise(P, y, args.choose)
        rule = {"abstain_below": args.choose, "coverage_in_distribution": float(cov),
                "accuracy_when_answered": float(acc),
                "precision_when_answered": {c: float(v) for c, v in zip(CLASSES, pr)},
                "source": "scripts/evaluation/abstain_sweep.py on this run's saved probabilities"}
        json.dump(rule, open(os.path.join(args.run, "decision_rule.json"), "w"), indent=2)
        print(f"\n  wrote {args.run}/decision_rule.json  (abstain below {args.choose})")


if __name__ == "__main__":
    main()
