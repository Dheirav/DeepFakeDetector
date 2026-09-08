#!/usr/bin/env python3
"""Upload the trained checkpoints to a Hugging Face model repo.

Uploads `best_model.pth` for every run plus its `training_summary.json`. The
summary is not optional -- `scripts/modules/model_builder.py` reads it to
reconstruct the architecture, and a checkpoint without it cannot be loaded at
all. That is the failure this project already had once, when three separate
loaders guessed the architecture from state-dict key names and two of them
guessed wrong.

Skips the per-epoch checkpoints. Each run directory keeps both
`best_model.pth` and a last-epoch copy; the epoch copies are 5.15 GB of the
8.2 GB total and nothing loads them.

Defaults to a PRIVATE repo. Pass --public deliberately.

    huggingface-cli login          # once
    venv-linux/bin/python tools/upload_to_hf.py --repo <user>/<name> --public
"""

import argparse
import json
import os
import sys

RUNS_DIR = "models"
RESULTS_DIR = "results"


def collect():
    """Return [(run, checkpoint_path, summary_path_or_None, bytes)] sorted by run."""
    out = []
    for run in sorted(os.listdir(RUNS_DIR)):
        ckpt = os.path.join(RUNS_DIR, run, "best_model.pth")
        if not os.path.isfile(ckpt):
            continue
        summary = os.path.join(RESULTS_DIR, run, "training_summary.json")
        out.append((run, ckpt, summary if os.path.isfile(summary) else None,
                    os.path.getsize(ckpt)))
    return out


def build_card(items, repo):
    rows, loadable, orphan = [], 0, []
    for run, _ckpt, summary, size in items:
        if summary:
            cfg = json.load(open(summary))
            c, val = cfg.get("config", {}), cfg.get("best_val_acc")
            feats = []
            if c.get("use_srm"):
                feats.append("SRM")
            if c.get("use_fft"):
                feats.append("FFT")
            if (c.get("attention_head") or "none") != "none":
                feats.append(c["attention_head"].upper())
            rows.append((run, c.get("backbone", "?"), c.get("augment") or "none",
                         ", ".join(feats) or "-", f"{val:.4f}" if val else "?",
                         f"{size/1048576:.0f} MB"))
            loadable += 1
        else:
            orphan.append(run)

    body = [
        "---", "license: mit", "tags:", "  - image-classification",
        "  - deepfake-detection", "  - ai-generated-image-detection", "---", "",
        "# Deepfake Detection — checkpoint archive", "",
        "> **These models do not detect AI-generated content. They recognise which",
        "> source dataset an image came from.** They are published as the evidence",
        "> behind a documented failure analysis, not as working detectors. Read the",
        "> limitations below before using any of them for anything.", "",
        "Code, audit and full measurements:",
        "https://github.com/Dheirav/DeepFakeDetector", "",
        "## What went wrong", "",
        "A 3-class classifier (real / AI-generated / AI-edited) trained on 77,865",
        "images from 20 source corpora. It reaches ~89% on its own held-out test",
        "split. That number does not measure generalisation:", "",
        "- Every source corpus maps to exactly one class, so corpus identity is a",
        "  perfect substitute for the label. A lookup table on **file format and",
        "  resolution alone — reading no pixels — reaches 87.4%** on the same task.",
        "- Downscaling an AI-generated image to 256px and re-saving it as JPEG q80",
        "  flips the prediction to `real` for **280 of 300** images at 87% mean",
        "  confidence. Accuracy goes 0.993 → 0.027.",
        "- Validation accuracy is **inversely** correlated with robustness across",
        "  runs (Pearson r = −0.956), so the selection procedure picked the most",
        "  corpus-dependent model available.",
        "- 4.34% of the test set is byte-identical to a training image; 7.09% is a",
        "  perceptual-hash duplicate.", "",
        "## Which checkpoint to use", "",
        "**`17__convnext-small__strong__0.4__cosine__focal__srm-gem`** if you want the",
        "one that survives ordinary image handling. It scores 5 points *lower* on",
        "validation than run 19 and holds **0.955** where run 19 collapses to",
        "**0.000** on downscaled, re-encoded images.", "",
        "**`19__convnext-small__light__0.4__cosine__focal__none`** is the model that",
        "produced the headline 89% and the collapse. It is here to reproduce the",
        "finding, not to be deployed.", "",
        "## Loading", "",
        "Architecture is reconstructed from each run's `training_summary.json`.",
        "Do not infer it from key names — that is how two published numbers in this",
        "project turned out to describe models that were never trained.", "",
        "```python",
        "import sys; sys.path.insert(0, 'scripts')",
        "from modules.model_builder import load_model",
        "model, cfg = load_model('best_model.pth')   # strict=True by default",
        "```", "",
        "## Checkpoints", "",
        "| run | backbone | augment | features | best val acc | size |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        body.append("| `{}` | {} | {} | {} | {} | {} |".format(*r))
    body += ["",
             "`best val acc` is a **maximum over epochs on the model-selection split** —",
             "a biased estimator, and not a test result. Test accuracies are in the",
             "repository's `results/` directory.", ""]
    if orphan:
        body += ["## Not uploaded", "",
                 "No `training_summary.json` exists for these, so their architecture",
                 "cannot be reconstructed and the checkpoint is unloadable:", ""]
        body += [f"- `{o}`" for o in orphan] + [""]
    body += ["## Known-bad artifacts", "",
             "`results/18` and `results/20` in the source repository contain corrupt",
             "predictions (70.94% and 34.91% against training logs of 83.56% and",
             "88.37%) because an earlier evaluator rebuilt the wrong architecture and",
             "hid the mismatch behind `strict=False`. The checkpoints themselves are",
             "fine; the saved predictions are not.", ""]
    return "\n".join(body), loadable, orphan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. Dheirav/deepfake-detector-checkpoints")
    ap.add_argument("--public", action="store_true", help="create a PUBLIC repo (default private)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = collect()
    if not items:
        sys.exit("no best_model.pth found under models/")
    total = sum(s for *_, s in items)
    card, loadable, orphan = build_card(items, args.repo)

    print(f"  checkpoints : {len(items)}  ({total/1073741824:.2f} GB)")
    print(f"  loadable    : {loadable}   (have training_summary.json)")
    if orphan:
        print(f"  unloadable  : {len(orphan)} -> {', '.join(orphan)}")
    print(f"  repo        : {args.repo}  [{'PUBLIC' if args.public else 'private'}]")

    if args.dry_run:
        open("MODEL_CARD_PREVIEW.md", "w").write(card)
        print("\n  [dry run] card written to MODEL_CARD_PREVIEW.md, nothing uploaded")
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    for run, ckpt, summary, size in items:
        print(f"  uploading {run}  ({size/1048576:.0f} MB)")
        api.upload_file(path_or_fileobj=ckpt, path_in_repo=f"{run}/best_model.pth",
                        repo_id=args.repo, repo_type="model")
        if summary:
            api.upload_file(path_or_fileobj=summary,
                            path_in_repo=f"{run}/training_summary.json",
                            repo_id=args.repo, repo_type="model")
    print(f"\n  done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
