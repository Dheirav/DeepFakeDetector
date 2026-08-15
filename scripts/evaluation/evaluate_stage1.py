import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ensure scripts dir and repo root on sys.path (same logic as evaluate.py)
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
if repo_root not in sys.path:
    sys.path.insert(1, repo_root)

from evaluation.evaluate import load_model, DeepfakeDataset, val_transform


def run_stage1(model_path, data_dir, save_dir, batch_size=64, num_workers=2, attention_head="none"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device, attention_head=attention_head)

    dataset = DeepfakeDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Stage1 inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = softmax(outputs).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

        # Collect image paths from dataset in order
    # DeepfakeDataset exposes .image_paths or equivalent; fallback to building using dataset
    try:
        paths = np.array(dataset.image_paths)
    except Exception:
        # build path list from dataset.samples
        try:
            paths = np.array([dataset.samples[i][0] for i in range(len(dataset))])
        except Exception:
            paths = np.array([str(i) for i in range(len(dataset))])

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "stage1_probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "stage1_preds.npy"), all_preds)
    np.save(os.path.join(save_dir, "y_true.npy"), all_labels)
    np.save(os.path.join(save_dir, "image_paths.npy"), paths)

    print(f"Saved Stage-1 outputs to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage-1 evaluation (3-class)")
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default="dataset_builder/test")
    parser.add_argument('--save_dir', default="results/cascade_stage1")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--attention_head', type=str, default="none", choices=['none','gem','cbam'])

    args = parser.parse_args()
    run_stage1(args.model_path, args.data_dir, args.save_dir, args.batch_size, args.num_workers, args.attention_head)


if __name__ == "__main__":
    main()
