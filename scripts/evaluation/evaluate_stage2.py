import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ensure scripts dir and repo root on sys.path
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
if repo_root not in sys.path:
    sys.path.insert(1, repo_root)

from evaluation.evaluate import load_model, DeepfakeDataset, val_transform


def run_stage2(stage2_model_path, data_dir, indices_file, save_dir, batch_size=64, num_workers=2, attention_head="none"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(stage2_model_path, device, attention_head=attention_head)

    indices = np.load(indices_file)
    if indices.size == 0:
        print("No samples to run in Stage-2. Exiting.")
        return

    dataset = DeepfakeDataset(data_dir, transform=val_transform)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    all_preds = []
    all_probs = []
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Stage2 inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = softmax(outputs).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "stage2_probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "stage2_preds.npy"), all_preds)
    np.save(os.path.join(save_dir, "stage2_indices.npy"), indices)

    print(f"Saved Stage-2 outputs to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage-2 evaluation (binary: Real vs AI Edited)")
    parser.add_argument('--stage2_model_path', required=True)
    parser.add_argument('--data_dir', default="dataset_builder/test")
    parser.add_argument('--indices_file', default="results/cascade_stage2/stage2_indices.npy")
    parser.add_argument('--save_dir', default="results/cascade_stage2")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--attention_head', type=str, default="none", choices=['none','gem','cbam'])

    args = parser.parse_args()
    run_stage2(args.stage2_model_path, args.data_dir, args.indices_file, args.save_dir, args.batch_size, args.num_workers, args.attention_head)


if __name__ == "__main__":
    main()
