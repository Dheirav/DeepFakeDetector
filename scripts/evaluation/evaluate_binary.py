import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ensure scripts dir and repo root on sys.path
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
if repo_root not in sys.path:
    sys.path.insert(1, repo_root)

from evaluation.evaluate import load_model
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import val_transform


def run_binary_eval(model_path, data_dir, save_dir, batch_size=64, num_workers=2, attention_head="none"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device, attention_head=attention_head)

    # only real and ai_edited
    dataset = DeepfakeDataset(data_dir, transform=val_transform, include_classes=["real","ai_edited"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    all_preds = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Binary Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = softmax(outputs).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)

    print("\nBinary classification report (0=Real,1=AI_Edited):")
    print(classification_report(y_true, y_pred, target_names=["Real","AI_Edited"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary model on Real vs AI_Edited validation set")
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='dataset_builder/val')
    parser.add_argument('--save_dir', default='results/stage2_binary_eval')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--attention_head', type=str, default='none', choices=['none','gem','cbam'])
    args = parser.parse_args()
    run_binary_eval(args.model_path, args.data_dir, args.save_dir, args.batch_size, args.num_workers, args.attention_head)


if __name__ == '__main__':
    main()
