import os
import cv2
import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    label_map = {
        "real": 0,
        "ai_generated": 1,
        "ai_edited": 2
    }

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for cls in self.label_map:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                print(f"[DeepfakeDataset] Warning: class folder not found, skipping: {cls_path}")
                continue
            for img in sorted(os.listdir(cls_path)):
                full_path = os.path.join(cls_path, img)
                if os.path.isfile(full_path):
                    self.samples.append((full_path, self.label_map[cls]))

        if not self.samples:
            raise RuntimeError(f"No samples found in '{root_dir}'. Expected at least one of: {list(self.label_map.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label)
