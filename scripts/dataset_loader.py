import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# -----------------------------
# Label Mapping
# -----------------------------
LABEL_MAP = {
    "real": 0,
    "ai_generated": 1,
    "ai_edited": 2
}

# -----------------------------
# Custom Dataset Class
# -----------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_name, label in LABEL_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# Transforms (Placeholder)
# -----------------------------
default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Train / Validation Split
# -----------------------------
def create_dataloaders(
    data_dir,
    batch_size=32,
    test_size=0.2,
    shuffle=True
):
    dataset = DeepfakeDataset(data_dir, transform=default_transforms)

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        shuffle=True,
        random_state=42
    )

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, dataset

# -----------------------------
# Dataset Statistics
# -----------------------------
def print_dataset_stats(dataset):
    from collections import Counter
    label_counts = Counter(dataset.labels)

    print("Dataset Statistics:")
    for class_name, label in LABEL_MAP.items():
        print(f"{class_name}: {label_counts.get(label, 0)} images")

# -----------------------------
# Test Run (Safe even if empty)
# -----------------------------
if __name__ == "__main__":
    data_dir = "data"
    train_loader, val_loader, dataset = create_dataloaders(data_dir)

    print_dataset_stats(dataset)

    print(f"Total images: {len(dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
