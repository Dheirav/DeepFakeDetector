import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── light ─────────────────────────────────────────────────────────────────────
# Minimal transforms — fastest training, use for quick sanity checks.
train_transform_light = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# ── standard (default) ────────────────────────────────────────────────────────
# Balanced augmentation used in all baseline and sweep runs.
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ImageCompression(quality_range=(30, 100), p=0.4),
    A.Normalize(),
    ToTensorV2()
])

# ── strong ────────────────────────────────────────────────────────────────────
# Targets the Real ↔ AI-Edited boundary:
#   - Wider JPEG compression range forces the model to learn artefact-independent
#     Real features rather than using compression quality as a shortcut.
#   - Gaussian / ISO noise simulates sensor noise present in real camera images
#     but absent in many AI-generated/edited outputs.
#   - Slight blur prevents the model from over-relying on high-frequency texture
#     that may differ between datasets rather than between classes.
train_transform_strong = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    # Wider JPEG range vs standard (20-95 vs 30-100), higher probability
    A.ImageCompression(quality_range=(20, 95), p=0.5),
    # Sensor / transmission noise
    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.4),
    # Slight blur to prevent texture overfitting
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.3),
    A.Normalize(),
    ToTensorV2()
])


def get_train_transform(mode: str = "standard") -> A.Compose:
    """Return the training transform for the given augmentation mode.

    Args:
        mode: One of ``'light'``, ``'standard'`` (default), or ``'strong'``.
    """
    if mode == "light":
        return train_transform_light
    elif mode == "strong":
        return train_transform_strong
    elif mode == "standard":
        return train_transform
    else:
        raise ValueError(f"Unknown augment mode '{mode}'. Choose: light, standard, strong")


val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])
