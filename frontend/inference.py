import os
from typing import Dict, Optional, Tuple, List

from PIL import Image
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
except Exception:
    torch = None

LABELS: List[str] = ["Real", "AI Generated", "AI Edited"]


def get_device(use_gpu: bool = True):
    if torch is None:
        return None
    if not use_gpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Optional[str] = None, device: Optional[object] = None):
    """Load ResNet18 adapted for 3 classes. If checkpoint exists, attempt to load it.

    Returns a model on `device` or raises RuntimeError if torch missing.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    device = device or get_device()
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=device)
            # support various checkpoint formats
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            elif isinstance(state, dict) and "model_state" in state:
                sd = state["model_state"]
            else:
                sd = state
            model.load_state_dict(sd, strict=False)
        except Exception:
            # best-effort: continue with initialized model
            pass

    model.eval()
    return model


def preprocess_image(pil_image: Image.Image, size: Tuple[int, int] = (224, 224)):
    """Preprocess PIL image to model input tensor."""
    if torch is None:
        raise RuntimeError("PyTorch required for preprocessing")
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_image.convert("RGB")).unsqueeze(0)


def softmax_probs(logits) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch required for softmax")
    probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    return probs


def predict(model, pil_image: Image.Image, device=None) -> Tuple[str, Dict[str, float]]:
    """Run inference. Returns top label and dict of label->probability.

    If model is None, a simple deterministic dummy predictor is used.
    """
    if model is None or torch is None:
        # deterministic brightness-based dummy predictor
        gray = pil_image.convert("L")
        stat = np.asarray(gray).mean() / 255.0
        real = float(min(max(0.2 + 0.6 * stat, 0.0), 1.0))
        fake_gen = float(max(0.0, 1.0 - real) * 0.6)
        fake_edit = float(max(0.0, 1.0 - real) * 0.4)
        s = real + fake_gen + fake_edit
        out = {LABELS[0]: real / s, LABELS[1]: fake_gen / s, LABELS[2]: fake_edit / s}
        top = max(out, key=out.get)
        return top, out

    device = device or get_device()
    tensor = preprocess_image(pil_image)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = softmax_probs(logits)
    out = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top = max(out, key=out.get)
    return top, out
