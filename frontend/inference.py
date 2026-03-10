import os
import sys
from typing import Dict, Optional, Tuple, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, "scripts")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_PATH)

from preprocessing.srm import SRMLayer
from modules.attention_heads import GeM, CBAMBlock

from PIL import Image
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    from torchvision.models import ResNet18_Weights
except Exception:
    torch = None

LABELS: List[str] = ["Real", "AI Generated", "AI Edited"]


def get_device(use_gpu: bool = True):
    if torch is None:
        return None
    if not use_gpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Optional[str] = None, device=None):

    if torch is None:
        raise RuntimeError("PyTorch is not available")

    device = device or get_device()

    raw_sd = torch.load(checkpoint_path, map_location=device)

    # remove torch.compile prefix
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}

    # detect SRM
    has_srm = any(k.startswith("backbone.") for k in raw_sd)

    # detect convnext
    is_convnext = any("features." in k for k in raw_sd)

    # detect resnet50
    is_resnet50 = any(
        v.shape[-1] == 2048
        for k, v in raw_sd.items()
        if "fc" in k and "weight" in k
    )

    # detect convnext small
    is_convnext_small = any(
        "features.5.20" in k or "backbone.features.5.20" in k
        for k in raw_sd
    )

    # detect sequential fc
    sequential_fc = any(
        k.endswith("fc.1.weight") or k.endswith("classifier.2.1.weight")
        for k in raw_sd
    )

    # determine architecture
    if is_convnext:
        arch = "convnext_small" if is_convnext_small else "convnext_tiny"
    else:
        arch = "resnet50" if is_resnet50 else "resnet18"

    # build model
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_feat = 512
        model.fc = (
            torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(in_feat, 3))
            if sequential_fc else torch.nn.Linear(in_feat, 3)
        )

    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        in_feat = 2048
        model.fc = (
            torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(in_feat, 3))
            if sequential_fc else torch.nn.Linear(in_feat, 3)
        )

    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = (
            torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(in_feat, 3))
            if sequential_fc else torch.nn.Linear(in_feat, 3)
        )

    elif arch == "convnext_small":
        model = models.convnext_small(weights=None)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = (
            torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(in_feat, 3))
            if sequential_fc else torch.nn.Linear(in_feat, 3)
        )

    # SRM wrapper
    if has_srm:
        from preprocessing.srm import SRMLayer, adapt_conv1_for_srm

        srm = SRMLayer().to(device)

        if arch in ["resnet18", "resnet50"]:
            model.conv1 = adapt_conv1_for_srm(model.conv1)

        elif arch in ["convnext_tiny", "convnext_small"]:
            stem = model.features[0][0]
            stem[0] = adapt_conv1_for_srm(stem[0])

        class SRMNet(torch.nn.Module):
            def __init__(self, srm, backbone):
                super().__init__()
                self.srm = srm
                self.backbone = backbone

            def forward(self, x):
                return self.backbone(self.srm(x))

        model = SRMNet(srm, model)

    model.load_state_dict(raw_sd)

    model.to(device)
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
