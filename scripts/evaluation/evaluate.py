import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.dataset import DeepfakeDataset
from preprocessing.preprocessing import val_transform
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm
from preprocessing.fft import FFTLayer
from modules.attention_heads import GeM, CBAMBlock

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]

# Project root = two levels above this script (scripts/evaluation/ → root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _adapt_conv1_generic(conv1: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    """Create a new Conv2d with `new_in_channels` inputs, copying existing weights.

    Extra input channels are initialised to a small fraction of the mean RGB weights.
    """
    out_ch, in_ch, kH, kW = conv1.weight.shape
    new_conv = nn.Conv2d(
        new_in_channels,
        out_ch,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        copy_in = min(in_ch, new_in_channels)
        new_conv.weight[:, :copy_in] = conv1.weight[:, :copy_in]
        if new_in_channels > in_ch:
            # initialise extra channels to 0.1 * mean of existing input-channel weights
            mean_w = conv1.weight.mean(dim=1, keepdim=True)
            extra = mean_w.repeat(1, new_in_channels - in_ch, 1, 1) * 0.1
            new_conv.weight[:, in_ch:] = extra
        if conv1.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    return new_conv

def _get_in_features(layer):
    """Return in_features even if layer is Sequential(Dropout, Linear)."""
    if isinstance(layer, nn.Sequential):
        for m in reversed(layer):
            if hasattr(m, "in_features"):
                return m.in_features
    return layer.in_features

def _derive_save_dir(model_path: str) -> str:
    """
    Derive the results directory from the model checkpoint path.

    Priority:
    1. If the path contains a timestamped run folder (run_YYYYMMDD_HHMMSS),
       mirror it under results/ so all artefacts from the same training run
       stay together (e.g. results/run_20260307_063053/).
    2. If the model lives inside a named subfolder (e.g. models/my_exp/best.pth),
       use that folder name (e.g. results/my_exp/).
    3. Otherwise fall back to results/<checkpoint_stem>/ derived from the
       filename itself (e.g. models/best_resnet18.pth → results/best_resnet18/).
    """
    abs_path = os.path.abspath(model_path)

    # 1. Timestamped run folder
    match = re.search(r"(run_\d{8}_\d{6})", abs_path)
    if match:
        return os.path.join(_PROJECT_ROOT, "results", match.group(1))

    # 2. Named parent folder (anything that isn't the bare models/ root)
    parent = os.path.basename(os.path.dirname(abs_path))
    models_root = os.path.basename(
        os.path.abspath(os.path.join(_PROJECT_ROOT, "models"))
    )
    if parent and parent != models_root:
        return os.path.join(_PROJECT_ROOT, "results", parent)

    # 3. Checkpoint filename stem
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    return os.path.join(_PROJECT_ROOT, "results", stem)

def _apply_attention(
    backbone: nn.Module,
    arch: str,
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
) -> nn.Module:
    """Apply optional GeM / CBAM heads so architecture matches training."""

    if attention_head in ("none", "", None):
        return backbone

    if arch == "vit":
        raise ValueError("attention_head is only supported for CNN backbones, not ViT.")

    # Keep a reference to the original model (may be a PreprocessNet wrapper)
    original_model = backbone
    # ─────────────────────────────────────────────────────────────
    # Determine where the real backbone lives (core)
    # If model is a PreprocessNet wrapper it stores backbone in .backbone
    # ─────────────────────────────────────────────────────────────
    if arch.endswith("_srm") and hasattr(original_model, "backbone"):
        core = original_model.backbone
        arch_family = arch.replace("_srm", "")
    else:
        core = original_model
        arch_family = arch

    # ─────────────────────────────────────────────────────────────
    # Extract channel size safely (handles Sequential heads)
    # ─────────────────────────────────────────────────────────────
    def get_in_features(layer):
        if isinstance(layer, nn.Sequential):
            for m in reversed(layer):
                if hasattr(m, "in_features"):
                    return m.in_features
        return layer.in_features

    # core is the real backbone module; inspect its head to find output feature size
    if hasattr(core, "fc"):
        in_channels = get_in_features(core.fc)
    elif hasattr(core, "classifier"):
        in_channels = get_in_features(core.classifier[2])
    else:
        raise ValueError(f"Unknown model type for attention: {type(core)}")

    # ─────────────────────────────────────────────────────────────
    # Apply attention module
    # ─────────────────────────────────────────────────────────────
    if attention_head == "cbam":

        cbam = CBAMBlock(
            channels=in_channels,
            reduction=cbam_reduction,
            kernel_size=cbam_kernel,
        )

        if arch_family in ("resnet18", "resnet50"):
            core.layer4 = nn.Sequential(core.layer4, cbam)

        elif arch_family in ("convnext_tiny", "convnext_small"):
            core.features = nn.Sequential(core.features, cbam)

    elif attention_head == "gem":

        gem = GeM(p=gem_p, learnable=gem_learnable)

        if arch_family in ("resnet18", "resnet50", "convnext_tiny", "convnext_small"):
            core.avgpool = gem

    else:
        raise ValueError(
            f"Unknown attention_head '{attention_head}'. Choose: none, gem, cbam."
        )

    # return the original model (wrapper) so PreprocessNet is preserved
    return original_model

def _build_srm_net(
    device,
    arch: str = "resnet18",
    sequential_fc: bool = False,
    has_srm: bool = False,
    has_fft: bool = False,
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
):
    """Reconstruct the SRM-wrapped model used during training."""


    def make_fc(in_features):
        if sequential_fc:
            return nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, 3))
        return nn.Linear(in_features, 3)

    # Initialize srm_layer to None at the start
    srm_layer = None
    in_channels = 3
    # Compute desired input channels: RGB + SRM residuals (3) + FFT magnitude (1)
    # Use the explicit `has_srm` argument here (avoid referencing `use_srm` before it's set)
    desired_in_channels = 3 + (3 if has_srm else 0) + (1 if has_fft else 0)

    if has_fft:
        in_channels += 3

    # Use explicit flag passed from load_model to decide SRM usage
    use_srm = bool(has_srm)
    if use_srm:
        in_channels += 3

    # Build backbone and adapt input channels if needed
    if arch.startswith("convnext_tiny"):
        backbone = models.convnext_tiny(weights=None)
        backbone.classifier[2] = make_fc(backbone.classifier[2].in_features)
        if use_srm:
            srm_layer = SRMLayer().to(device)
            # adapt ConvNeXt stem conv to accept desired_in_channels
            stem = backbone.features[0][0]
            # try common locations for the conv
            if hasattr(stem, "conv") and isinstance(stem.conv, nn.Conv2d):
                stem.conv = adapt_conv1_for_srm(stem.conv) if desired_in_channels == 6 else _adapt_conv1_generic(stem.conv, desired_in_channels)
            elif isinstance(stem, (nn.Sequential,)):
                replaced = False
                for i, m in enumerate(stem):
                    if isinstance(m, nn.Conv2d):
                        stem[i] = adapt_conv1_for_srm(m) if desired_in_channels == 6 else _adapt_conv1_generic(m, desired_in_channels)
                        replaced = True
                        break
                if not replaced and len(stem) > 0 and isinstance(stem[0], nn.Conv2d):
                    stem[0] = adapt_conv1_for_srm(stem[0]) if desired_in_channels == 6 else _adapt_conv1_generic(stem[0], desired_in_channels)
            elif isinstance(stem, nn.Conv2d):
                backbone.features[0][0] = adapt_conv1_for_srm(stem) if desired_in_channels == 6 else _adapt_conv1_generic(stem, desired_in_channels)
            else:
                try:
                    backbone.features[0][0] = adapt_conv1_for_srm(stem)
                except Exception:
                    pass
        arch_label = "convnext_tiny" + ("_srm" if use_srm else "")

    elif arch.startswith("convnext_small"):
        backbone = models.convnext_small(weights=None)
        backbone.classifier[2] = make_fc(backbone.classifier[2].in_features)
        if use_srm:
            srm_layer = SRMLayer().to(device)
            if use_srm or has_fft:
                srm_layer = SRMLayer().to(device) if use_srm else None
                stem = backbone.features[0][0]
                if hasattr(stem, "conv") and isinstance(stem.conv, nn.Conv2d):
                    stem.conv = adapt_conv1_for_srm(stem.conv) if desired_in_channels == 6 else _adapt_conv1_generic(stem.conv, desired_in_channels)
                elif isinstance(stem, (nn.Sequential,)):
                    for i, m in enumerate(stem):
                        if isinstance(m, nn.Conv2d):
                            stem[i] = adapt_conv1_for_srm(m) if desired_in_channels == 6 else _adapt_conv1_generic(m, desired_in_channels)
                            break
                elif isinstance(stem, nn.Conv2d):
                    backbone.features[0][0] = adapt_conv1_for_srm(stem) if desired_in_channels == 6 else _adapt_conv1_generic(stem, desired_in_channels)
                else:
                    try:
                        backbone.features[0][0] = adapt_conv1_for_srm(stem)
                    except Exception:
                        pass
        arch_label = "convnext_small" + ("_srm" if use_srm else "")

    elif arch.startswith("resnet50"):
        backbone = models.resnet50(weights=None)
        backbone.fc = make_fc(2048)
        if use_srm:
            srm_layer = SRMLayer().to(device)
            backbone.conv1 = adapt_conv1_for_srm(backbone.conv1)
        if use_srm or has_fft:
            srm_layer = SRMLayer().to(device) if use_srm else None
            # adapt conv1 to desired_in_channels
            backbone.conv1 = adapt_conv1_for_srm(backbone.conv1) if desired_in_channels == 6 else _adapt_conv1_generic(backbone.conv1, desired_in_channels)
        arch_label = "resnet50" + ("_srm" if use_srm else "")

    else:  # default resnet18
        backbone = models.resnet18(weights=None)
        backbone.fc = make_fc(512)
        if use_srm or has_fft:
            srm_layer = SRMLayer().to(device) if use_srm else None
            backbone.conv1 = adapt_conv1_for_srm(backbone.conv1) if desired_in_channels == 6 else _adapt_conv1_generic(backbone.conv1, desired_in_channels)
        arch_label = "resnet18" + ("_srm" if use_srm else "")

    class PreprocessNet(nn.Module):
        def __init__(self, srm, fft, backbone):
            super().__init__()
            self.srm = srm
            self.fft = fft
            self.backbone = backbone

            # Expected input channels for the backbone's first conv (filled later)
            self.expected_in_channels = None

        def forward(self, x):

            rgb = x
            features = [rgb]

            # SRM: some SRM implementations return [rgb || residuals] (6ch)
            # while others return only residuals (3ch). Normalize to residuals.
            if self.srm is not None:
                srm_out = self.srm(rgb)
                # If SRM returned rgb+residuals, keep only the residual channels
                if srm_out.shape[1] == 6:
                    residuals = srm_out[:, 3:, ...]
                else:
                    residuals = srm_out
                features.append(residuals)

            # FFT: some FFT layers return rgb+magnitude (4ch) or just magnitude (1ch)
            if self.fft is not None:
                fft_out = self.fft(rgb)
                if fft_out.shape[1] >= 4:
                    # assume last channel(s) are magnitude(s)
                    magnitude = fft_out[:, -1:, ...]
                else:
                    magnitude = fft_out
                features.append(magnitude)

            x = torch.cat(features, dim=1)

            # If backbone was trained with extra channels (e.g. RGB+SRM+FFT=7)
            # but our preprocessing produced fewer channels, pad with zeros so
            # the tensor shape matches the backbone's expected input channels.
            if getattr(self, "expected_in_channels", None) is not None:
                cur_ch = x.shape[1]
                req_ch = self.expected_in_channels
                if cur_ch < req_ch:
                    pad = req_ch - cur_ch
                    zeros = torch.zeros(x.size(0), pad, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
                    x = torch.cat([x, zeros], dim=1)
                elif cur_ch > req_ch:
                    x = x[:, :req_ch, ...]

            return self.backbone(x)

    fft_layer = FFTLayer().to(device) if has_fft else None
    net = PreprocessNet(srm_layer, fft_layer, backbone)
    # Determine backbone's first conv expected input channels and record it
    try:
        target = backbone
        model_conv = None
        if hasattr(target, "features"):
            try:
                stem = target.features[0][0]
            except Exception:
                stem = target.features[0]
            if hasattr(stem, "conv") and isinstance(stem.conv, nn.Conv2d):
                model_conv = stem.conv
            elif isinstance(stem, (nn.Sequential,)):
                for i, m in enumerate(stem):
                    if isinstance(m, nn.Conv2d):
                        model_conv = m
                        break
            elif isinstance(stem, nn.Conv2d):
                model_conv = stem
        if model_conv is None and hasattr(target, "conv1") and isinstance(target.conv1, nn.Conv2d):
            model_conv = target.conv1
        if model_conv is not None:
            net.expected_in_channels = model_conv.weight.shape[1]
    except Exception:
        pass
    net = _apply_attention(
        net,
        arch=arch_label,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )
    return net


def load_model(
    model_path,
    device,
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
):
    raw_sd = torch.load(model_path, map_location=device)

    # torch.compile wraps every key with "_orig_mod."
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}

    # Detect if SRM wrapper was used
    has_srm = any("srm" in k.lower() for k in raw_sd)
    has_fft = any("fft" in k.lower() or "FFT" in k.lower() for k in raw_sd)

    # Detect ConvNeXt
    is_convnext = any("features." in k for k in raw_sd)

    # Detect ResNet50 vs ResNet18
    is_resnet50 = any(
        v.shape[-1] == 2048
        for k, v in raw_sd.items()
        if "fc" in k and "weight" in k
    )

    # Detect ConvNeXt Small by looking at stage-3 block depth
    is_convnext_small = any(
        "features.5.20" in k or "backbone.features.5.20" in k
        for k in raw_sd
    )

    # Detect Sequential FC head
    sequential_fc = any(
        k.endswith("fc.1.weight") or k.endswith("classifier.2.1.weight")
        for k in raw_sd
    )

    # Determine architecture name
    if is_convnext:
        arch = "convnext_small" if is_convnext_small else "convnext_tiny"
    else:
        arch = "resnet50" if is_resnet50 else "resnet18"

    fc_label = "Sequential(Dropout+Linear)" if sequential_fc else "Linear"

    if has_srm or has_fft:
        print(f"Detected preprocessing: SRM={has_srm} FFT={has_fft}")
        print(f"Detected SRM/FFT checkpoint — rebuilding PreprocessNet ({arch}, FC: {fc_label}).")

        model = _build_srm_net(
            device,
            arch=arch,
            sequential_fc=sequential_fc,
            has_srm=has_srm,
            has_fft=has_fft,
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
        )

    elif arch == "convnext_tiny":
        print("Detected plain ConvNeXt-Tiny checkpoint.")
        model = models.convnext_tiny(weights=None)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_feat, 3))
            if sequential_fc else nn.Linear(in_feat, 3)
        )
        model = _apply_attention(
            model,
            arch="convnext_tiny",
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
        )

    elif arch == "convnext_small":
        print("Detected plain ConvNeXt-Small checkpoint.")
        model = models.convnext_small(weights=None)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_feat, 3))
            if sequential_fc else nn.Linear(in_feat, 3)
        )
        model = _apply_attention(
            model,
            arch="convnext_small",
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
        )

    elif arch == "resnet50":
        print("Detected plain ResNet50 checkpoint.")
        model = models.resnet50(weights=None)
        in_feat = 2048
        model.fc = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_feat, 3))
            if sequential_fc else nn.Linear(in_feat, 3)
        )
        model = _apply_attention(
            model,
            arch="resnet50",
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
        )

    else:
        print("Detected plain ResNet18 checkpoint.")
        model = models.resnet18(weights=None)
        in_feat = 512
        model.fc = (
            nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_feat, 3))
            if sequential_fc else nn.Linear(in_feat, 3)
        )
        model = _apply_attention(
            model,
            arch="resnet18",
            attention_head=attention_head,
            gem_p=gem_p,
            gem_learnable=gem_learnable,
            cbam_reduction=cbam_reduction,
            cbam_kernel=cbam_kernel,
        )

    # If checkpoint's first conv expects a different number of input channels
    # (e.g. RGB + SRM + FFT = 7) adapt the constructed model's stem conv to match
    # so weights can be loaded without size-mismatch errors.
    try:
        ckpt_in_ch = None
        # common checkpoint keys for first conv in various backbones
        for k, v in raw_sd.items():
            if k.endswith("features.0.0.weight") or k.endswith("backbone.features.0.0.weight"):
                ckpt_in_ch = v.shape[1]
                break
            if k.endswith("conv1.weight") or k.endswith("backbone.conv1.weight"):
                ckpt_in_ch = v.shape[1]
                break

        if ckpt_in_ch is not None:
            # operate on the real backbone when wrapped by PreprocessNet
            target = model.backbone if hasattr(model, "backbone") else model

            # locate model's first conv module for common backbones
            model_conv = None
            # ConvNeXt-style: target.features[0][0] or target.features[0]
            if hasattr(target, "features"):
                try:
                    stem = target.features[0][0]
                except Exception:
                    stem = target.features[0]
                if hasattr(stem, "conv") and isinstance(stem.conv, nn.Conv2d):
                    model_conv = stem.conv
                elif isinstance(stem, (nn.Sequential,)):
                    for i, m in enumerate(stem):
                        if isinstance(m, nn.Conv2d):
                            model_conv = m
                            break
                elif isinstance(stem, nn.Conv2d):
                    model_conv = stem

            # ResNet-style
            if model_conv is None and hasattr(target, "conv1") and isinstance(target.conv1, nn.Conv2d):
                model_conv = target.conv1

            # Adapt if mismatch
            if model_conv is not None:
                model_in_ch = model_conv.weight.shape[1]
                if model_in_ch != ckpt_in_ch:
                    # prefer SRM-specific adaptor when adapting to 6 channels
                    if ckpt_in_ch == 6:
                        new_conv = adapt_conv1_for_srm(model_conv, in_channels=6)
                    else:
                        new_conv = _adapt_conv1_generic(model_conv, ckpt_in_ch)

                    # place adapted conv back into the target backbone (handle a few shapes)
                    replaced = False
                    if hasattr(target, "features"):
                        try:
                            if hasattr(target.features[0][0], "conv") and isinstance(target.features[0][0].conv, nn.Conv2d):
                                target.features[0][0].conv = new_conv
                                replaced = True
                        except Exception:
                            pass
                        if not replaced:
                            # try to replace first Conv2d inside features[0]
                            try:
                                stem = target.features[0]
                                for i, m in enumerate(stem):
                                    if isinstance(m, nn.Conv2d):
                                        stem[i] = new_conv
                                        replaced = True
                                        break
                            except Exception:
                                pass
                    if not replaced and hasattr(target, "conv1"):
                        target.conv1 = new_conv

                    # ensure PreprocessNet knows the backbone's expected channels
                    if hasattr(model, "expected_in_channels"):
                        model.expected_in_channels = ckpt_in_ch

    except Exception:
        # best-effort adaptation — if anything goes wrong, continue to loading
        pass

    model.load_state_dict(raw_sd, strict=False)
    model.to(device)
    model.eval()

    return model


def run_evaluation(
    model_path,
    data_dir,
    batch_size=64,
    save_dir="../results",
    attention_head: str = "none",
    gem_p: float = 3.0,
    gem_learnable: bool = False,
    cbam_reduction: int = 16,
    cbam_kernel: int = 7,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Attention head: {attention_head}")
    model = load_model(
        model_path,
        device,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )

    dataset = DeepfakeDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(dataset)} | Batches: {len(loader)}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Only report on classes that are actually present in the dataset so
    # classification_report doesn't fail when some splits are missing.
    present_labels = sorted(set(y_true))
    present_names  = [CLASS_NAMES[i] for i in present_labels]

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, digits=4))

    print("CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred, labels=present_labels))

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
    print(f"\nSaved y_true.npy and y_pred.npy to {save_dir}")
    print(f"Overall accuracy: {(y_true == y_pred).mean()*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model on test set")
    parser.add_argument('--model_path',  type=str, required=True,
                        help='Path to model .pth checkpoint')
    parser.add_argument('--data_dir',    type=str, default="dataset_builder/test",
                        help='Test data directory (with real/, ai_generated/, ai_edited/ subfolders)')
    parser.add_argument('--batch_size',  type=int, default=64, help='Batch size')
    parser.add_argument('--save_dir',    type=str, default=None,
                        help=(
                            'Directory to save y_true.npy and y_pred.npy. '
                            'Defaults to results/<run_id>/ derived from --model_path, '
                            'or results/ if no run ID is found in the checkpoint path.'
                        ))
    parser.add_argument(
        '--attention_head',
        type=str,
        default=None,
        choices=['none', 'gem', 'cbam'],
        help='Attention head used during training. '
             'If omitted, will be inferred from training_summary.json when available.',
    )
    parser.add_argument('--gem_p', type=float, default=None,
                        help='GeM pooling exponent p (overrides summary if set)')
    parser.add_argument('--gem_learnable', action='store_true',
                        help='Use learnable GeM exponent p (overrides summary if set)')
    parser.add_argument('--cbam_reduction', type=int, default=None,
                        help='CBAM channel reduction ratio (overrides summary if set)')
    parser.add_argument('--cbam_kernel', type=int, default=None,
                        help='CBAM spatial attention kernel size, 3 or 7 (overrides summary if set)')

    args = parser.parse_args()

    # Auto-derive save_dir from the checkpoint path when not explicitly set.
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else _derive_save_dir(args.model_path)
    print(f"Results will be saved to: {save_dir}")

    # Try to infer attention configuration from training summary when not provided.
    summary_path = os.path.join(save_dir, "training_summary.json")
    summary_cfg = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            summary_cfg = data.get("config", {})
            print(f"Loaded attention config from {summary_path}")
        except Exception as e:
            print(f"Warning: failed to read {summary_path}: {e}")

    def _cfg_or(default, cli_value, key):
        if cli_value is not None:
            return cli_value
        return summary_cfg.get(key, default)

    attention_head = args.attention_head if args.attention_head is not None else summary_cfg.get("attention_head", "none")
    gem_p = _cfg_or(3.0, args.gem_p, "gem_p")
    # gem_learnable stored as bool; CLI flag only enables it.
    gem_learnable = summary_cfg.get("gem_learnable", False) or args.gem_learnable
    cbam_reduction = _cfg_or(16, args.cbam_reduction, "cbam_reduction")
    cbam_kernel = _cfg_or(7, args.cbam_kernel, "cbam_kernel")

    run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        save_dir=save_dir,
        attention_head=attention_head,
        gem_p=gem_p,
        gem_learnable=gem_learnable,
        cbam_reduction=cbam_reduction,
        cbam_kernel=cbam_kernel,
    )
