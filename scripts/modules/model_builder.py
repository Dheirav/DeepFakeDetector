"""Single source of truth for building and loading models.

Before this module there were three independent reimplementations of model
construction -- in ``train_full.py``, ``evaluation/evaluate.py`` and
``frontend/inference.py`` -- and they disagreed in eleven documented ways. The
two loaders guessed the architecture from state-dict *key names*, which failed
silently and produced published numbers for models that were never trained:

* An FFT-only checkpoint has no key containing "fft" (``FFTLayer`` has no
  parameters), so it was rebuilt as a bare backbone and loaded **zero** weights.
  ``results/20`` scored 34.91% -- chance -- against an 88.37% training log.
* CBAM re-nests ``features``, so the ConvNeXt-Small probe (``features.5.20``)
  missed and Small loaded as Tiny. Identical channel widths meant *no size
  mismatch*, so ``strict=False`` swallowed 162 surplus tensors silently.
  ``results/18`` scored 70.94% against 83.56%.

The fix is to stop guessing. ``training_summary.json`` sits next to every
checkpoint and records exactly what was built. Read that, and load ``strict=True``
so a mismatch is an error rather than a quiet wrong answer.
"""

import json
import os

import torch
import torch.nn as nn
import torchvision.models as tvm

from preprocessing.srm import SRMLayer, adapt_conv1_for_srm
from preprocessing.fft import FFTLayer
from modules.attention_heads import GeM, CBAMBlock

__all__ = ["PreprocessNet", "build_model", "load_model", "config_for_checkpoint"]


class PreprocessNet(nn.Module):
    """Wraps a backbone with optional SRM residual and FFT magnitude channels.

    ``SRMLayer`` returns ``[rgb || residuals]``; only the residual half is
    appended so the backbone sees 3 + 3 + 1 channels at most, not 6 + 3 + 1.
    """

    def __init__(self, srm, fft, backbone):
        super().__init__()
        self.srm = srm
        self.fft = fft
        self.backbone = backbone

    def forward(self, x):
        feats = [x]
        if self.srm is not None:
            out = self.srm(x)
            feats.append(out[:, x.shape[1]:, ...] if out.shape[1] == x.shape[1] * 2 else out)
        if self.fft is not None:
            feats.append(self.fft(x))
        return self.backbone(torch.cat(feats, dim=1))


def _make_head(in_features, num_classes, dropout_p):
    if dropout_p and dropout_p > 0:
        return nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def _backbone(name, num_classes, dropout_p):
    """Return ``(module, penultimate_channels, arch_family)``."""
    if name == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = _make_head(512, num_classes, dropout_p)
        return m, 512, "resnet"
    if name == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = _make_head(2048, num_classes, dropout_p)
        return m, 2048, "resnet"
    if name in ("convnext_tiny", "convnext_small"):
        m = getattr(tvm, name)(weights=None)
        ch = m.classifier[2].in_features
        m.classifier[2] = _make_head(ch, num_classes, dropout_p)
        return m, ch, "convnext"
    if name == "efficientnet_b3":
        m = tvm.efficientnet_b3(weights=None)
        ch = m.classifier[1].in_features
        m.classifier[1] = _make_head(ch, num_classes, dropout_p)
        return m, ch, "efficientnet"
    if name == "vit_b_16":
        m = tvm.vit_b_16(weights=None)
        m.heads.head = _make_head(m.heads.head.in_features, num_classes, dropout_p)
        return m, None, "vit"
    raise ValueError(
        f"Unsupported backbone {name!r}. Choose: resnet18, resnet50, "
        "convnext_tiny, convnext_small, efficientnet_b3, vit_b_16"
    )


def _replace_stem_conv(module, in_channels):
    """Depth-first replace the first Conv2d, preserving pretrained RGB weights.

    ConvNeXt stem layouts differ between torchvision versions -- in 0.25
    ``features[0]`` is a ``Conv2dNormActivation`` whose ``[0]`` is already the
    ``Conv2d``. Searching rather than indexing is what makes this version-proof;
    indexing is the bug that made every ConvNeXt+SRM checkpoint unloadable in the
    frontend (``'Conv2d' object is not subscriptable``).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, adapt_conv1_for_srm(child, in_channels))
            return True
        if _replace_stem_conv(child, in_channels):
            return True
    return False


def build_model(cfg):
    """Build a model from a ``training_summary.json``-style config dict.

    Recognised keys: ``backbone`` (required), ``num_classes`` (default 3),
    ``dropout_p``, ``attention_head`` (none/gem/cbam), ``gem_p``,
    ``gem_learnable``, ``cbam_reduction``, ``cbam_kernel``, ``use_srm``, ``use_fft``.
    """
    name = cfg.get("backbone")
    if not name:
        raise ValueError("config has no 'backbone' key -- cannot build blind")

    model, channels, arch = _backbone(
        name, int(cfg.get("num_classes", 3)), float(cfg.get("dropout_p", 0.4) or 0.0)
    )

    head = cfg.get("attention_head") or "none"
    if head != "none":
        if arch == "vit":
            raise ValueError("attention_head is only supported for CNN backbones")
        if head == "cbam":
            block = CBAMBlock(
                channels=channels,
                reduction=int(cfg.get("cbam_reduction", 16)),
                kernel_size=int(cfg.get("cbam_kernel", 7)),
            )
            if arch == "resnet":
                model.layer4 = nn.Sequential(model.layer4, block)
            else:
                model.features = nn.Sequential(model.features, block)
        elif head == "gem":
            model.avgpool = GeM(
                p=float(cfg.get("gem_p", 3.0)),
                learnable=bool(cfg.get("gem_learnable", False)),
            )
        else:
            raise ValueError(f"Unknown attention_head {head!r}. Choose: none, gem, cbam.")

    srm = SRMLayer() if cfg.get("use_srm") else None
    fft = FFTLayer() if cfg.get("use_fft") else None
    if srm is not None or fft is not None:
        in_ch = 3 + (3 if srm is not None else 0) + (1 if fft is not None else 0)
        if arch == "resnet":
            model.conv1 = adapt_conv1_for_srm(model.conv1, in_ch)
        elif not _replace_stem_conv(model.features[0], in_ch):
            raise RuntimeError(f"could not locate stem Conv2d in {name} to adapt for SRM/FFT")
        model = PreprocessNet(srm, fft, model)

    return model


def config_for_checkpoint(path):
    """Find the config describing a checkpoint.

    Preferred: embedded in the checkpoint under ``"config"`` (written by
    ``train_full.py`` for new runs). Fallback: ``results/<run>/training_summary.json``
    for the runs trained before checkpoints carried their own config.
    """
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "config" in blob and "state_dict" in blob:
        return blob["config"]

    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    for candidate in (
        os.path.join(root, "results", run, "training_summary.json"),
        os.path.join(os.path.dirname(os.path.abspath(path)), "training_summary.json"),
    ):
        if os.path.isfile(candidate):
            return json.load(open(candidate))["config"]

    raise FileNotFoundError(
        f"No config found for {path}. Expected it embedded in the checkpoint or at "
        f"results/{run}/training_summary.json. Refusing to guess the architecture -- "
        "guessing is what produced results/18 and results/20."
    )


def load_model(path, device="cpu", cfg=None, strict=True):
    """Load a checkpoint into the architecture its config describes.

    ``strict=True`` by default and you should leave it that way: every silently
    wrong number in this repository came from ``strict=False`` hiding a mismatch.
    """
    cfg = cfg if cfg is not None else config_for_checkpoint(path)
    blob = torch.load(path, map_location=device)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    # torch.compile prefixes every key; strip it so compiled and eager
    # checkpoints are interchangeable.
    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    model = build_model(cfg)
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval(), cfg
