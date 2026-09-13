"""Frontend adapter for the mask-head checkpoints (`results/mask_head_*`).

The old app loaded a fine-tuned ConvNeXt and explained it with Grad-CAM on a
conv layer. The rebuilt models are a frozen (or partly fine-tuned) ViT encoder
with a mask decoder and a small classifier, so two things change:

* The model already produces a supervised explanation: the predicted edit
  mask, trained against OpenSDI's ground-truth masks. That is a stronger
  account of "where" than any post-hoc heatmap, so it is the primary view.
* Grad-CAM still works, but on a ViT the "layer" is the final token grid.
  The hook below detaches the last transformer block's output and makes it a
  leaf, so the gradient of a class logit with respect to those tokens can be
  taken without unfreezing anything. Reshaped to the patch grid it is the same
  formula as the conv version: ReLU(sum_c mean_hw(dL/dA_c) * A_c).

Inputs are re-encoded to 512px JPEG q90 by default because every training
image went through exactly that; see scripts/inference/predict_mask_head.py.
"""

import io
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_root, os.path.join(_root, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

LABELS = ["Real", "AI Generated", "AI Edited"]
ABSTAIN = "Cannot tell"
DEFAULT_ABSTAIN_BELOW = 0.9


def load_decision_rule(checkpoint: str) -> dict:
    """The abstain threshold chosen by scripts/evaluation/abstain_sweep.py for
    this run, if it was written; otherwise the project default. Kept next to
    the checkpoint so a different run can carry a different line."""
    path = os.path.join(os.path.dirname(checkpoint), "decision_rule.json")
    if os.path.isfile(path):
        import json
        with open(path) as f:
            return json.load(f)
    return {"abstain_below": DEFAULT_ABSTAIN_BELOW}


def apply_rule(probs: Dict[str, float], abstain_below: float) -> str:
    """Argmax if the top probability clears the line, else abstain. The
    probabilities are softmax outputs and not calibrated; the line was set by
    measuring what it costs in coverage, not by reading 0.9 as 90 percent."""
    top = max(probs, key=probs.get)
    return top if probs[top] >= abstain_below else ABSTAIN


def is_mask_head_checkpoint(path: str) -> bool:
    """A mask-head checkpoint is a dict with a decoder; the legacy ones carry a
    single state dict plus config. Cheap enough to call before choosing a loader."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(ck, dict) and "decoder" in ck and "classifier" in ck


class MaskHeadModel:
    def __init__(self, checkpoint: str, device=None):
        import torch, torch.nn as nn
        from training.train_mask_head import build_decoder, build_encoder
        from torchvision import transforms

        self.torch, self.nn = torch, nn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.size = ck["size"]
        self.encoder_name = ck["encoder"]
        self.unfreeze = ck.get("unfreeze", 0)
        self.epoch = ck.get("epoch")
        self.enc, self.features, dim, self.grid = build_encoder(
            self.encoder_name, self.size, self.device, unfreeze=self.unfreeze)
        if ck.get("encoder_state"):
            res = self.enc.load_state_dict(ck["encoder_state"], strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
        self.enc.eval()
        self.dec = build_decoder(dim).to(self.device)
        self.dec.load_state_dict(ck["decoder"]); self.dec.eval()
        self.clf = nn.Sequential(nn.Linear(dim + 3, 256), nn.GELU(),
                                 nn.Linear(256, 3)).to(self.device)
        self.clf.load_state_dict(ck["classifier"]); self.clf.eval()
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._is_clip = self.encoder_name.startswith("clip:")

    # ------------------------------------------------------------------ input
    def describe(self) -> str:
        how = (f"last {self.unfreeze} encoder blocks fine-tuned" if self.unfreeze
               else "frozen encoder")
        return f"{self.encoder_name} @ {self.size}px, {how}, epoch {self.epoch}"

    @staticmethod
    def match_training_encoding(img: Image.Image) -> Image.Image:
        """Squash to 512x512 and round-trip through JPEG q90, as convert_opensdi
        did to every training image. Returned image is what the model 'sees'."""
        im = ImageOps.exif_transpose(img).convert("RGB").resize((512, 512), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=90, optimize=True)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    def to_tensor(self, img: Image.Image):
        from torchvision.transforms import functional as TF
        x = self.norm(TF.to_tensor(img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)))
        return x.unsqueeze(0).to(self.device)

    # -------------------------------------------------------------- forward
    def _head(self, fmap, cls):
        F = self.torch.nn.functional
        up = F.interpolate(self.dec(fmap), size=(self.size, self.size),
                           mode="bilinear", align_corners=False).squeeze(1)
        prob = self.torch.sigmoid(up)
        summary = self.torch.stack([prob.mean((1, 2)), prob.amax((1, 2)),
                                    (prob > 0.5).float().mean((1, 2))], dim=1)
        logits = self.clf(self.torch.cat([cls, summary], 1))
        return logits, prob

    def predict(self, img: Image.Image) -> Tuple[str, Dict[str, float], np.ndarray]:
        """Returns (top label, {label: prob}, edit-mask probability map HxW in [0,1])."""
        F = self.torch.nn.functional
        with self.torch.no_grad():
            fmap, cls = self.features(self.to_tensor(img))
            logits, prob = self._head(fmap, cls)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        out = {LABELS[i]: float(probs[i]) for i in range(3)}
        return max(out, key=out.get), out, prob[0].cpu().numpy()

    def gradcam(self, img: Image.Image, class_idx: int) -> np.ndarray:
        """Grad-CAM on the final token grid, for the given class logit.
        Returns a grid x grid map normalised to [0,1]."""
        torch = self.torch
        blocks = (self.enc.visual.transformer.resblocks if self._is_clip else self.enc.blocks)
        captured = {}

        def hook(_m, _i, out):
            # Start the graph here: nothing upstream requires grad, so the
            # block output has no grad_fn until we make it a leaf.
            leaf = out.detach().requires_grad_(True)
            captured["tokens"] = leaf
            return leaf

        h = blocks[-1].register_forward_hook(hook)
        try:
            with torch.enable_grad():
                fmap, cls = self.features(self.to_tensor(img))
                logits, _ = self._head(fmap, cls)
                logits[0, class_idx].backward()
        finally:
            h.remove()
        tok = captured["tokens"]; g = tok.grad
        if self._is_clip:                       # CLIP: (L+1, B, dim) -> (B, L, dim)
            tok, g = tok.permute(1, 0, 2), g.permute(1, 0, 2)
        tok, g = tok[:, 1:, :], g[:, 1:, :]     # drop CLS
        B, L, D = tok.shape
        A = tok.reshape(B, self.grid, self.grid, D).permute(0, 3, 1, 2)
        G = g.reshape(B, self.grid, self.grid, D).permute(0, 3, 1, 2)
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * A).sum(1))[0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)
