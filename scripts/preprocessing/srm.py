"""
SRM (Steganalysis Rich Model) filter layer for manipulation detection.

AI editing tools leave microscopic high-frequency residuals at edit boundaries —
pixel-level noise inconsistencies, blending seam artefacts, and GAN spectral
fingerprints — that are invisible to the human eye but captured by high-pass filters.

This module prepends three fixed (non-trainable) high-pass kernels to the network
input, producing 3 residual channels that are concatenated with the original 3 RGB
channels → 6-channel input to the first conv layer.

The three kernels used:
  1. Laplacian  — isotropic 2nd-order derivative; highlights all edges/discontinuities
  2. Horizontal — 1st-order horizontal gradient; captures left-right pixel jumps
  3. Vertical   — 1st-order vertical gradient; captures top-bottom pixel jumps

Each kernel is applied depthwise to the grayscale (mean of RGB), giving one residual
map per kernel.  This keeps the channel count manageable (3 extra) while covering
different frequency orientations.

Usage:
    srm = SRMLayer()
    x_6ch = srm(x_3ch)   # [B, 3, H, W] → [B, 6, H, W]

To plug into ResNet18:
    model.conv1 = adapt_conv1_for_srm(model.conv1)
    # Then wrap: model = SRMResNet(srm_layer, model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fixed SRM kernels ─────────────────────────────────────────────────────────
# Shape: [1, 1, H, W] — applied depthwise to grayscale

# Kernel 1: Discrete Laplacian (5×5)
# Zero-sum, captures second-order intensity discontinuities in all directions.
# Edit boundaries break the local smoothness of a real photograph → elevated response.
_LAPLACIAN_5x5 = torch.tensor([[
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0],
]], dtype=torch.float32).unsqueeze(0) / 8.0   # [1, 1, 5, 5]

# Kernel 2: Horizontal first-order difference (3×3 embedded in 5×5)
# Detects left-right pixel value jumps — common at horizontal splice boundaries.
_HORIZONTAL_5x5 = torch.tensor([[
    [0,  0,  0,  0, 0],
    [0,  0,  0,  0, 0],
    [0, -1,  2, -1, 0],
    [0,  0,  0,  0, 0],
    [0,  0,  0,  0, 0],
]], dtype=torch.float32).unsqueeze(0) / 2.0   # [1, 1, 5, 5]

# Kernel 3: Vertical first-order difference (3×3 embedded in 5×5)
# Detects top-bottom pixel value jumps — common at vertical splice boundaries.
_VERTICAL_5x5 = torch.tensor([[
    [0,  0,  0,  0, 0],
    [0,  0, -1,  0, 0],
    [0,  0,  2,  0, 0],
    [0,  0, -1,  0, 0],
    [0,  0,  0,  0, 0],
]], dtype=torch.float32).unsqueeze(0) / 2.0   # [1, 1, 5, 5]

# Stack into [3, 1, 5, 5] for a single depthwise-style conv call
_SRM_KERNELS = torch.cat([_LAPLACIAN_5x5, _HORIZONTAL_5x5, _VERTICAL_5x5], dim=0)


class SRMLayer(nn.Module):
    """
    Fixed high-pass residual extraction layer.

    Input:   [B, 3, H, W]  (RGB, already normalised)
    Output:  [B, 6, H, W]  (original RGB concatenated with 3 residual maps)

    The kernels are registered as a non-trainable buffer so they move to the correct
    device automatically with model.to(device) and are saved/restored in state_dict.
    """

    def __init__(self, clamp_range: float = 2.0):
        """
        Args:
            clamp_range: Residual values are clamped to [-clamp_range, clamp_range]
                         to prevent large outlier activations from dominating.
                         Typical SRM implementations use a range of 2 or 3.
        """
        super().__init__()
        # Register as buffer — device-portable, included in state_dict, not trained
        self.register_buffer("kernels", _SRM_KERNELS.clone())   # [3, 1, 5, 5]
        self.clamp_range = clamp_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        # Step 1: Convert to grayscale (mean across RGB channels)
        gray = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]

        # Step 2: Apply all 3 kernels at once.
        # Treat the 3 kernels as 3 output channels over a 1-channel input.
        # kernels shape [3, 1, 5, 5] → standard conv(1 in, 3 out, k=5, p=2)
        residuals = F.conv2d(gray, self.kernels, padding=2)  # [B, 3, H, W]

        # Step 3: Clamp to suppress extreme noise outliers
        residuals = residuals.clamp(-self.clamp_range, self.clamp_range)

        # Step 4: Concatenate residuals with original RGB → 6 channels
        return torch.cat([x, residuals], dim=1)   # [B, 6, H, W]


def adapt_conv1_for_srm(conv1: nn.Conv2d) -> nn.Conv2d:
    """
    Replace a 3-channel Conv2d (ResNet's conv1) with a 6-channel version, preserving
    the pretrained weights for the original RGB channels.

    The 3 new residual channels are initialised to 0.1× the pretrained weights so
    they start by contributing a small correction rather than a random large signal.
    This lets the model warm-start from ImageNet features and gradually learn how
    much to trust the residual channels.

    Args:
        conv1: The existing nn.Conv2d(3, 64, ...) layer from a pretrained model.

    Returns:
        A new nn.Conv2d(6, 64, ...) with the original weights preserved and
        the residual-channel weights initialised to 0.1× the RGB weights.
    """
    out_ch, _, kH, kW = conv1.weight.shape   # [64, 3, 7, 7]
    new_conv = nn.Conv2d(
        6, out_ch,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = conv1.weight          # pretrained RGB weights
        new_conv.weight[:, 3:] = conv1.weight * 0.1   # small init for residual channels
        if conv1.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    return new_conv
