import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).

    Replaces global average pooling. For p>1 this behaves closer to max pooling
    and preserves more signal from strong local activations.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = False):
        super().__init__()
        self.eps = eps
        # Always register p as a parameter so state_dicts are consistent; optionally freeze it.
        self.p = nn.Parameter(torch.ones(1) * p)
        if not learnable:
            self.p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        p = torch.clamp(self.p, min=1e-3, max=10.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.avg_pool2d(x, kernel_size=x.shape[-2:])
        x = x.pow(1.0 / p)
        return x


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "CBAM spatial kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        scale = self.sigmoid(self.conv(concat))
        return x * scale


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Lightweight channel + spatial attention applied to a feature map.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = _ChannelAttention(channels, reduction=reduction)
        self.spatial_att = _SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

