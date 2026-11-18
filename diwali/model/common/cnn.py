"""
CNN image encoder backbone for vision-conditioned policies.

Drop-in alternative to VitEncoder: same attribute interface
(num_patch, patch_repr_dim, repr_dim) and forward output shape (B, N, C).
"""

from dataclasses import dataclass, field
from typing import List, Sequence

import einops
import torch
from torch import nn


@dataclass
class CnnEncoderConfig:
    """Default channels yield ~1.5M params for 6x64x64 input."""

    channels: Sequence[int] = field(default_factory=lambda: [64, 128, 256, 320, 128])
    kernel_size: int = 3
    out_channels: int = 128  # patch feature dim; last channel overrides if set via channels[-1]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # GroupNorm is batch-size independent (small batch / eval friendly)
        num_groups = min(32, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CnnEncoder(nn.Module):
    """
    5-layer CNN backbone producing spatial patch tokens like VitEncoder.

    Default channel schedule [64, 128, 256, 320, 128] with three stride-2
    downsamples (64 -> 32 -> 16 -> 8) targets ~1.5M parameters for 6-channel
    64x64 inputs.
    """

    def __init__(
        self,
        obs_shape: List[int],
        cfg: CnnEncoderConfig,
        num_channel: int = 3,
        img_h: int = 96,
        img_w: int = 96,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.img_h = img_h
        self.img_w = img_w

        channels = list(cfg.channels)
        assert len(channels) >= 3, "CnnEncoder expects at least 3 conv layers"

        layers: List[nn.Module] = []
        in_ch = num_channel
        # First three layers downsample; remaining layers keep spatial size
        for i, out_ch in enumerate(channels):
            stride = 2 if i < 3 else 1
            layers.append(ConvBlock(in_ch, out_ch, stride=stride, kernel_size=cfg.kernel_size))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

        # Infer spatial size after the stem (no padding tricks beyond ConvBlock)
        with torch.no_grad():
            dummy = torch.zeros(1, num_channel, img_h, img_w)
            feat = self.net(dummy)
        _, c, h, w = feat.shape

        self.patch_repr_dim = c
        self.num_patch = h * w
        self.repr_dim = self.patch_repr_dim * self.num_patch
        self._feat_h = h
        self._feat_w = w

    def forward(self, obs: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        # Match VitEncoder preprocessing
        obs = obs / 255.0 - 0.5
        feats = self.net(obs)  # (B, C, H, W)
        feats = einops.rearrange(feats, "b c h w -> b (h w) c")
        if flatten:
            feats = feats.flatten(1, 2)
        return feats


if __name__ == "__main__":
    obs_shape = [6, 64, 64]
    enc = CnnEncoder(
        obs_shape,
        CnnEncoderConfig(),
        num_channel=obs_shape[0],
        img_h=obs_shape[1],
        img_w=obs_shape[2],
    )
    n_params = sum(p.numel() for p in enc.parameters())
    x = torch.rand(2, *obs_shape) * 255
    y = enc(x)
    print(enc)
    print(f"params: {n_params:,}")
    print(f"output: {tuple(y.shape)}")
    print(f"num_patch={enc.num_patch}, patch_repr_dim={enc.patch_repr_dim}, repr_dim={enc.repr_dim}")
