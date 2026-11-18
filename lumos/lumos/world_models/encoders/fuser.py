from typing import List

import torch
from torch import nn

from lumos.utils.nn_utils import flatten_batch, get_activation, unflatten_batch


class Fuser(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        activation: str,
    ):
        super(Fuser, self).__init__()

        self.activation = get_activation(activation)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim, eps=1e-3), self.activation
        )

    def forward(self, y_s, y_g):
        y_s, bd = flatten_batch(y_s, 1)
        y_g, _ = flatten_batch(y_g, 1)
        y_combined = torch.cat([y_s, y_g], dim=1)
        y = self.fuse(y_combined)
        y = unflatten_batch(y, bd)
        return y
