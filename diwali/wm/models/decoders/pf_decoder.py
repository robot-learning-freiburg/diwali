
from typing import List, Optional
import torch
import torch.nn as nn

from diwali.wm.model_utils.nn_utils import flatten_batch, get_activation, NoNorm, unflatten_batch


class PFDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        latent_action_dim: int,
        layer_norm: bool,
        activation: str,
        hidden_dims: List[int],
        num_points: int,
        input_layer_norm: bool = False,
    ):
        super().__init__()
        self.in_dim = embed_dim + latent_action_dim
        self.norm = nn.LayerNorm if layer_norm else NoNorm
        self.activation = get_activation(activation)
        self.hidden_dims = hidden_dims
        self.num_points = num_points

        self.embed_ln = nn.LayerNorm(embed_dim) if input_layer_norm else nn.Identity()
        self.lat_ln = nn.LayerNorm(latent_action_dim) if input_layer_norm else nn.Identity()

        layers = []
        in_dim = self.in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self.norm(h, eps=1e-3))
            layers.append(self.activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, num_points * 2))
        self.head = nn.Sequential(*layers)

    def forward(self, embed: torch.Tensor, latent_action: torch.Tensor):
        embed_pairs = torch.cat([self.embed_ln(embed), self.lat_ln(latent_action)], dim=-1)

        x, bd = flatten_batch(embed_pairs, 1)
        x = self.head(x)
        x = x.view(x.shape[0], self.num_points, 2)

        return unflatten_batch(x, bd)
