
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from diwali.wm.model_utils.nn_utils import flatten_batch, unflatten_batch, get_activation, NoNorm

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture.

    A flexible MLP implementation that supports customizable hidden layers,
    normalization, and activation functions.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension (None for no final linear layer)
        hidden_dim: Hidden layer dimension
        hidden_layers: Number of hidden layers
        layer_norm: Whether to apply layer normalization
        activation: Activation function name
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int],
        mlp_layers: List[int],
        layer_norm: bool,
        activation: str,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp_layers = mlp_layers
        self.norm = nn.LayerNorm if layer_norm else NoNorm
        self.activation = get_activation(activation)
        self.dropout = dropout

        layers = []
        for i in range(len(self.mlp_layers)):
            input_dim = self.in_dim if i == 0 else self.mlp_layers[i-1]
            layers.extend(
                [
                    nn.Linear(input_dim, self.mlp_layers[i]),
                    self.norm(self.mlp_layers[i], eps=1e-3),
                    self.activation,
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        if self.out_dim is not None:
            if len(self.mlp_layers) == 0:
                layers.append(nn.Linear(self.in_dim, self.out_dim))
            else:
                layers.append(nn.Linear(self.mlp_layers[-1], self.out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = torch.cat(xs, dim=-1)
        x, batch_dims = flatten_batch(x)
        output = self.model(x)
        output = unflatten_batch(output, batch_dims)
        return output

    def get_action(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.forward(*xs), None


class GaussianDecoder(nn.Module):
    """Gaussian decoder for continuous action spaces.

    This decoder produces mean and log standard deviation for a Gaussian distribution
    over actions, with optional sampling capabilities. The log standard deviation
    is bounded to ensure numerical stability.

    Args:
        in_dim: Dimension of input embeddings
        hidden_size: Hidden layer size for MLPs
        layers: Number of hidden layers
        layer_norm: Whether to use layer normalization
        activation: Activation function name
        out_features: Output dimension (action space size)
        log_std_min: Minimum log standard deviation
        log_std_max: Maximum log standard deviation
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: List[int],
        layer_norm: bool,
        activation: str,
        out_features: int,
        log_std_min: float,
        log_std_max: float,
        mean_layers: List[int] = [],
        dropout: float = 0.0,
    ):
        super(GaussianDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_features = out_features
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.dropout = dropout

        # Feature extractor
        self.feature_extractor = MLP(self.in_dim, None, hidden_size, layer_norm, activation, dropout=self.dropout)

        # Separate heads for mean and log standard deviation
        self.mean_head = MLP(hidden_size[-1], out_features, mean_layers, layer_norm, activation, dropout=self.dropout)
        self.log_std_head = MLP(hidden_size[-1], out_features, [], layer_norm, activation, dropout=self.dropout)

    def _sample_actions(self, log_stds: Tensor, means: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample actions from a Normal distribution and apply tanh squashing.

        Args:
            log_stds: Log standard deviations
            means: Mean values

        Returns:
            Tuple of (sampled_actions, entropy)
        """
        action_stds = torch.exp(log_stds)
        normal_dist = torch.distributions.Normal(means, action_stds)

        # Sample using reparameterization trick
        raw_actions = normal_dist.rsample()

        # Apply tanh squashing to bound actions to [-1, 1]
        actions = torch.tanh(raw_actions)

        # Calculate entropy
        entropy = normal_dist.entropy()

        return actions, entropy

    def forward(self, embed: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass to compute mean and log standard deviation.

        Args:
            embed: Input embeddings

        Returns:
            Tuple of (log_stds, means) for the Gaussian distribution
        """
        features = self.feature_extractor(embed)

        means = self.mean_head(features)
        log_stds = self.log_std_head(features)

        # Bound log standard deviations using tanh activation
        log_stds = torch.tanh(log_stds)
        log_stds = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_stds + 1)

        return log_stds, means

    def get_action(self, *xs: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample actions from the Gaussian distribution.

        Args:
            embed: Input embeddings

        Returns:
            Tuple of (sampled_actions, entropy)
        """
        embed = torch.cat(xs, dim=-1) if len(xs) > 1 else xs[0]
        log_stds, means = self(embed)
        actions, entropy = self._sample_actions(log_stds, means)
        return actions, entropy