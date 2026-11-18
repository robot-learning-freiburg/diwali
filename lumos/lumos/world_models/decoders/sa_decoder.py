
from typing import List, Optional
import torch
import torch.nn as nn

from lumos.utils.nn_utils import flatten_batch, get_activation, NoNorm, unflatten_batch


class SADecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cnn_depth: int,
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        out_channels: List[int],
        layer_norm: bool,
        activation: str,
        mlp_layers: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.out_channels = out_channels
        self.norm = nn.LayerNorm if layer_norm else NoNorm
        self.activation = get_activation(activation)
        self.mlp_layers = mlp_layers

        self.deepest_dim = self.cnn_depth * (2 ** (len(self.kernels) + 1))

        # Build initial MLP layers
        self.initial_layers = self._make_initial_layers()

        # Build decoder layers with skip connection handling
        self.decoder_layers = self._make_decoder_layers()

        # Final layer to convert to desired output size and channels
        # From [batch, 96, 31, 31] to [batch, 2, 64, 64]
        self.final_layer = nn.ConvTranspose2d(
            in_channels=96,  # Assuming this matches the last decoder layer output
            out_channels=2,
            kernel_size=6,
            stride=2,
            padding=1,
            output_padding=0,
        )

    def _make_initial_layers(self):
        in_channels = self.deepest_dim

        if self.mlp_layers == 0:
            layers = [nn.Linear(self.in_dim, in_channels)]
        else:
            hidden_dim = in_channels
            layers = [nn.Linear(self.in_dim // 2, hidden_dim), self.norm(hidden_dim, eps=1e-3), self.activation]
            for _ in range(self.mlp_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), self.norm(hidden_dim, eps=1e-3), self.activation]

        layers.append(nn.Unflatten(-1, (self.deepest_dim, 1, 1)))
        return nn.Sequential(*layers)

    def _make_decoder_layers(self):
        out_channels_list = self.out_channels
        num_conv_layers = len(self.kernels)
        assert len(out_channels_list) == num_conv_layers, "Mismatch in kernel/stride/output channel list lengths"

        layers = nn.ModuleList()
        in_channels = self.deepest_dim

        for i in range(num_conv_layers):
            out_channels = out_channels_list[i]

            # Create transposed convolution layer
            conv_transpose = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i],
            )

            layers.append(conv_transpose)

            # For skip connections, we need to adjust input channels for next layer
            # Skip features will be concatenated, so we double the input channels
            if i < num_conv_layers - 1:
                # Skip connection will add channels from encoder
                skip_channels = self.cnn_depth * (2 ** (num_conv_layers - 1 - i))
                in_channels = out_channels + skip_channels
            else:
                in_channels = out_channels

        return layers

    def forward(self, embed: torch.Tensor, latent_action: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None):
        embed_pairs = torch.cat([embed, latent_action], dim=-1)

        x, bd = flatten_batch(embed_pairs, 1)

        # Initial MLP processing
        x = self.initial_layers(x)

        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Apply transposed convolution
            x = decoder_layer(x)

            # Add skip connection if available and not the last layer
            if skip_features is not None and i < len(self.decoder_layers):
                # Skip connections are in forward order, but we need them in reverse
                skip_idx = len(skip_features) - 1 - i
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip_feat, _ = flatten_batch(skip_features[skip_idx][:-1], 3)

                    # Ensure spatial dimensions match
                    if skip_feat.shape[-2:] == x.shape[-2:]:
                        x = torch.cat([x, skip_feat], dim=1)

            # Apply activation (except for the last layer)
            if i < len(self.decoder_layers) - 1:
                x = self.activation(x)

        # Apply final layer to get desired output dimensions
        x = self.final_layer(x)

        dcd_flow = unflatten_batch(x, bd)
        return dcd_flow
