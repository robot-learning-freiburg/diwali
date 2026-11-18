
from typing import List, Tuple
from torch import nn
from lumos.utils.nn_utils import flatten_batch, get_activation, unflatten_batch


class SAEncoder(nn.Module):
    def __init__(
        self,
        cnn_depth: int,
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        activation: str,
    ):
        super(SAEncoder, self).__init__()
        self.in_dim = 3
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.activation = get_activation(activation)

        self.conv_layers = self._make_encoder_layers()

    def _make_encoder_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_dim

        for i in range(len(self.kernels)):
            out_channels = self.cnn_depth * (2**i)
            conv_layer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i],
            )
            layers.append(conv_layer)
            in_channels = out_channels

        return layers

    def forward(self, x) -> Tuple[any, List[any]]:
        x, bd = flatten_batch(x, 3)

        # Store intermediate features for skip connections
        skip_features = []

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            # Store feature before activation (some architectures prefer this)
            skip_features.append(x)
            x = self.activation(x)

        # Flatten the final output
        x = nn.Flatten()(x)

        # Unflatten the final output and skip features
        final_output = unflatten_batch(x, bd)
        skip_features_unflattened = [unflatten_batch(feat, bd) for feat in skip_features]

        return final_output, skip_features_unflattened
