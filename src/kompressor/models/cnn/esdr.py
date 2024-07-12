from typing import Sequence

import flax.linen as nn

from ..utils import Sequential


class ResidualBlock(nn.Module):
    features: int
    kernel_size: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return x + Sequential(
            [
                nn.Conv(features=self.features, kernel_size=self.kernel_size),
                nn.activation.relu,
                nn.Conv(features=self.features, kernel_size=self.kernel_size),
            ]
        )(x)


class EDSR(nn.Module):
    """Enhanced Deep Residual Networks for Single Image Super-Resolution
    https://arxiv.org/pdf/1707.02921v1.pdf"""

    features: int = 256
    feature_kernel_size: int = 3
    num_blocks: int = 2
    output_kernel_size: int = 2
    patches: int = 5
    channels: int = 1

    @nn.compact
    def __call__(self, features):
        features = x_res = nn.Conv(
            features=self.features,
            kernel_size=(self.feature_kernel_size, self.feature_kernel_size),
            padding="VALID",
        )(features)
        res_blocks = [
            ResidualBlock(
                features=self.features,
                kernel_size=(self.feature_kernel_size, self.feature_kernel_size),
            )
            for _ in range(self.num_blocks)
        ]
        x_res = Sequential(res_blocks)(x_res)
        features = features + nn.Conv(
            features=self.features,
            kernel_size=(self.feature_kernel_size, self.feature_kernel_size),
        )(x_res)

        features = nn.Conv(
            features=self.patches * self.channels,
            kernel_size=(self.output_kernel_size, self.output_kernel_size),
            padding="VALID",
        )(features)
        batch, height, width = features.shape[0:3]
        return features.reshape(batch, height, width, self.patches, self.channels)
