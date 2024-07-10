import flax.linen as nn


class VDSR(nn.Module):
    """Very Deep Super Resolution https://arxiv.org/pdf/1511.04587.pdf"""

    name: str = "VDSR"
    num_layers: int = 10
    features: int = 64
    kernel_size: int = 3
    patches: int = 5
    channels: int = 1

    @nn.compact
    def __call__(self, low_resolution):
        feature_resolution = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="VALID",
        )(low_resolution)
        for i in range(self.num_layers - 1):
            feature_resolution = nn.activation.relu(
                nn.Conv(
                    features=self.features,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    padding="SAME",
                )(feature_resolution)
            )
        feature_resolution = nn.Conv(
            features=self.patches * self.channels,
            kernel_size=(2, 2),
            padding="VALID",
        )(feature_resolution)
        low_resolution = nn.Conv(
            self.patches * self.channels, kernel_size=(3, 3), padding="VALID"
        )(low_resolution)
        low_resolution = nn.Conv(
            self.patches * self.channels, kernel_size=(2, 2), padding="VALID"
        )(low_resolution)
        x = low_resolution + feature_resolution
        batch, height, width = x.shape[0:3]
        return x.reshape(batch, height, width, self.patches, self.channels)
