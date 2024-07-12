import flax.linen as nn


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network https://arxiv.org/pdf/1609.05158v2.pdf
    """

    name: str = "ESPCN"
    features: int = 64
    input_conv_kernel_size: int = 5
    encoding_conv_kernel_size: int = 3
    output_kernel_size: int = 2
    patches: int = 5
    channels: int = 1

    @nn.compact
    def __call__(self, low_resolution):
        features = nn.activation.relu(
            nn.Conv(
                features=self.features,
                kernel_size=(self.input_conv_kernel_size, self.input_conv_kernel_size),
            )(low_resolution)
        )
        features = nn.activation.relu(
            nn.Conv(
                features=self.features,
                kernel_size=(
                    self.encoding_conv_kernel_size,
                    self.encoding_conv_kernel_size,
                ),
            )(features)
        )
        features = nn.activation.relu(
            nn.Conv(
                features=self.features // 2,
                kernel_size=(
                    self.encoding_conv_kernel_size,
                    self.encoding_conv_kernel_size,
                ),
                padding="VALID",
            )(features)
        )
        features = nn.Conv(
            features=self.patches * self.channels,
            kernel_size=(self.output_kernel_size, self.output_kernel_size),
            padding="VALID",
        )(features)
        batch, height, width = features.shape[0:3]
        return features.reshape(batch, height, width, self.patches, self.channels)
