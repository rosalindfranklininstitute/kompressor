import flax.linen as nn


class SRCNN(nn.Module):
    """
    A simple convolutional neural network for Super Resolution.

    Model based on Super-Resolution Convolutional Neural Network https://arxiv.org/pdf/1501.00092v3.pdf
    """

    name: str = "SRCNN"
    encoding_features: int = 300
    neighbourhood_features: int = 100
    encoding_kernel_size: int = 3
    neighbourhood: int = 1

    output_kernel_size: int = 2
    patches: int = 5
    channels: int = 1

    @nn.compact
    def __call__(self, low_resolution):
        features = nn.activation.relu(
            nn.Conv(
                features=300,
                kernel_size=(self.encoding_kernel_size, self.encoding_kernel_size),
                padding="VALID",
            )(low_resolution)
        )
        for _ in range(self.neighbourhood):
            features = nn.activation.relu(
                nn.Conv(
                    features=self.neighbourhood_features,
                    kernel_size=(1, 1),
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
