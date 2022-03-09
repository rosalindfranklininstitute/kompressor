from training.models import Model
import jax
import haiku as hk


def convolutional_encoder(padding: int):
    def net_fn(lowres):
        # Extract data dimensions
        P, C = 5, 3

        # Extract features from the raw RGB image batch using an even width kernel convolution
        features = jax.nn.swish(hk.Conv2D(300, 2, padding='VALID')(lowres))

        for _ in range(padding):
            # Widen the receptive field up to the padding value using 3x3 convolutions to pool by [-1, +1) pixels
            # with each layer
            features = jax.nn.swish(hk.Conv2D(100, 3, padding='VALID')(features))

        # Output predictions for each neighbourhood
        features = hk.Conv2D((P * C), 1)(features)
        H, W = features.shape[1:3]
        return hk.Reshape((H, W, P, C))(features)

    return net_fn


_model = Model("cnn", convolutional_encoder)
