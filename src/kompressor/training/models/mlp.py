import jax
import haiku as hk
import kompressor as kom
from kompressor.training.models import Model

def mlp_encoder(padding, convolutional=False):
    def net_fn(lowres):

        # Extract data dimensions
        P, C = 5, 3

        # Fix kernel size to 1x1 to lock receptive field from growing
        K = 1

        # Standard LeNet-300-100 MLP network, parallelized over 2D feature maps

        # Extract the features for each neighborhood
        if convolutional:
            # Extract features from the raw RGB image batch using an even width kernel convolution
            features = jax.nn.relu(hk.Conv2D(300, ((padding + 1) * 2), padding='VALID')(lowres))
            H, W = features.shape[1:3]
        else:
            # Extract features from the raw RGB image batch using slicing and stacking followed by a 1x1 kernel convolution
            features = kom.image.features_from_lowres(lowres, padding)
            H, W = features.shape[1:3]
            features = jax.nn.relu(hk.Conv2D(300, K)(hk.Reshape((H, W, -1))(features)))

        # features.shape == (B, H, W, N*C)
        # where N = ((padding * 2) + 1) ^ 2 is the number of surrounding neighbours included for this padding value

        mlp = hk.Sequential([
            # Apply the rest of the network as 1x1 kernel convolutions
            hk.Conv2D(100, K), jax.nn.relu,

            # Output predictions for each neighbourhood
            hk.Conv2D((P * C), K),
            hk.Reshape((H, W, P, C)),
            # reshape (B, H, W, P*C) -> (B, H, W, P, C)
            # where P = 5, the number of values that need to be predicted for each neighbourhood
        ])

        # features(B, H, W, N, C) -> predictions(B, H, W, P, C)
        return mlp(features)

    return net_fn


_model = Model("mlp", mlp_encoder)
