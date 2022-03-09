import jax
import jax.numpy as jnp
import kompressor as kom
from training.predictors import Predictor


def regression_predictor(model, params):

    # Regression predictor function applies convolutional MLP network
    @jax.jit
    def predictions_fn(lowres):

        # Get predictions for neighbourhoods, first layer of network is a convolutional feature extractor
        predictions = model.apply(params, jnp.float32(lowres) / 256.)
        # Convert predictions to uint8
        predictions = jnp.floor(jnp.clip(predictions, 0, 1) * 256.).astype(lowres.dtype)
        # predictions.shape == (B, H, W, P, C)
        # where P = 5, the number of values that need to be predicted for each neighbourhood

        # Extract the maps from the predictions
        maps = kom.image.maps_from_predictions(predictions)
        return maps

    return predictions_fn


_predictor = Predictor("reg", regression_predictor)
