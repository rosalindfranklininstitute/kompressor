import jax
import jax.numpy as jnp
import kompressor as kom

UINT16_MAX_VALUE: float = 65536.0


def regression_predictor(model, params):
    # Regression predictor function
    @jax.jit
    def predictions_fn(lowres):
        # lowres.shape == (B, H, W, C)

        # Get predictions for neighbourhoods
        predictions = model.apply(params, jnp.float32(lowres) / UINT16_MAX_VALUE)
        # Convert predictions to uint16
        predictions = jnp.floor(jnp.clip(predictions, 0, 1) * UINT16_MAX_VALUE).astype(
            lowres.dtype
        )
        # predictions.shape == (B, H, W, P, C) where P = 5, the number of maps that
        # need to be predicted for each channel.

        # Extract the maps from the predictions
        maps = kom.image.maps_from_predictions(predictions)
        # lrmap, udmap, cmap = maps
        return maps

    return predictions_fn
