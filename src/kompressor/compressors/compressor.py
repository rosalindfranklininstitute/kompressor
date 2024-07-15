import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm

from ..image import decode, decode_chunks, encode, encode_chunks


class Kompressor:
    def __init__(self, encode_fn, decode_fn, padding):
        self.encode_fn, self.decode_fn = encode_fn, decode_fn
        self.padding = padding

    def _predictions_fn(self):
        raise NotImplementedError()

    def encode(self, highres, levels=1, chunk=None, progress_fn=None, debug=False):
        assert levels > 0

        predictions_fn = self._predictions_fn()

        maps = list()
        for level in range(levels):
            if chunk is None:
                lowres, maps_dims = encode(
                    predictions_fn, self.encode_fn, highres, padding=self.padding
                )
            else:
                lowres, maps_dims = encode_chunks(
                    predictions_fn,
                    self.encode_fn,
                    highres,
                    padding=self.padding,
                    chunk=chunk,
                    progress_fn=progress_fn,
                )

            if debug:
                maps.append((lowres, maps_dims, highres))
            else:
                maps.append(maps_dims)

            highres = lowres

        return lowres, maps

    def decode(self, lowres, maps, chunk=None, progress_fn=None, debug=False):
        assert len(maps) > 0

        predictions_fn = self._predictions_fn()

        for maps_dims in reversed(maps):
            if debug:
                _, maps_dims, _ = maps_dims

            if chunk is None:
                highres = decode(
                    predictions_fn,
                    self.decode_fn,
                    lowres,
                    maps_dims,
                    padding=self.padding,
                )
            else:
                highres = decode_chunks(
                    predictions_fn,
                    self.decode_fn,
                    lowres,
                    maps_dims,
                    padding=self.padding,
                    chunk=chunk,
                    progress_fn=progress_fn,
                )

            lowres = highres

        return highres


class FlaxKompressor(Kompressor):
    def __init__(self, encode_fn, decode_fn, padding, model_fn, predictions_fn):
        super().__init__(encode_fn=encode_fn, decode_fn=decode_fn, padding=padding)
        self.model_fn = model_fn
        self.__predictions_fn = predictions_fn
        self.params = self.avg_params = None
        self.local_devices = jax.local_devices()
        self.opt_state = None

    def _predictions_fn(self):
        return self.__predictions_fn(self.model_fn, self.avg_params)

    def init(self, ds_train, seed=None):
        """Initialize the model"""
        ds_train = iter(ds_train)
        if self.avg_params is None:
            self.avg_params = self.model_fn.init(
                jax.random.PRNGKey(seed or np.random.randint(1e6)),
                next(ds_train)["lowres"],
            )

        return self

    def fit(
        self,
        ds_train,
        start_step=0,
        end_step=1,
        learning_rate=1e-5,
        checkpoint_manager=None,
        callbacks=None,
    ):
        """
        Train the model
        Args:
            ds_train: Train Dataset
            start_step: Start Epoch
            end_step: End Epoch
            learning_rate: Rate to learn
            checkpoint_manager: Manager of Saving model weights
            callbacks: Callbacks for logging during training

        Returns:
            self
        """
        callbacks = callbacks or list()

        assert 0 <= start_step < end_step

        params = self.avg_params
        opt = optax.adam(learning_rate)
        if self.opt_state:
            opt_state = self.opt_state
        else:
            opt_state = opt.init(params)

        @jax.jit
        def l2(params):
            return 0.5 * sum(
                jnp.sum(jnp.square(param))
                for param in jax.tree_util.tree_leaves(params)
            )

        @jax.jit
        def loss(params, batch):
            predictions = self.model_fn.apply(params, batch["lowres"])
            prediction_loss = jnp.mean(optax.l2_loss(predictions, batch["targets"]))
            return prediction_loss + (1e-6 * l2(params))

        @jax.jit
        def update(params, opt_state, batch):
            value, grads = jax.value_and_grad(loss)(params, batch)
            updates, opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return value, new_params, opt_state

        @jax.jit
        def ema_update(params, avg_params):
            return optax.incremental_update(params, avg_params, step_size=0.001)

        # Train/eval loop
        train_len = len(ds_train)

        for epoch in tqdm(range(start_step, end_step), desc="Epochs"):
            average_loss = []
            for iters, train_batch in enumerate(ds_train):
                for callback in callbacks:
                    callback.on_step_start(
                        step=(epoch * train_len) + iters, compressor=self
                    )

                # Update params
                loss, params, opt_state = update(params, opt_state, train_batch)
                self.avg_params = ema_update(params, self.avg_params)
                for callback in callbacks:
                    callback.on_step_end(
                        step=(epoch * train_len) + iters, loss=loss, compressor=self
                    )
            if checkpoint_manager:
                checkpoint = {
                    "model": {"params": self.avg_params, "opt_state": opt_state}
                }
                checkpoint_manager.save(
                    args=ocp.args.StandardSave(checkpoint), step=epoch
                )
        # Return self to enable chaining
        return self
