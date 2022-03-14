from training.compressors.base import BaseCompressor
from training.compressors import Compressor
import haiku as hk
import optax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

class HaikuCompressor(BaseCompressor):

    def __init__(self, encode_fn, decode_fn, padding, model_fn, predictions_fn, seed=None):
        super().__init__(encode_fn=encode_fn, decode_fn=decode_fn, padding=padding)
        self.model_fn = model_fn
        self.__predictions_fn = predictions_fn
        self.params = self.avg_params = None

        self.model = hk.without_apply_rng(hk.transform(model_fn(self.padding)))

    def _predictions_fn(self):
        return self.__predictions_fn(self.model, self.avg_params)

    def init(self, ds_train, seed=None):

        ds_train = ds_train.as_numpy_iterator()

        if self.avg_params is None:
            self.avg_params = self.model.init(jax.random.PRNGKey(seed or np.random.randint(1e6)),
                                              next(ds_train)['lowres'])

        return self

    def save_model(self, file_name):
        np.save(file_name, self.avg_params, allow_pickle=True)

    def load_model(self, file_name):
        np_params = np.load(file_name, allow_pickle=True)

        def jax_encoder(obj):
            if isinstance(obj, np.ndarray):
                jnp_array = jnp.asarray(obj)
                return jnp_array
            if isinstance(obj, dict):
                for key in obj:
                    obj[key] = jax_encoder(obj[key])
            return obj
        self.avg_params = jax_encoder(np_params.all())

    def fit(self, ds_train, start_step=0, end_step=1, callbacks=None):

        callbacks = callbacks or list()

        assert 0 <= start_step < end_step

        ds_train = ds_train.as_numpy_iterator()

        params = self.avg_params
        opt = optax.adam(1e-5)
        opt_state = opt.init(params)

        @jax.jit
        def l2(params):
            return 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params))

        @jax.jit
        def loss(params, batch):
            predictions = self.model.apply(params, batch['lowres'])
            prediction_loss = jnp.mean(optax.l2_loss(predictions, batch['targets']))
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
        for step in trange(start_step, end_step, desc='steps'):

            train_batch = next(ds_train)

            for callback in callbacks:
                callback.on_step_start(step=step, compressor=self)

            # Update params
            loss, params, opt_state = update(params, opt_state, train_batch)
            self.avg_params = ema_update(params, self.avg_params)

            for callback in callbacks:
                callback.on_step_end(step=step, loss=loss, compressor=self)

        # Return self to enable chaining
        return self


_compressor = Compressor("haiku", HaikuCompressor)
