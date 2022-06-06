import functools
from kompressor.training.compressors.base import BaseCompressor
from kompressor.training.compressors import Compressor
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
        self.opt = optax.adam(1e-5)
        self.local_devices = jax.local_devices()

    def _predictions_fn(self):
        return self.__predictions_fn(self.model, self.avg_params)

    def double_buffer_dataset(self, ds):
        batch = None
        for next_batch in ds:
            assert next_batch is not None
            next_batch = jax.tree_map(lambda x: jax.device_put_sharded(list(x.numpy()), self.local_devices), next_batch)
            if batch is not None:
                yield batch
            batch = next_batch
        if batch is not None:
            yield batch

    @functools.partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="devices")
    def init(self, ds_train, seed=None):
        params = self.model.init(jax.random.PRNGKey(seed or np.random.randint(1e6)), ds_train['lowres'])
        opt_state = self.opt.init(params)
        return params, opt_state

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

    @functools.partial(jax.jit, static_argnums=(0,))
    def l2(self, params):
        return 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params))

    @functools.partial(jax.jit, static_argnums=(0,))
    def ema_update(self, params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, batch):
        predictions = self.model.apply(params, batch['lowres'])
        prediction_loss = jnp.mean(optax.l2_loss(predictions, batch['targets']))
        return prediction_loss + (1e-6 * self.l2(params))

    @functools.partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="devices")
    def update(self, params, opt_state, batch):
        value, grads = jax.value_and_grad(self.loss)(params, batch)
        grads = jax.lax.pmean(grads, axis_name="devices")
        value = jax.lax.pmean(value, axis_name="devices")
        updates, opt_state = self.opt.update(grads, opt_state)
        updates = jax.lax.pmean(updates, axis_name="devices")
        new_params = optax.apply_updates(params, updates)
        return value, new_params, opt_state

    def fit(self, ds_train, start_step=0, end_step=1, steps_per_epoch=1, callbacks=None):
        callbacks = callbacks or list()
        assert 0 <= start_step < end_step

        ds_train = self.double_buffer_dataset(ds_train)
        if self.avg_params is None:

            self.avg_params, opt_state = self.init(next(ds_train))
            sharded_params = self.avg_params
        else:
            opt_state = self.opt.init(self.avg_params)
            sharded_params = jax.device_put_replicated(self.avg_params, self.local_devices)
            opt_state = jax.device_put_replicated(opt_state, self.local_devices)

        # Train/eval loop
        for step in trange(start_step, end_step, desc='epochs'):

            for callback in callbacks:
                callback.on_step_start(step=step, compressor=self)

            for _ in trange(0, steps_per_epoch, desc="Steps"):
                train_batch = next(ds_train)
                # Update params
                loss, sharded_params, opt_state = self.update(sharded_params, opt_state, train_batch)

                self.avg_params = self.ema_update(sharded_params, self.avg_params)

            for callback in callbacks:
                self.avg_params = jax.tree_map(lambda x: x[0], self.avg_params)
                self.params = jax.tree_map(lambda x: x[0], sharded_params)
                loss = jax.tree_map(lambda x: x[0], loss)
                callback.on_step_end(step=step, loss=loss, compressor=self)

        # Return self to enable chaining
        return self


_compressor = Compressor("haiku", HaikuCompressor)
