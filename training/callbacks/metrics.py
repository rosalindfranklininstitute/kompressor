from training.callbacks.base import Callback
from collections import defaultdict
import os
import time
import jax
import tensorflow as tf
import jax.numpy as jnp
import kompressor as kom


class MetricsCallback(Callback):

    def __init__(self, log_dir, chunk, ds_train, ds_test=None, log_freq=100, levels=1):
        super().__init__()

        self.log_dir = log_dir

        assert log_freq > 0
        self.log_freq = log_freq

        assert levels > 0
        self.levels = levels

        self.chunk = chunk

        self.ds_train = ds_train
        self.ds_test = ds_test

        self.writers = dict()
        self.writers['train'] = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))
        self.writers['test'] = tf.summary.create_file_writer(os.path.join(self.log_dir, 'test'))

    def on_step_end(self, step, loss, compressor, *args, **kargs):

        if not ((step > 0) and (step % self.log_freq == 0)):
            return

        def log(dataset, mode):

            start = time.time()

            summaries = defaultdict(list)

            for highres in dataset:

                lowres, level_encoded_maps = compressor.encode(highres, levels=self.levels, chunk=self.chunk,
                                                               debug=True)

                for level, (level_highres, (level_encoded_maps, _), level_lowres) in enumerate(level_encoded_maps):

                    level_highres_maps = kom.image.maps_from_highres(level_highres)

                    writer_path = os.path.join(self.log_dir, f'level={level}',
                                               f'lowres={"x".join(map(str, level_lowres.shape[1:]))}', mode)

                    for label in level_encoded_maps.keys() & level_highres_maps.keys():
                        centred_encoded_map = jnp.float32(
                            kom.mapping.uint8.encode_transform_centre(level_encoded_maps[label])) - 128.

                        summaries[(writer_path, f'{label} | total variation')].append(
                            kom.image.losses.mean_total_variation(centred_encoded_map))
                        summaries[(writer_path, f'{label} | run length')].append(
                            kom.image.metrics.mean_run_length(level_encoded_maps[label]))

                        for k in [1, 8]:
                            summaries[(writer_path, f'{label} | within k={k}')].append(
                                kom.image.metrics.mean_within_k(centred_encoded_map, k))

                        highres_bpp = kom.image.metrics.imageio_rgb_bpp(level_highres_maps[label], format='png')
                        encoded_bpp = kom.image.metrics.imageio_rgb_bpp(level_encoded_maps[label], format='png')
                        summaries[(writer_path, f'{label} | png bpp')].append(encoded_bpp)
                        summaries[(writer_path, f'{label} | png ratio')].append(1. - (encoded_bpp / highres_bpp))

            for (writer_path, label), trace in summaries.items():
                if writer_path not in self.writers:
                    self.writers[writer_path] = tf.summary.create_file_writer(writer_path)

                with self.writers[writer_path].as_default():
                    trace = jnp.concatenate(trace, axis=0)
                    tf.summary.scalar(label, jnp.mean(trace), step=step)
                    tf.summary.histogram(f'{label} | hist', trace, step=step)

            end = time.time()
            return (end - start)

        @jax.jit
        def l2(params):
            return 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params))

        with self.writers['train'].as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('l2', l2(compressor.params), step=step)

        with self.writers['test'].as_default():
            tf.summary.scalar('l2', l2(compressor.avg_params), step=step)

        if self.ds_train is not None:
            train_eval_time = log(self.ds_train.as_numpy_iterator(), 'train')
            with self.writers['train'].as_default():
                tf.summary.scalar('eval time', train_eval_time, step=step)

        if self.ds_test is not None:
            test_eval_time = log(self.ds_test.as_numpy_iterator(), 'test')
            with self.writers['test'].as_default():
                tf.summary.scalar('eval time', test_eval_time, step=step)