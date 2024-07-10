import os
import time
from collections import defaultdict

import jax.numpy as jnp
import wandb

from ..image import maps_from_highres
from ..image.metrics import mean_run_length


class Callback:
    def on_step_start(self, *args, **kargs):
        pass

    def on_step_end(self, *args, **kargs):
        pass


class MetricsCallback(Callback):
    def __init__(self, chunk, ds_train, ds_test=None, log_freq=1, levels=1):
        super().__init__()

        assert log_freq > 0
        self.log_freq = log_freq

        assert levels >= 0
        self.levels = levels

        self.chunk = chunk

        self.ds_train = ds_train
        self.ds_test = ds_test

    def on_step_end(self, step, loss, compressor, *args, **kargs):
        # Record Loss at every step
        wandb.log({"train/loss": loss}, step=step)
        if not ((step > 0) and (step % self.log_freq == 0)):
            return

        def log(dataset, mode):
            start = time.time()
            summaries = defaultdict(list)
            for highres in dataset:
                lowres, level_encoded_maps = compressor.encode(
                    highres, levels=self.levels, chunk=self.chunk, debug=True
                )

                for level, (
                    level_highres,
                    (level_encoded_maps, _),
                    level_lowres,
                ) in enumerate(level_encoded_maps):
                    level_highres_maps = maps_from_highres(level_highres)

                    writer_path = os.path.join(
                        f"{mode}/level={level}/",
                        f'lowres={"x".join(map(str, level_lowres.shape[1:]))}',
                    )

                    for label in level_encoded_maps.keys() & level_highres_maps.keys():
                        summaries[f"{writer_path}/{label}/run_length"].extend(
                            list(mean_run_length(level_encoded_maps[label]).flatten())
                        )

            for key in summaries.keys():
                wandb.log(
                    {key: (jnp.array(summaries[key]).flatten().mean())}, step=step
                )
            end = time.time()
            return end - start

        if self.ds_train is not None:
            train_eval_time = log(self.ds_train, "train")
            wandb.log({"train/eval_time": train_eval_time}, step=step)

        if self.ds_test is not None:
            test_eval_time = log(self.ds_test, "test")
            wandb.log({"test/eval_time": test_eval_time}, step=step)
