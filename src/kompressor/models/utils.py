from typing import Sequence, Callable

import flax.linen as nn


class Sequential(nn.Module):
    layers: Sequence[Callable]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
