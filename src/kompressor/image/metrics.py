# MIT License
#
# Copyright (c) 2020 Joss Whittle
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
import io
import imageio
import numpy as np
import jax
import jax.numpy as jnp


def imageio_rgb_bpp(batch, *imageio_args, **imageio_kargs):
    # Compute the image encoded bpp's for a batch of RGB images using imageio and a given file format
    assert batch.ndim in [3, 4]
    assert batch.shape[-1] == 3

    # Compute metric for a single RGB image
    def bpp_fn(image):
        with io.BytesIO() as stream:
            imageio.imwrite(stream, image, *imageio_args, **imageio_kargs)
            # Normalize byte size to bits per pixel (24 bpp raw)
            return np.float32((stream.tell() * 8) / np.prod(image.shape[:-1]))

    # Return a scalar for single image input or array for batched inputs
    return bpp_fn(batch) if (batch.ndim == 3) else np.array(list(map(bpp_fn, batch)))


@partial(jax.jit, static_argnames=('k',))
def mean_within_k(batch, k):
    # Compute the percentage of pixels that fall within [-k, +k] of the target
    assert batch.ndim >= 3
    assert 0 < k

    delta = jnp.abs(jnp.float32(batch))
    delta = jnp.reshape(delta, (batch.shape[0], -1))
    return jnp.mean((delta <= k), axis=-1)


@jax.jit
def mean_run_length(batch):
    # Compute percentage of the image that is spent in constant valued run lengths
    assert batch.ndim >= 3
    delta = jnp.transpose(batch, (0, *range(3, batch.ndim), 1, 2))
    delta = jnp.reshape(delta, (batch.shape[0], -1))
    return jnp.mean(jnp.diff(delta, axis=-1) == 0, axis=-1)

