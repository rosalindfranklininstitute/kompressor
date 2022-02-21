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


import jax
import jax.numpy as jnp


@jax.jit
def encode_values(pred, gt):
    # Encode predictions in [0, +256) to deltas in [-256, +256) to encoded values in [0, +256)
    # Leads to peaks at 0 and +255
    return jnp.uint8(((jnp.int32(gt) - jnp.int32(pred)) + 256) % 256)

@jax.jit
def decode_values(pred, encoded):
    # Decode twin peaks distribution
    return jnp.uint8(((jnp.int32(pred) + jnp.int32(encoded)) + 256) % 256)


@jax.jit
def encode_transform_interleaved(input):
    # Take twin peaks at 0 and +255 and shift them to be a single peak centered around 0 in [-128, +128)
    # Encode positive values a strided by 1, and negatives values interleaved between them in [0, +256)
    encoded = ((jnp.int32(input) + 128) % 256) - 128
    encoded = jnp.where((encoded >= 0), (encoded * 2), ((encoded + 1) * -2) + 1)
    return jnp.uint8(encoded)


@jax.jit
def decode_transform_interleaved(encoded):
    # Decode interleaved in [0, +256) to [-128, +128) to [0, 256+)
    # Positive values strided by 1, and negatives values interleaved between them
    # Leads to peaks at 0 and +255
    encoded = jnp.int32(encoded)
    encoded = jnp.where((encoded % 2 == 0), (encoded // 2), -(((encoded - 1) // 2) + 1))
    return jnp.uint8((encoded + 256) % 256)


@jax.jit
def encode_transform_centre(input):
    # Take twin peaks at 0 and +255 and shift them to be a single peak centered around +128 in [0, +256)
    return jnp.uint8((jnp.int32(input) + 128) % 256)


@jax.jit
def decode_transform_centre(encoded):
    # Decode single peak at +128 into twin peaks at 0 and +255 in [0, +256)
    return jnp.uint8((jnp.int32(encoded) + 128) % 256)
