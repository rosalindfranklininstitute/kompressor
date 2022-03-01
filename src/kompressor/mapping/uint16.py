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
    return jnp.uint16(((jnp.int32(gt) - jnp.int32(pred)) + 65536) % 65536)


@jax.jit
def decode_values(pred, encoded):
    return jnp.uint16(((jnp.int32(pred) + jnp.int32(encoded)) + 65536) % 65536)


@jax.jit
def encode_transform_interleaved(input):
    encoded = ((jnp.int32(input) + 32768) % 65536) - 32768
    encoded = jnp.where((encoded >= 0), (encoded * 2), ((encoded + 1) * -2) + 1)
    return jnp.uint16(encoded)


@jax.jit
def decode_transform_interleaved(encoded):
    encoded = jnp.int32(encoded)
    encoded = jnp.where((encoded % 2 == 0), (encoded // 2), -(((encoded - 1) // 2) + 1))
    return jnp.uint16((encoded + 65536) % 65536)


@jax.jit
def encode_transform_centre(input):
    return jnp.uint16((jnp.int32(input) + 32768) % 65536)


@jax.jit
def decode_transform_centre(encoded):
    return jnp.uint16((jnp.int32(encoded) + 32768) % 65536)
