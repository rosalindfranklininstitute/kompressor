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
import jax
import jax.numpy as jnp


@jax.jit
def mean_squared_error(pred, gt):
    batch_size = gt.shape[0]
    delta = jnp.square(jnp.float32(gt) - jnp.float32(pred))
    return jnp.mean(jnp.reshape(delta, (batch_size, -1)), axis=-1)


@jax.jit
def mean_abs_error(pred, gt):
    batch_size = gt.shape[0]
    delta = jnp.abs(jnp.float32(gt) - jnp.float32(pred))
    return jnp.mean(jnp.reshape(delta, (batch_size, -1)), axis=-1)


@partial(jax.jit, static_argnames=('eps',))
def mean_charbonnier_error(pred, gt, eps):
    batch_size = gt.shape[0]
    delta = jnp.sqrt(jnp.square(jnp.float32(gt) - jnp.float32(pred)) + jnp.square(eps))
    return jnp.mean(jnp.reshape(delta, (batch_size, -1)), axis=-1)
