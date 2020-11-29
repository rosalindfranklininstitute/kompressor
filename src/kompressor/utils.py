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
def encode_values_raw(pred, gt):
    return jnp.int32(gt) - jnp.int32(pred)


@jax.jit
def decode_values_raw(pred, encoded):
    return jnp.int32(pred) + jnp.int32(encoded)


@jax.jit
def encode_values_uint8(pred, gt):
    return jnp.uint8(((jnp.int32(gt) - jnp.int32(pred)) + 256) % 256)


@jax.jit
def decode_values_uint8(pred, encoded):
    return jnp.uint8(((jnp.int32(pred) + jnp.int32(encoded)) + 256) % 256)


@jax.jit
def encode_values_uint16(pred, gt):
    return jnp.uint16(((jnp.int32(gt) - jnp.int32(pred)) + 65536) % 65536)


@jax.jit
def decode_values_uint16(pred, encoded):
    return jnp.uint16(((jnp.int32(pred) + jnp.int32(encoded)) + 65536) % 65536)


@jax.jit
def encode_categorical(pred, gt):

    # Output shape and dtype determined from reference values
    shape = gt.shape
    dtype = gt.dtype

    # Determine the descending order indexing for the logits at each spatial location and channel
    logit_ranks = jnp.argsort(pred).astype(dtype)[..., ::-1]

    # Flatten logit ranks from [B, ...SPATIAL..., C, L] to [-1, L]
    flat_logit_ranks = jnp.reshape(logit_ranks, (-1, logit_ranks.shape[-1]))
    # Flatten gt from [B, ...SPATIAL..., C] to [-1,]
    flat_gt = jnp.reshape(gt, (-1,))

    # Perform argwhere for a single location and channel
    def argwhere(logit_ranks, gt):
        return jnp.argmax(logit_ranks == gt).astype(dtype)

    # Distribute the argwhere over all spatial locations and channels
    flat_encoded = jax.vmap(argwhere)(flat_logit_ranks, flat_gt)

    # Reshape the encoded values back into the shape of the gt tensor
    encoded = jnp.reshape(flat_encoded, shape)

    return encoded


@jax.jit
def decode_categorical(pred, encoded):

    # Output shape and dtype determined from encoded values
    shape = encoded.shape
    dtype = encoded.dtype

    # Determine the descending order indexing for the logits at each spatial location and channel
    logit_ranks = jnp.argsort(pred).astype(dtype)[..., ::-1]

    # Flatten logit ranks from [B, ...SPATIAL..., C, L] to [-1, L]
    flat_logit_ranks = jnp.reshape(logit_ranks, (-1, logit_ranks.shape[-1]))
    # Flatten encoded from [B, ...SPATIAL..., C] to [-1,]
    flat_encoded = jnp.reshape(encoded, (-1,))

    # Perform indexing for a single location and channel
    def index(logit_ranks, encoded):
        return logit_ranks[encoded]

    # Distribute the indexing over all spatial locations and channels
    flat_decoded = jax.vmap(index)(flat_logit_ranks, flat_encoded)

    # Reshape the decoded values back into the shape of the encoded tensor
    decoded = jnp.reshape(flat_decoded, shape)

    return decoded


def yield_chunks(max_value, chunk):
    # Assert max value is positive
    assert max_value > 0

    # Assert chunk size is valid
    assert chunk > 3

    if chunk >= max_value:
        # If we can process in a single chunk than yield that chunk with no padding
        yield (0, max_value), (0, 0)

    else:
        # Yield a set of constant sized chunks along one axis including boundary conditions
        for idx in range(0, max_value, (chunk-3)):
            # Far edge of chunk, clamped against the right edge
            i1 = min(max_value, (idx+(chunk-2)))

            # Does this chunk border on the rightmost edge
            last = (i1 == max_value)

            # Near edge of chunk, constraining that every chunk must be of constant size
            i0 = max(0, ((i1-(chunk-2)) if last else idx))

            # Does this chunk border on the left most edge
            first = (i0 == 0)

            # Calculate constant width padding
            p0 = 0 if first else (2 if last  else 1)
            p1 = 0 if last  else (2 if first else 1)

            # Assert singleton chunk was handled by the other if-branch
            assert not (first and last)

            # Assert that the total padding was length 2 to ensure constant sized chunks
            assert (p0 + p1) == 2

            # Yield the chunk and the dynamic padding
            yield (i0, i1), (p0, p1)

            # Prevent duplicating last chunk
            if last:
                break
