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

import numpy as np
import jax
import jax.numpy as jnp

from ..utils import \
    yield_chunks, validate_padding


########################################################################################################################
# Public API functions used for building compression models
########################################################################################################################


@jax.jit
def targets_from_highres(highres):
    # Slice out each value of the pluses
    lmap  = highres[:,  1::2,  1::2, :-1:2]
    rmap  = highres[:,  1::2,  1::2,  2::2]
    umap  = highres[:,  1::2, :-1:2,  1::2]
    dmap  = highres[:,  1::2,  2::2,  1::2]
    fmap  = highres[:, :-1:2,  1::2,  1::2]
    bmap  = highres[:,  2::2,  1::2,  1::2]
    cmap  = highres[:,  1::2,  1::2,  1::2]

    # Slice out the four corners of the central plane on the z-axis
    z0map = highres[:,  1::2, :-1:2, :-1:2]
    z1map = highres[:,  1::2, :-1:2,  2::2]
    z2map = highres[:,  1::2,  2::2,  2::2]
    z3map = highres[:,  1::2,  2::2, :-1:2]

    # Slice out the four corners of the central plane on the y-axis
    y0map = highres[:, :-1:2,  1::2, :-1:2]
    y1map = highres[:, :-1:2,  1::2,  2::2]
    y2map = highres[:,  2::2,  1::2,  2::2]
    y3map = highres[:,  2::2,  1::2, :-1:2]

    # Slice out the four corners of the central plane on the x-axis
    x0map = highres[:, :-1:2, :-1:2,  1::2]
    x1map = highres[:, :-1:2,  2::2,  1::2]
    x2map = highres[:,  2::2,  2::2,  1::2]
    x3map = highres[:,  2::2, :-1:2,  1::2]

    # Stack the vectors LRUDFBC order with dim [B,D,H,W,19,...]
    targets = jnp.stack([
        lmap, rmap, umap, dmap, fmap, bmap, cmap,
        z0map, z1map, z2map, z3map,
        y0map, y1map, y2map, y3map,
        x0map, x1map, x2map, x3map
    ], axis=4)

    return targets


@jax.jit
def lowres_from_highres(highres):
    # Downsample by skip sampling
    return highres[:, ::2, ::2, ::2]


@jax.jit
def maps_from_predictions(predictions):
    # Given a tensor of predictions for each neighborhood, decode the [B,D,H,W,19,...] predictions into 7 maps
    # Averages predictions when there are two for a pixel from adjacent neighborhoods.

    # Determine the size of the highres volume given the size of the predictions
    dtype = predictions.dtype
    (batch_size, pd, ph, pw), channels = predictions.shape[:4], predictions.shape[5:]

    # Map for containing aggregated predictions of the left and right of the pluses
    lrmap = jnp.zeros((batch_size, pd, ph, pw + 1, *channels), dtype=jnp.float32)
    lrmap = lrmap.at[:, :, :, :-1].add(predictions[:, :, :, :, 0])  # Left predictions
    lrmap = lrmap.at[:, :, :,  1:].add(predictions[:, :, :, :, 1])  # Right predictions
    # Normalize LCR map to account for left and right value double predictions
    lrmap = lrmap.at[:, :, :, 1:-1].mul(0.5)
    lrmap = lrmap.astype(dtype)

    # Map for containing aggregated predictions of the up and down of the pluses
    udmap = jnp.zeros((batch_size, pd, ph + 1, pw, *channels), dtype=jnp.float32)
    udmap = udmap.at[:, :, :-1, :].add(predictions[:, :, :, :, 2])  # Up predictions
    udmap = udmap.at[:, :, 1:,  :].add(predictions[:, :, :, :, 3])  # Down predictions
    # Normalize UD map to account for up and down value double predictions
    udmap = udmap.at[:, :, 1:-1, :].mul(0.5)
    udmap = udmap.astype(dtype)

    # Map for containing aggregated predictions of the front and back of the pluses
    fbmap = jnp.zeros((batch_size, pd + 1, ph, pw, *channels), dtype=jnp.float32)
    fbmap = fbmap.at[:, :-1, :, :].add(predictions[:, :, :, :, 4])  # Front predictions
    fbmap = fbmap.at[:, 1:,  :, :].add(predictions[:, :, :, :, 5])  # Back predictions
    # Normalize FB map to account for front and back value double predictions
    fbmap = fbmap.at[:, 1:-1, :, :].mul(0.5)
    fbmap = fbmap.astype(dtype)

    # Map for containing aggregated predictions of the centre of the pluses
    cmap = predictions[:, :, :, :, 6]

    # Map for containing aggregated predictions of the corners of the central z-axis plane
    zmap = jnp.zeros((batch_size, pd, ph + 1, pw + 1, *channels), dtype=jnp.float32)
    zmap = zmap.at[:, :,  :-1,  :-1].add(predictions[:, :, :, :, 7])  # Top-Left predictions
    zmap = zmap.at[:, :,  :-1,   1:].add(predictions[:, :, :, :, 8])  # Top-Right predictions
    zmap = zmap.at[:, :,   1:,   1:].add(predictions[:, :, :, :, 9])  # Bottom-Right predictions
    zmap = zmap.at[:, :,   1:,  :-1].add(predictions[:, :, :, :, 10])  # Bottom-Left predictions
    # Normalize Z map to account for front and back value double and quad predictions
    zmap = zmap.at[:, :, 1:-1, 1:-1].mul(0.25)
    zmap = zmap.at[:, :, 1:-1, ::pw].mul(0.5)
    zmap = zmap.at[:, :, ::ph, 1:-1].mul(0.5)
    zmap = zmap.astype(dtype)

    # Map for containing aggregated predictions of the corners of the central y-axis plane
    ymap = jnp.zeros((batch_size, pd + 1, ph, pw + 1, *channels), dtype=jnp.float32)
    ymap = ymap.at[:,  :-1, :,  :-1].add(predictions[:, :, :, :, 11])  # Top-Left predictions
    ymap = ymap.at[:,  :-1, :,   1:].add(predictions[:, :, :, :, 12])  # Top-Right predictions
    ymap = ymap.at[:,   1:, :,   1:].add(predictions[:, :, :, :, 13])  # Bottom-Right predictions
    ymap = ymap.at[:,   1:, :,  :-1].add(predictions[:, :, :, :, 14])  # Bottom-Left predictions
    # Normalize Y map to account for front and back value double and quad predictions
    ymap = ymap.at[:, 1:-1, :, 1:-1].mul(0.25)
    ymap = ymap.at[:, 1:-1, :, ::pw].mul(0.5)
    ymap = ymap.at[:, ::pd, :, 1:-1].mul(0.5)
    ymap = ymap.astype(dtype)

    # Map for containing aggregated predictions of the corners of the central x-axis plane
    xmap = jnp.zeros((batch_size, pd + 1, ph + 1, pw, *channels), dtype=jnp.float32)
    xmap = xmap.at[:,  :-1,  :-1, :].add(predictions[:, :, :, :, 15])  # Top-Left predictions
    xmap = xmap.at[:,  :-1,   1:, :].add(predictions[:, :, :, :, 16])  # Top-Right predictions
    xmap = xmap.at[:,   1:,   1:, :].add(predictions[:, :, :, :, 17])  # Bottom-Right predictions
    xmap = xmap.at[:,   1:,  :-1, :].add(predictions[:, :, :, :, 18])  # Bottom-Left predictions
    # Normalize X map to account for front and back value double and quad predictions
    xmap = xmap.at[:, 1:-1, 1:-1, :].mul(0.25)
    xmap = xmap.at[:, 1:-1, ::ph, :].mul(0.5)
    xmap = xmap.at[:, ::pd, 1:-1, :].mul(0.5)
    xmap = xmap.astype(dtype)

    return lrmap, udmap, fbmap, cmap, zmap, ymap, xmap


@jax.jit
def maps_from_highres(highres):
    # Given a highres volume extract the four maps of known values from the pluses
    lrmap = highres[:, 1::2, 1::2,  ::2]
    udmap = highres[:, 1::2,  ::2, 1::2]
    fbmap = highres[:,  ::2, 1::2, 1::2]
    cmap  = highres[:, 1::2, 1::2, 1::2]

    # Extract the central axis corners
    zmap  = highres[:, 1::2,  ::2,  ::2]
    ymap  = highres[:,  ::2, 1::2,  ::2]
    xmap  = highres[:,  ::2,  ::2, 1::2]

    return lrmap, udmap, fbmap, cmap, zmap, ymap, xmap


@jax.jit
def highres_from_lowres_and_maps(lowres, maps):
    # Merge together a lowres volume and the seven maps of known values representing the missing voxels
    lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps

    # Determine the size of the highres volume given the size of the lowres
    dtype = lowres.dtype
    (batch_size, ld, lh, lw), channels = lowres.shape[:4], lowres.shape[4:]
    hd, hh, hw = ((ld - 1) * 2) + 1, ((lh - 1) * 2) + 1, ((lw - 1) * 2) + 1

    # Volume for containing the merged output
    highres = jnp.zeros((batch_size, hd, hh, hw, *channels), dtype=dtype)
    highres = highres.at[:,  ::2,  ::2,  ::2].set(lowres)  # Apply the values from the lowres volume
    highres = highres.at[:, 1::2, 1::2,  ::2].set(lrmap)   # Apply the values from the left and right of the pluses
    highres = highres.at[:, 1::2,  ::2, 1::2].set(udmap)   # Apply the values from the up and down of the pluses
    highres = highres.at[:,  ::2, 1::2, 1::2].set(fbmap)   # Apply the values from the front and back of the pluses
    highres = highres.at[:, 1::2, 1::2, 1::2].set(cmap)    # Apply the values from the centre of the pluses
    highres = highres.at[:, 1::2,  ::2,  ::2].set(zmap)    # Apply the values from the central z-axis plane corners
    highres = highres.at[:,  ::2, 1::2,  ::2].set(ymap)    # Apply the values from the central y-axis plane corners
    highres = highres.at[:,  ::2,  ::2, 1::2].set(xmap)    # Apply the values from the central x-axis plane corners

    return highres


# TODO make padding default to 0 when JAX merges being able to have static named args in jit functions
@jax.partial(jax.jit, static_argnums=1)
def features_from_lowres(lowres, padding):
    # Extract the features around each 2x2x2 neighborhood (assumes the lowres is already padded)
    pd, ph, pw = (lowres.shape[1] - (padding * 2)) - 1, \
                 (lowres.shape[2] - (padding * 2)) - 1, \
                 (lowres.shape[3] - (padding * 2)) - 1

    # Extract the N neighbors and stack them together [B, D, H, W, N, ...] where N = ((padding * 2) + 1) ^ 3
    return jnp.stack([lowres[:, z:(z+pd), y:(y+ph), x:(x+pw)]
                      for z in range((padding*2)+2)
                      for y in range((padding*2)+2)
                      for x in range((padding*2)+2)], axis=4)


@jax.partial(jax.jit, static_argnums=1)
def pad_neighborhood(lowres, padding):
    # Pad only the 3 spatial dimensions
    spatial_padding = ((padding, padding),) * 3
    data_padding    = ((0, 0),) * len(lowres.shape[4:])
    return jnp.pad(lowres, ((0, 0), *spatial_padding, *data_padding), mode='symmetric')


########################################################################################################################
# Padding and trim functions for handling inputs with even dimension sizes
########################################################################################################################


@jax.jit
def pad_highres(highres):
    # Determine the size of the input and the padding to apply
    hd, hh, hw = highres.shape[1:4]
    pd, ph, pw = (hd + 1) % 2, (hh + 1) % 2, (hw + 1) % 2

    # Pad highres using reflect to match lowres padded with symmetric
    data_padding = ((0, 0),) * len(highres.shape[4:])
    padded_highres = jnp.pad(highres, ((0, 0), (0, pd), (0, ph), (0, pw), *data_padding), mode='reflect')

    # Return the padded highres and the padding values
    return padded_highres, (pd, ph, pw)


def pad_lowres(lowres, padding):
    # Pad lowres using symmetric to match lowres padded with reflect
    pd, ph, pw = padding
    data_padding = ((0, 0),) * len(lowres.shape[4:])
    return jnp.pad(lowres, ((0, 0), (0, pd), (0, ph), (0, pw), *data_padding), mode='symmetric')


def pad_map(inputs, padding):
    # Pad map using reflect to match lowres padded with symmetric
    pd, ph, pw = padding
    data_padding = ((0, 0),) * len(inputs.shape[4:])
    return jnp.pad(inputs, ((0, 0), (0, pd), (0, ph), (0, pw), *data_padding), mode='symmetric')


def pad_maps(maps, padding):
    # Unpack the maps and the padding
    lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps
    pd, ph, pw = padding
    # Pad maps based on even padding
    return pad_map(lrmap, (0, 0, pw)), pad_map(udmap, (0, ph, 0)), pad_map(fbmap, (pd, 0, 0)), cmap, \
           pad_map(zmap, (0, ph, pw)), pad_map(ymap, (pd, 0, pw)), pad_map(xmap, (pd, ph, 0))


def trim(inputs, padding):
    # Determine the size of the input and the padding to remove
    d, h, w = inputs.shape[1:4]
    pd, ph, pw = padding
    return inputs[:, :(d-pd), :(h-ph), :(w-pw)]


def trim_maps(maps, padding):
    # Unpack the maps and the padding
    lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps
    pd, ph, pw = padding
    # Trim maps based on even padding
    return trim(lrmap, (0, 0, pw)), trim(udmap, (0, ph, 0)), trim(fbmap, (pd, 0, 0)), cmap, \
           trim(zmap, (0, ph, pw)), trim(ymap, (pd, 0, pw)), trim(xmap, (pd, ph, 0))


########################################################################################################################
# Validation functions
########################################################################################################################


def validate_highres(highres):
    # Assert the input is large enough
    assert highres.ndim >= 5
    assert np.prod(highres.shape) > 0
    hd, hh, hw = highres.shape[1:4]
    assert hd > 2 and (hd % 2) != 0
    assert hh > 2 and (hh % 2) != 0
    assert hw > 2 and (hw % 2) != 0
    return hd, hh, hw


def validate_lowres(lowres):
    # Assert the input is large enough
    assert lowres.ndim >= 5
    assert np.prod(lowres.shape) > 0
    ld, lh, lw = lowres.shape[1:4]
    assert ld >= 2
    assert lh >= 2
    assert lw >= 2
    return ld, lh, lw


def validate_chunk(chunk):
    # Assert valid chunk size
    if isinstance(chunk, int):
        assert chunk > 3
        cd, ch, cw = (chunk,) * 3
    elif isinstance(chunk, tuple):
        cd, ch, cw = chunk
        assert cd > 3
        assert ch > 3
        assert cw > 3
    else:
        raise AssertionError('chunk must be int or tuple(int, int, int)')
    return cd, ch, cw
