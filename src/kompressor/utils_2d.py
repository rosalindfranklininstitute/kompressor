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


import jax.numpy as jnp

from enum import Enum, auto


class Neighbors(Enum):
    # Central plus
    L = auto()
    R = auto()
    U = auto()
    D = auto()
    C = auto()


def targets_from_highres(highres):
    # Slice out each value of the pluses
    lmap = highres[:,  1::2, :-1:2]
    rmap = highres[:,  1::2,  2::2]
    umap = highres[:, :-1:2,  1::2]
    dmap = highres[:,  2::2,  1::2]
    cmap = highres[:,  1::2,  1::2]

    # Stack the vectors LRUDC order with dim [B,H,W,5,...]
    targets = jnp.stack([lmap, rmap, umap, dmap, cmap], axis=-2)

    return targets


def lowres_from_highres(highres):
    # Downsample by skip sampling
    return highres[:, ::2, ::2]


def maps_from_predictions(predictions):
    # Given a tensor of 5 predictions for each neighborhood, decode the [B,H,W,5,...] predictions into three maps.
    # Averages predictions when there are two for a pixel from adjacent neighborhoods.

    # Determine the size of the highres image given the size of the predictions
    dtype = predictions.dtype
    (batch_size, ph, pw), channels = predictions.shape[:3], predictions.shape[4:]

    # Map for containing aggregated predictions of the left and right of the pluses
    lrmap = jnp.zeros((batch_size, ph, pw + 1, *channels), dtype=dtype)
    lrmap = lrmap.at[:, :, :-1].add(predictions[:, :, :, Neighbors.L])  # Left predictions
    lrmap = lrmap.at[:, :,  1:].add(predictions[:, :, :, Neighbors.R])  # Right predictions
    # Normalize LCR map to account for left and right value double predictions
    lrmap = lrmap.at[:, :, 1:-1].mul(0.5)

    # Map for containing aggregated predictions of the up and down of the pluses
    udmap = jnp.zeros((batch_size, ph + 1, pw, *channels), dtype=dtype)
    udmap = udmap.at[:, :-1, :].add(predictions[:, :, :, Neighbors.U])  # Up predictions
    udmap = udmap.at[:, 1:,  :].add(predictions[:, :, :, Neighbors.D])  # Down predictions
    # Normalize UD map to account for up and down value double predictions
    udmap = udmap.at[:, 1:-1, :].mul(0.5)

    # Map for containing aggregated predictions of the centre of the pluses
    cmap = predictions[:, :, :, Neighbors.C]

    return lrmap, udmap, cmap


def maps_from_highres(highres):
    # Given a highres image extract the three maps of known values from the pluses
    lrmap = highres[:, 1::2,  ::2]
    udmap = highres[:,  ::2, 1::2]
    cmap  = highres[:, 1::2, 1::2]

    return lrmap, udmap, cmap


def highres_from_lowres_and_maps(lowres, lrmap, udmap, cmap):
    # Merge together a lowres image and the three maps of known values representing the missing pluses

    # Determine the size of the highres image given the size of the lowres
    dtype = lowres.dtype
    (batch_size, lh, lw), channels = lowres.shape[:3], lowres.shape[3:]
    hh, hw = ((lh - 1) * 2) + 1, ((lw - 1) * 2) + 1

    # Image for containing the merged output
    highres = jnp.zeros((batch_size, hh, hw, *channels), dtype=dtype)
    highres = highres.at[:,  ::2,  ::2].set(lowres)  # Apply the values from the lowres image
    highres = highres.at[:, 1::2,  ::2].set(lrmap)   # Apply the values from the left and right of the pluses
    highres = highres.at[:,  ::2, 1::2].set(udmap)   # Apply the values from the up and down of the pluses
    highres = highres.at[:, 1::2, 1::2].set(cmap)    # Apply the values from the centre of the pluses

    return highres
