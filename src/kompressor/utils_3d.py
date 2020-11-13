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

from enum import IntEnum

# For each 2x2x2 8 voxel neighborhood there are 19 missing voxels to predict,
# stored in the following consistent order
Neighbors = IntEnum('Neighbors', [
    'L', 'R', 'U', 'D', 'F', 'B', 'C',
    'Z0', 'Z1', 'Z2', 'Z3',
    'Y0', 'Y1', 'Y2', 'Y3',
    'X0', 'X1', 'X2', 'X3'
], start=0)


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


def lowres_from_highres(highres):
    # Downsample by skip sampling
    return highres[:, ::2, ::2, ::2]


def maps_from_predictions(predictions):
    # Given a tensor of predictions for each neighborhood, decode the [B,D,H,W,19,...] predictions into 7 maps
    # Averages predictions when there are two for a pixel from adjacent neighborhoods.

    # Determine the size of the highres image given the size of the predictions
    dtype = predictions.dtype
    (batch_size, pd, ph, pw), channels = predictions.shape[:4], predictions.shape[5:]

    # Map for containing aggregated predictions of the left and right of the pluses
    lrmap = jnp.zeros((batch_size, pd, ph, pw + 1, *channels), dtype=dtype)
    lrmap = lrmap.at[:, :, :, :-1].add(predictions[:, :, :, :, Neighbors.L])  # Left predictions
    lrmap = lrmap.at[:, :, :,  1:].add(predictions[:, :, :, :, Neighbors.R])  # Right predictions
    # Normalize LCR map to account for left and right value double predictions
    lrmap = lrmap.at[:, :, :, 1:-1].mul(0.5)

    # Map for containing aggregated predictions of the up and down of the pluses
    udmap = jnp.zeros((batch_size, pd, ph + 1, pw, *channels), dtype=dtype)
    udmap = udmap.at[:, :, :-1, :].add(predictions[:, :, :, :, Neighbors.U])  # Up predictions
    udmap = udmap.at[:, :, 1:,  :].add(predictions[:, :, :, :, Neighbors.D])  # Down predictions
    # Normalize UD map to account for up and down value double predictions
    udmap = udmap.at[:, :, 1:-1, :].mul(0.5)

    # Map for containing aggregated predictions of the front and back of the pluses
    fbmap = jnp.zeros((batch_size, pd + 1, ph, pw, *channels), dtype=dtype)
    fbmap = fbmap.at[:, :-1, :, :].add(predictions[:, :, :, :, Neighbors.F])  # Front predictions
    fbmap = fbmap.at[:, 1:,  :, :].add(predictions[:, :, :, :, Neighbors.B])  # Back predictions
    # Normalize FB map to account for front and back value double predictions
    fbmap = fbmap.at[:, 1:-1, :, :].mul(0.5)

    # Map for containing aggregated predictions of the centre of the pluses
    cmap = predictions[:, :, :, :, Neighbors.C]

    # Map for containing aggregated predictions of the corners of the central z-axis plane
    zmap = jnp.zeros((batch_size, pd, ph + 1, pw + 1, *channels), dtype=dtype)
    zmap = zmap.at[:, :,  :-1,  :-1].add(predictions[:, :, :, :, Neighbors.Z0])  # Top-Left predictions
    zmap = zmap.at[:, :,  :-1,   1:].add(predictions[:, :, :, :, Neighbors.Z1])  # Top-Right predictions
    zmap = zmap.at[:, :,   1:,   1:].add(predictions[:, :, :, :, Neighbors.Z2])  # Bottom-Right predictions
    zmap = zmap.at[:, :,   1:,  :-1].add(predictions[:, :, :, :, Neighbors.Z3])  # Bottom-Left predictions
    # Normalize Z map to account for front and back value double and quad predictions
    zmap = zmap.at[:, :, 1:-1, 1:-1].mul(0.25)
    zmap = zmap.at[:, :, 1:-1, ::pw].mul(0.5)
    zmap = zmap.at[:, :, ::ph, 1:-1].mul(0.5)

    # Map for containing aggregated predictions of the corners of the central y-axis plane
    ymap = jnp.zeros((batch_size, pd + 1, ph, pw + 1, *channels), dtype=dtype)
    ymap = ymap.at[:,  :-1, :,  :-1].add(predictions[:, :, :, :, Neighbors.Y0])  # Top-Left predictions
    ymap = ymap.at[:,  :-1, :,   1:].add(predictions[:, :, :, :, Neighbors.Y1])  # Top-Right predictions
    ymap = ymap.at[:,   1:, :,   1:].add(predictions[:, :, :, :, Neighbors.Y2])  # Bottom-Right predictions
    ymap = ymap.at[:,   1:, :,  :-1].add(predictions[:, :, :, :, Neighbors.Y3])  # Bottom-Left predictions
    # Normalize Y map to account for front and back value double and quad predictions
    ymap = ymap.at[:, 1:-1, :, 1:-1].mul(0.25)
    ymap = ymap.at[:, 1:-1, :, ::pw].mul(0.5)
    ymap = ymap.at[:, ::pd, :, 1:-1].mul(0.5)

    # Map for containing aggregated predictions of the corners of the central x-axis plane
    xmap = jnp.zeros((batch_size, pd + 1, ph + 1, pw, *channels), dtype=dtype)
    xmap = xmap.at[:,  :-1,  :-1, :].add(predictions[:, :, :, :, Neighbors.X0])  # Top-Left predictions
    xmap = xmap.at[:,  :-1,   1:, :].add(predictions[:, :, :, :, Neighbors.X1])  # Top-Right predictions
    xmap = xmap.at[:,   1:,   1:, :].add(predictions[:, :, :, :, Neighbors.X2])  # Bottom-Right predictions
    xmap = xmap.at[:,   1:,  :-1, :].add(predictions[:, :, :, :, Neighbors.X3])  # Bottom-Left predictions
    # Normalize X map to account for front and back value double and quad predictions
    xmap = xmap.at[:, 1:-1, 1:-1, :].mul(0.25)
    xmap = xmap.at[:, 1:-1, ::ph, :].mul(0.5)
    xmap = xmap.at[:, ::pd, 1:-1, :].mul(0.5)

    return lrmap, udmap, fbmap, cmap, zmap, ymap, xmap


def maps_from_highres(highres):
    # Given a highres image extract the four maps of known values from the pluses
    lrmap = highres[:, 1::2, 1::2,  ::2]
    udmap = highres[:, 1::2,  ::2, 1::2]
    fbmap = highres[:,  ::2, 1::2, 1::2]
    cmap  = highres[:, 1::2, 1::2, 1::2]

    # Extract the central axis corners
    zmap  = highres[:, 1::2,  ::2,  ::2]
    ymap  = highres[:,  ::2, 1::2,  ::2]
    xmap  = highres[:,  ::2,  ::2, 1::2]

    return lrmap, udmap, fbmap, cmap, zmap, ymap, xmap


def highres_from_lowres_and_maps(lowres, lrmap, udmap, fbmap, cmap, zmap, ymap, xmap):
    # Merge together a lowres image and the four maps of known values representing the missing pluses

    # Determine the size of the highres image given the size of the lowres
    dtype = lowres.dtype
    (batch_size, ld, lh, lw), channels = lowres.shape[:4], lowres.shape[4:]
    hd, hh, hw = ((ld - 1) * 2) + 1, ((lh - 1) * 2) + 1, ((lw - 1) * 2) + 1

    # Image for containing the merged output
    highres = jnp.zeros((batch_size, hd, hh, hw, *channels), dtype=dtype)
    highres = highres.at[:,  ::2,  ::2,  ::2].set(lowres)  # Apply the values from the lowres image
    highres = highres.at[:, 1::2, 1::2,  ::2].set(lrmap)   # Apply the values from the left and right of the pluses
    highres = highres.at[:, 1::2,  ::2, 1::2].set(udmap)   # Apply the values from the up and down of the pluses
    highres = highres.at[:,  ::2, 1::2, 1::2].set(fbmap)   # Apply the values from the front and back of the pluses
    highres = highres.at[:, 1::2, 1::2, 1::2].set(cmap)    # Apply the values from the centre of the pluses
    highres = highres.at[:, 1::2,  ::2,  ::2].set(zmap)    # Apply the values from the central z-axis plane corners
    highres = highres.at[:,  ::2, 1::2,  ::2].set(ymap)    # Apply the values from the central y-axis plane corners
    highres = highres.at[:,  ::2,  ::2, 1::2].set(xmap)    # Apply the values from the central x-axis plane corners

    return highres
