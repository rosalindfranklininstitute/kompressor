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

from itertools import product

import jax
import jax.numpy as jnp


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


def encode(predictions_fn, encode_fn, highres, padding=0):
    # Assert valid padding
    assert padding >= 0

    # Assert the input is large enough
    hd, hh, hw = highres.shape[1:4]
    assert hd > 2 and (hd % 2) != 0
    assert hh > 2 and (hh % 2) != 0
    assert hw > 2 and (hw % 2) != 0

    # Extract the lowres volume from the highres volume
    lowres = lowres_from_highres(highres)

    # Extract the plus values from the highres volume
    gt_maps = maps_from_highres(highres)

    # Extract the predicted values from the lowres volume
    spatial_padding = [(padding, padding)] * 3
    data_padding    = ((0, 0),) * len(lowres.shape[4:])
    pred_maps       = predictions_fn(jnp.pad(lowres, ((0, 0), *spatial_padding, *data_padding), mode='symmetric'))

    # Compare the predictions to the true values for the pluses
    encoded_maps = [encode_fn(*maps) for maps in zip(pred_maps, gt_maps)]

    return lowres, encoded_maps


def decode(predictions_fn, decode_fn, lowres, encoded_maps, padding=0):
    # Assert valid padding
    assert padding >= 0

    # Assert the input is large enough
    ld, lh, lw = lowres.shape[1:4]
    assert ld >= 2
    assert lh >= 2
    assert lw >= 2

    # Extract the predicted values from the lowres volume
    spatial_padding = [(padding, padding)] * 3
    data_padding    = ((0, 0),) * len(lowres.shape[4:])
    pred_maps       = predictions_fn(jnp.pad(lowres, ((0, 0), *spatial_padding, *data_padding), mode='symmetric'))

    # Correct the predictions using the provided encoded maps
    decoded_maps = [decode_fn(*maps) for maps in zip(pred_maps, encoded_maps)]

    # Reconstruct highres volume from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    return highres


def yield_chunks(max_value, chunk):
    # Assert max value is positive
    assert max_value > 0

    # Assert chunk size is valid
    assert chunk > 1

    # Yield a set of chunks along one axis including boundary conditions
    for idx in range(0, max_value, chunk):
        # Is this the last chunk?
        last = ((idx + chunk) < max_value)
        # Determine if this chunk needs start or end padding
        p0, p1 = (1 if (idx > 0) else 0), (1 if last else 0)
        # Determine the start and end coordinates of this chunk
        i1 = min(max_value, idx+chunk+1)
        # Pad the start of the chunk by 1 extra pixel if it is the last and a singleton
        i0 = max(0, idx-(0 if (last and ((i1-idx) <= 1)) else 1))
        # Yield the chunk
        yield (i0, i1), (p0, p1)


def chunk_from_lowres(lowres, z, y, x, padding):
    # Assert valid padding
    assert padding >= 0

    # Determine size of lowres input
    ld, lh, lw = lowres.shape[1:4]

    # Extract coordinates
    (z0, z1), (y0, y1), (x0, x1) = z, y, x

    # Assert chunk has valid size
    assert (z1 - z0) >= 2
    assert (y1 - y0) >= 2
    assert (x1 - x0) >= 2

    # Calculate the amount of padding we can collect directly from the input
    cz0, cz1 = max(0, (z0-padding)), min(ld, (z1+padding))
    cy0, cy1 = max(0, (y0-padding)), min(lh, (y1+padding))
    cx0, cx1 = max(0, (x0-padding)), min(lw, (x1+padding))

    # Calculate how much additional padding is still needed
    pz0, pz1 = (padding - (z0 - cz0)), (padding - (cz1 - z1))
    py0, py1 = (padding - (y0 - cy0)), (padding - (cy1 - y1))
    px0, px1 = (padding - (x0 - cx0)), (padding - (cx1 - x1))

    # Extract the chunk and apply the additional padding
    spatial_padding = ((pz0, pz1), (py0, py1), (px0, px1))
    data_padding    = ((0, 0),) * len(lowres.shape[4:])
    return jnp.pad(lowres[:, cz0:cz1, cy0:cy1, cx0:cx1], ((0, 0), *spatial_padding, *data_padding), mode='symmetric')


def encode_chunks(predictions_fn, encode_fn, highres, chunk=8, padding=0, progress_fn=None):
    # Assert valid padding
    assert padding >= 0

    # Assert chunk size is valid
    assert chunk > 1

    # Assert the input is large enough
    hd, hh, hw = highres.shape[1:4]
    assert hd > 2 and (hd % 2) != 0
    assert hh > 2 and (hh % 2) != 0
    assert hw > 2 and (hw % 2) != 0

    # Extract the lowres volume from the highres volume
    lowres = lowres_from_highres(highres)
    ld, lh, lw = lowres.shape[1:4]

    # Extract the plus values from the highres volume
    gt_maps = maps_from_highres(highres)

    # Pre-allocate full encoded maps
    encoded_maps = [jnp.zeros_like(gt_map) for gt_map in gt_maps]

    chunks = product(yield_chunks(ld, chunk), yield_chunks(lh, chunk), yield_chunks(lw, chunk))
    if progress_fn is not None:
        # If a progress callback was given wrap the list of chunks
        chunks = progress_fn(list(chunks))

    for ((z0, z1), (pz0, pz1)), ((y0, y1), (py0, py1)), ((x0, x1), (px0, px1)) in chunks:

        # Extract the chunks predicted values from the lowres chunk
        chunk_pred_maps = predictions_fn(chunk_from_lowres(lowres, z=((z0-pz0), (z1+pz1)),
                                                                   y=((y0-py0), (y1+py1)),
                                                                   x=((x0-px0), (x1+px1)), padding=padding))

        # Update each encoded maps with the values for this chunk
        for idx, (chunk_pred_map, gt_map) in enumerate(zip(chunk_pred_maps, gt_maps)):
            pd, ph, pw = (chunk_pred_map.shape[1] - (pz0+pz1)), \
                         (chunk_pred_map.shape[2] - (py0+py1)), \
                         (chunk_pred_map.shape[3] - (px0+px1))
            chunk_encoded_map = encode_fn(chunk_pred_map[:, pz0:(pz0+pd), py0:(py0+ph), px0:(px0+pw)],
                                          gt_map[:, z0:(z0+pd), y0:(y0+ph), x0:(x0+pw)])
            encoded_maps[idx] = encoded_maps[idx].at[:, z0:(z0+pd), y0:(y0+ph), x0:(x0+pw)].set(chunk_encoded_map)

    return lowres, encoded_maps


def decode_chunks(predictions_fn, decode_fn, lowres, encoded_maps, chunk=8, padding=0, progress_fn=None):
    # Assert valid padding
    assert padding >= 0

    # Assert chunk size is valid
    assert chunk > 1

    # Assert the input is large enough
    ld, lh, lw = lowres.shape[1:4]
    assert ld >= 2
    assert lh >= 2
    assert lw >= 2

    # Pre-allocate full decoded maps
    decoded_maps = [jnp.zeros_like(encoded_map) for encoded_map in encoded_maps]

    chunks = product(yield_chunks(ld, chunk), yield_chunks(lh, chunk), yield_chunks(lw, chunk))
    if progress_fn is not None:
        # If a progress callback was given wrap the list of chunks
        chunks = progress_fn(list(chunks))

    for ((z0, z1), (pz0, pz1)), ((y0, y1), (py0, py1)), ((x0, x1), (px0, px1)) in chunks:

        # Extract the chunks predicted values from the lowres chunk
        chunk_pred_maps = predictions_fn(chunk_from_lowres(lowres, z=((z0-pz0), (z1+pz1)),
                                                                   y=((y0-py0), (y1+py1)),
                                                                   x=((x0-px0), (x1+px1)), padding=padding))

        # Update each decoded maps with the values for this chunk
        for idx, (chunk_pred_map, encoded_map) in enumerate(zip(chunk_pred_maps, encoded_maps)):
            pd, ph, pw = (chunk_pred_map.shape[1] - (pz0+pz1)), \
                         (chunk_pred_map.shape[2] - (py0+py1)), \
                         (chunk_pred_map.shape[3] - (px0+px1))
            chunk_decoded_map = decode_fn(chunk_pred_map[:, pz0:(pz0+pd), py0:(py0+ph), px0:(px0+pw)],
                                          encoded_map[:, z0:(z0+pd), y0:(y0+ph), x0:(x0+pw)])
            decoded_maps[idx] = decoded_maps[idx].at[:, z0:(z0+pd), y0:(y0+ph), x0:(x0+pw)].set(chunk_decoded_map)

    # Reconstruct highres volume from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    return highres
