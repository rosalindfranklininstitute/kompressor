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

    # Determine the size of the highres image given the size of the predictions
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


@jax.jit
def highres_from_lowres_and_maps(lowres, maps):
    # Merge together a lowres image and the seven maps of known values representing the missing voxels
    lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps

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


def encode(predictions_fn, encode_fn, highres):

    # Assert the input is large enough
    hd, hh, hw = highres.shape[1:4]
    assert hd > 2 and (hd % 2) != 0
    assert hh > 2 and (hh % 2) != 0
    assert hw > 2 and (hw % 2) != 0

    # Extract the lowres image from the highres image
    lowres = lowres_from_highres(highres)

    # Extract the plus values from the highres image
    gt_maps = maps_from_highres(highres)

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(lowres)

    # Compare the predictions to the true values for the pluses
    encoded_maps = [encode_fn(*maps) for maps in zip(pred_maps, gt_maps)]

    return lowres, encoded_maps


def decode(predictions_fn, decode_fn, lowres, encoded_maps):

    # Assert the input is large enough
    ld, lh, lw = lowres.shape[1:4]
    assert ld >= 2
    assert lh >= 2
    assert lw >= 2

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(lowres)

    # Correct the predictions using the provided encoded maps
    decoded_maps = [decode_fn(*maps) for maps in zip(pred_maps, encoded_maps)]

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    return highres


def encode_chunks(predictions_fn, encode_fn, highres, chunk=8, progress_fn=None):

    # Assert chunk size is valid
    assert chunk > 1

    # Assert the input is large enough
    hd, hh, hw = highres.shape[1:4]
    assert hd > 2 and (hd % 2) != 0
    assert hh > 2 and (hh % 2) != 0
    assert hw > 2 and (hw % 2) != 0

    # Extract the lowres image from the highres image
    lowres = lowres_from_highres(highres)
    ld, lh, lw = lowres.shape[1:4]

    # Extract the plus values from the highres image
    gt_lrmap, gt_udmap, gt_fbmap, gt_cmap, gt_zmap, gt_ymap, gt_xmap = maps_from_highres(highres)

    # Pre-allocate full encoded maps
    encoded_lrmap = jnp.zeros_like(gt_lrmap)
    encoded_udmap = jnp.zeros_like(gt_udmap)
    encoded_fbmap = jnp.zeros_like(gt_fbmap)
    encoded_cmap  = jnp.zeros_like(gt_cmap)
    encoded_zmap  = jnp.zeros_like(gt_zmap)
    encoded_ymap  = jnp.zeros_like(gt_ymap)
    encoded_xmap  = jnp.zeros_like(gt_xmap)

    # Update a chunk of the encoded map given a chunk of predictions and the gt map
    def update_map_chunk(encoded_map, z, y, x, chunk_pred_map, gt_map):
        pd, ph, pw = chunk_pred_map.shape[1:4]
        chunk_encoded_map = encode_fn(chunk_pred_map, gt_map[:, z:(z+pd), y:(y+ph), x:(x+pw)])
        return encoded_map.at[:, z:(z+pd), y:(y+ph), x:(x+pw)].set(chunk_encoded_map)

    def yield_chunks():

        for lz0 in range(0, ld, chunk):
            lz1 = min(ld, lz0+chunk)

            # Skip chunks with size 1 in this axis
            if (lz1 - lz0) <= 1:
                continue

            for ly0 in range(0, lh, chunk):
                ly1 = min(lh, ly0+chunk)

                # Skip chunks with size 1 in this axis
                if (ly1 - ly0) <= 1:
                    continue

                for lx0 in range(0, lw, chunk):
                    lx1 = min(lw, lx0+chunk)

                    # Skip chunks with size 1 in this axis
                    if (lx1 - lx0) <= 1:
                        continue

                    yield (lz0, lz1), (ly0, ly1), (lx0, lx1)

    chunks = yield_chunks()
    if progress_fn is not None:
        # If a progress callback was given wrap the list of chunks
        chunks = progress_fn(list(chunks))

    for (lz0, lz1), (ly0, ly1), (lx0, lx1) in chunks:

        # Extract this chunk from the lowres
        chunk_lowres = lowres[:, lz0:min(ld, lz1+1), ly0:min(lh, ly1+1), lx0:min(lw, lx1+1)]

        # Extract the chunks predicted values from the lowres chunk
        chunk_pred_lrmap, chunk_pred_udmap, chunk_pred_fbmap, chunk_pred_cmap,\
            chunk_pred_zmap, chunk_pred_ymap, chunk_pred_xmap = predictions_fn(chunk_lowres)

        # Update each encoded maps with the values for this chunk
        encoded_lrmap = update_map_chunk(encoded_lrmap, lz0, ly0, lx0, chunk_pred_lrmap, gt_lrmap)
        encoded_udmap = update_map_chunk(encoded_udmap, lz0, ly0, lx0, chunk_pred_udmap, gt_udmap)
        encoded_fbmap = update_map_chunk(encoded_fbmap, lz0, ly0, lx0, chunk_pred_fbmap, gt_fbmap)
        encoded_cmap  = update_map_chunk(encoded_cmap,  lz0, ly0, lx0, chunk_pred_cmap,  gt_cmap)
        encoded_zmap  = update_map_chunk(encoded_zmap,  lz0, ly0, lx0, chunk_pred_zmap,  gt_zmap)
        encoded_ymap  = update_map_chunk(encoded_ymap,  lz0, ly0, lx0, chunk_pred_ymap,  gt_ymap)
        encoded_xmap  = update_map_chunk(encoded_xmap,  lz0, ly0, lx0, chunk_pred_xmap,  gt_xmap)

    return lowres, (encoded_lrmap, encoded_udmap, encoded_fbmap, encoded_cmap,
                    encoded_zmap, encoded_ymap, encoded_xmap)


def decode_chunks(predictions_fn, decode_fn, lowres, encoded_maps, chunk=8, progress_fn=None):

    # Assert chunk size is valid
    assert chunk > 1

    # Assert the input is large enough
    ld, lh, lw = lowres.shape[1:4]
    assert ld >= 2
    assert lh >= 2
    assert lw >= 2

    # Unpack the encoded maps
    encoded_lrmap, encoded_udmap, encoded_fbmap, encoded_cmap, \
        encoded_zmap, encoded_ymap, encoded_xmap = encoded_maps

    # Pre-allocate full decoded maps
    decoded_lrmap = jnp.zeros_like(encoded_lrmap)
    decoded_udmap = jnp.zeros_like(encoded_udmap)
    decoded_fbmap = jnp.zeros_like(encoded_fbmap)
    decoded_cmap  = jnp.zeros_like(encoded_cmap)
    decoded_zmap  = jnp.zeros_like(encoded_zmap)
    decoded_ymap  = jnp.zeros_like(encoded_ymap)
    decoded_xmap  = jnp.zeros_like(encoded_xmap)

    # Update a chunk of the encoded map given a chunk of predictions and the gt map
    def update_map_chunk(decoded_map, z, y, x, chunk_pred_map, encoded_map):
        pd, ph, pw = chunk_pred_map.shape[1:4]
        chunk_decoded_map = decode_fn(chunk_pred_map, encoded_map[:, z:(z+pd), y:(y+ph), x:(x+pw)])
        return decoded_map.at[:, z:(z+pd), y:(y+ph), x:(x+pw)].set(chunk_decoded_map)

    def yield_chunks():

        for lz0 in range(0, ld, chunk):
            lz1 = min(ld, lz0+chunk)

            # Skip chunks with size 1 in this axis
            if (lz1 - lz0) <= 1:
                continue

            for ly0 in range(0, lh, chunk):
                ly1 = min(lh, ly0+chunk)

                # Skip chunks with size 1 in this axis
                if (ly1 - ly0) <= 1:
                    continue

                for lx0 in range(0, lw, chunk):
                    lx1 = min(lw, lx0+chunk)

                    # Skip chunks with size 1 in this axis
                    if (lx1 - lx0) <= 1:
                        continue

                    yield (lz0, lz1), (ly0, ly1), (lx0, lx1)

    chunks = yield_chunks()
    if progress_fn is not None:
        # If a progress callback was given wrap the list of chunks
        chunks = progress_fn(list(chunks))

    for (lz0, lz1), (ly0, ly1), (lx0, lx1) in chunks:

        # Extract this chunk from the lowres
        chunk_lowres = lowres[:, lz0:min(ld, lz1+1), ly0:min(lh, ly1+1), lx0:min(lw, lx1+1)]

        # Extract the chunks predicted values from the lowres chunk
        chunk_pred_lrmap, chunk_pred_udmap, chunk_pred_fbmap, chunk_pred_cmap,\
            chunk_pred_zmap, chunk_pred_ymap, chunk_pred_xmap = predictions_fn(chunk_lowres)

        # Update each decoded maps with the values for this chunk
        decoded_lrmap = update_map_chunk(decoded_lrmap, lz0, ly0, lx0, chunk_pred_lrmap, encoded_lrmap)
        decoded_udmap = update_map_chunk(decoded_udmap, lz0, ly0, lx0, chunk_pred_udmap, encoded_udmap)
        decoded_fbmap = update_map_chunk(decoded_fbmap, lz0, ly0, lx0, chunk_pred_fbmap, encoded_fbmap)
        decoded_cmap  = update_map_chunk(decoded_cmap,  lz0, ly0, lx0, chunk_pred_cmap,  encoded_cmap)
        decoded_zmap  = update_map_chunk(decoded_zmap,  lz0, ly0, lx0, chunk_pred_zmap,  encoded_zmap)
        decoded_ymap  = update_map_chunk(decoded_ymap,  lz0, ly0, lx0, chunk_pred_ymap,  encoded_ymap)
        decoded_xmap  = update_map_chunk(decoded_xmap,  lz0, ly0, lx0, chunk_pred_xmap,  encoded_xmap)

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, (decoded_lrmap, decoded_udmap, decoded_fbmap, decoded_cmap,
                                                    decoded_zmap, decoded_ymap, decoded_xmap))

    return highres
