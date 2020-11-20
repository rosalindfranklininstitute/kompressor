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
    lmap = highres[:,  1::2, :-1:2]
    rmap = highres[:,  1::2,  2::2]
    umap = highres[:, :-1:2,  1::2]
    dmap = highres[:,  2::2,  1::2]
    cmap = highres[:,  1::2,  1::2]

    # Stack the vectors LRUDC order with dim [B,H,W,5,...]
    targets = jnp.stack([lmap, rmap, umap, dmap, cmap], axis=3)

    return targets


@jax.jit
def lowres_from_highres(highres):
    # Downsample by skip sampling
    return highres[:, ::2, ::2]


@jax.jit
def maps_from_predictions(predictions):
    # Given a tensor of 5 predictions for each neighborhood, decode the [B,H,W,5,...] predictions into three maps.
    # Averages predictions when there are two for a pixel from adjacent neighborhoods.

    # Determine the size of the highres image given the size of the predictions
    dtype = predictions.dtype
    (batch_size, ph, pw), channels = predictions.shape[:3], predictions.shape[4:]

    # Map for containing aggregated predictions of the left and right of the pluses
    lrmap = jnp.zeros((batch_size, ph, pw + 1, *channels), dtype=jnp.float32)
    lrmap = lrmap.at[:, :, :-1].add(predictions[:, :, :, 0])  # Left predictions
    lrmap = lrmap.at[:, :,  1:].add(predictions[:, :, :, 1])  # Right predictions
    # Normalize LCR map to account for left and right value double predictions
    lrmap = lrmap.at[:, :, 1:-1].mul(0.5)
    lrmap = lrmap.astype(dtype)

    # Map for containing aggregated predictions of the up and down of the pluses
    udmap = jnp.zeros((batch_size, ph + 1, pw, *channels), dtype=jnp.float32)
    udmap = udmap.at[:, :-1, :].add(predictions[:, :, :, 2])  # Up predictions
    udmap = udmap.at[:, 1:,  :].add(predictions[:, :, :, 3])  # Down predictions
    # Normalize UD map to account for up and down value double predictions
    udmap = udmap.at[:, 1:-1, :].mul(0.5)
    udmap = udmap.astype(dtype)

    # Map for containing aggregated predictions of the centre of the pluses
    cmap = predictions[:, :, :, 4]

    return lrmap, udmap, cmap


@jax.jit
def maps_from_highres(highres):
    # Given a highres image extract the three maps of known values from the pluses
    lrmap = highres[:, 1::2,  ::2]
    udmap = highres[:,  ::2, 1::2]
    cmap  = highres[:, 1::2, 1::2]

    return lrmap, udmap, cmap


@jax.jit
def highres_from_lowres_and_maps(lowres, maps):
    # Merge together a lowres image and the three maps of known values representing the missing pluses
    lrmap, udmap, cmap = maps

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


def encode(predictions_fn, encode_fn, highres):

    # Assert the input is large enough
    hh, hw = highres.shape[1:3]
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
    lh, lw = lowres.shape[1:3]
    assert lh >= 2
    assert lw >= 2

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(lowres)

    # Correct the predictions using the provided encoded maps
    decoded_maps = [decode_fn(*maps) for maps in zip(pred_maps, encoded_maps)]

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    return highres


def encode_chunks(predictions_fn, encode_fn, highres, chunk=32):

    # Assert chunk size is valid
    assert chunk > 1

    # Assert the input is large enough
    hh, hw = highres.shape[1:3]
    assert hh > 2 and (hh % 2) != 0
    assert hw > 2 and (hw % 2) != 0

    # Extract the lowres image from the highres image
    lowres = lowres_from_highres(highres)
    lh, lw = lowres.shape[1:3]

    # Extract the plus values from the highres image
    gt_lrmap, gt_udmap, gt_cmap = maps_from_highres(highres)

    # Pre-allocate full encoded maps
    encoded_lrmap = jnp.zeros_like(gt_lrmap)
    encoded_udmap = jnp.zeros_like(gt_udmap)
    encoded_cmap  = jnp.zeros_like(gt_cmap)

    # Update a chunk of the encoded map given a chunk of predictions and the gt map
    def update_map_chunk(encoded_map, y, x, chunk_pred_map, gt_map):
        ph, pw = chunk_pred_map.shape[1:3]
        chunk_encoded_map = encode_fn(chunk_pred_map, gt_map[:, y:(y+ph), x:(x+pw)])
        return encoded_map.at[:, y:(y+ph), x:(x+pw)].set(chunk_encoded_map)

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

            # Extract this chunk from the lowres
            chunk_lowres = lowres[:, ly0:min(lh, ly1+1), lx0:min(lw, lx1+1)]

            # Extract the chunks predicted values from the lowres chunk
            chunk_pred_lrmap, chunk_pred_udmap, chunk_pred_cmap = predictions_fn(chunk_lowres)

            # Update each encoded maps with the values for this chunk
            encoded_lrmap = update_map_chunk(encoded_lrmap, ly0, lx0, chunk_pred_lrmap, gt_lrmap)
            encoded_udmap = update_map_chunk(encoded_udmap, ly0, lx0, chunk_pred_udmap, gt_udmap)
            encoded_cmap  = update_map_chunk(encoded_cmap,  ly0, lx0, chunk_pred_cmap,  gt_cmap)

    return lowres, (encoded_lrmap, encoded_udmap, encoded_cmap)


def decode_chunks(predictions_fn, decode_fn, lowres, encoded_maps, chunk=32):

    # Assert chunk size is valid
    assert chunk > 1

    # Assert the input is large enough
    lh, lw = lowres.shape[1:3]
    assert lh >= 2
    assert lw >= 2

    # Unpack the encoded maps
    encoded_lrmap, encoded_udmap, encoded_cmap = encoded_maps

    # Pre-allocate full decoded maps
    decoded_lrmap = jnp.zeros_like(encoded_lrmap)
    decoded_udmap = jnp.zeros_like(encoded_udmap)
    decoded_cmap  = jnp.zeros_like(encoded_cmap)

    # Update a chunk of the encoded map given a chunk of predictions and the gt map
    def update_map_chunk(decoded_map, y, x, chunk_pred_map, encoded_map):
        ph, pw = chunk_pred_map.shape[1:3]
        chunk_decoded_map = decode_fn(chunk_pred_map, encoded_map[:, y:(y+ph), x:(x+pw)])
        return decoded_map.at[:, y:(y+ph), x:(x+pw)].set(chunk_decoded_map)

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

            # Extract this chunk from the lowres
            chunk_lowres = lowres[:, ly0:min(lh, ly1+1), lx0:min(lw, lx1+1)]

            # Extract the chunks predicted values from the lowres chunk
            chunk_pred_lrmap, chunk_pred_udmap, chunk_pred_cmap = predictions_fn(chunk_lowres)

            # Update each decoded maps with the values for this chunk
            decoded_lrmap = update_map_chunk(decoded_lrmap, ly0, lx0, chunk_pred_lrmap, encoded_lrmap)
            decoded_udmap = update_map_chunk(decoded_udmap, ly0, lx0, chunk_pred_udmap, encoded_udmap)
            decoded_cmap  = update_map_chunk(decoded_cmap,  ly0, lx0, chunk_pred_cmap,  encoded_cmap)

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, (decoded_lrmap, decoded_udmap, decoded_cmap))

    return highres
