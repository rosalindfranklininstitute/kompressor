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
import jax.numpy as jnp

# Import kompressor volume utilities
from .utils import \
    lowres_from_highres, maps_from_highres, highres_from_lowres_and_maps, \
    pad_neighborhood, pad_highres, pad_lowres, trim, pad_maps, trim_maps, \
    validate_highres, validate_lowres, validate_chunk, validate_padding, yield_chunks


def encode_chunks(predictions_fn, encode_fn, highres, chunk=32, padding=0, progress_fn=None):
    # If highres has even spatial dimensions pad by 1
    highres, dims = pad_highres(highres)

    # Assert the input is large enough
    validate_highres(highres)

    # Extract the lowres volume from the highres volume
    lowres = lowres_from_highres(highres)

    # Extract the plus values from the highres volume
    gt_maps = maps_from_highres(highres)

    # Compute encoded maps in chunks
    encoded_maps = trim_maps(process_chunks(predictions_fn, encode_fn, lowres, gt_maps,
                                            chunk=chunk, padding=padding, progress_fn=progress_fn), dims)

    # Trim even padding off lowres if needed
    lowres = trim(lowres, dims)

    return lowres, (encoded_maps, dims)


def decode_chunks(predictions_fn, decode_fn, lowres, encoded, chunk=32, padding=0, progress_fn=None):
    # Unpack maps and dynamic padding
    encoded_maps, dims = encoded

    # Pad lowres using reflection if the original highres had even spatial dimensions
    lowres = pad_lowres(lowres, dims)

    # Apply even padding to encoded maps if needed
    encoded_maps = pad_maps(encoded_maps, dims)

    # Compute decoded maps in chunks
    decoded_maps = process_chunks(predictions_fn, decode_fn, lowres, encoded_maps,
                                  chunk=chunk, padding=padding, progress_fn=progress_fn)

    # Reconstruct highres volume from the corrected true values, trim off even padding if needed
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    # Trim off even padding if needed
    return trim(highres, dims)


def process_chunks(predictions_fn, code_fn, lowres, reference_maps, chunk, padding, progress_fn):
    # Assert valid padding
    validate_padding(padding)

    # Assert chunk size is valid
    cd, ch, cw = validate_chunk(chunk)

    # Assert the input is large enough
    ld, lh, lw = validate_lowres(lowres)

    # Pad lowres input to allow chunk processing
    padded_lowres = pad_neighborhood(lowres, padding)

    # Pre-allocate full decoded maps
    coded_maps = [jnp.zeros_like(reference_map) for reference_map in reference_maps]

    chunks = product(yield_chunks(ld, cd), yield_chunks(lh, ch), yield_chunks(lw, cw))
    if progress_fn is not None:
        # If a progress callback was given wrap the list of chunks
        chunks = progress_fn(list(chunks))

    for ((z0, z1), (pz0, pz1)), ((y0, y1), (py0, py1)), ((x0, x1), (px0, px1)) in chunks:

        # Extract the current chunk with padding and overlaps
        chunk_lowres = padded_lowres[:, (z0-pz0):(z1+pz1+(padding*2)),
                                        (y0-py0):(y1+py1+(padding*2)),
                                        (x0-px0):(x1+px1+(padding*2))]

        # Extract the chunks predicted values from the lowres chunk
        chunk_pred_maps = predictions_fn(chunk_lowres)

        # Update each decoded maps with the values for this chunk
        for idx, (chunk_pred_map, reference_map) in enumerate(zip(chunk_pred_maps, reference_maps)):
            pd, ph, pw = (chunk_pred_map.shape[1] - (pz0+pz1)), \
                         (chunk_pred_map.shape[2] - (py0+py1)), \
                         (chunk_pred_map.shape[3] - (px0+px1))
            chunk_coded_map = code_fn(chunk_pred_map[:, pz0:(pz0+pd), py0:(py0+ph), px0:(px0+pw)],
                                      reference_map[:, z0:(z0+pd), y0:(y0+ph), x0:(x0+pw)])
            coded_maps[idx] = coded_maps[idx].at[:, z0:(z0+pd), y0:(y0+ph), x0:(x0+pw)].set(chunk_coded_map)

    return coded_maps
