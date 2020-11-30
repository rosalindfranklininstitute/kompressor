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

# Import kompressor image utilities
from .utils import \
    lowres_from_highres, maps_from_highres, highres_from_lowres_and_maps, pad, \
    validate_highres, validate_lowres, validate_chunk, validate_padding, yield_chunks


def encode(predictions_fn, encode_fn, highres, padding=0):
    # Assert valid padding
    validate_padding(padding)

    # Assert the input is large enough
    validate_highres(highres)

    # Extract the lowres image from the highres image
    lowres = lowres_from_highres(highres)
    validate_lowres(lowres)

    # Extract the plus values from the highres image
    gt_maps = maps_from_highres(highres)

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(pad(lowres, padding))

    # Compare the predictions to the true values for the pluses
    encoded_maps = [encode_fn(*maps) for maps in zip(pred_maps, gt_maps)]

    return lowres, encoded_maps


def decode(predictions_fn, decode_fn, lowres, encoded_maps, padding=0):
    # Assert valid padding
    validate_padding(padding)

    # Assert the input is large enough
    validate_lowres(lowres)

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(pad(lowres, padding))

    # Correct the predictions using the provided encoded maps
    decoded_maps = [decode_fn(*maps) for maps in zip(pred_maps, encoded_maps)]

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    return highres


def process_chunks(predictions_fn, code_fn, lowres, reference_maps, chunk, padding, progress_fn):
    # Assert valid padding
    validate_padding(padding)

    # Assert chunk size is valid
    ch, cw = validate_chunk(chunk)

    # Extract the lowres image from the highres image
    lh, lw = validate_lowres(lowres)

    # Pad lowres input to allow chunk processing
    padded_lowres = pad(lowres, padding)

    # Pre-allocate full coded maps
    coded_maps = [jnp.zeros_like(reference_map) for reference_map in reference_maps]

    chunks = product(yield_chunks(lh, ch), yield_chunks(lw, cw))
    if progress_fn is not None:
        # If a progress callback was given wrap the list of chunks
        chunks = progress_fn(list(chunks))

    for ((y0, y1), (py0, py1)), ((x0, x1), (px0, px1)) in chunks:

        # Extract the current chunk with padding and overlaps
        chunk_lowres = padded_lowres[:, (y0-py0):(y1+py1+(padding*2)),
                                        (x0-px0):(x1+px1+(padding*2))]

        # Extract the chunks predicted values from the lowres chunk
        chunk_pred_maps = predictions_fn(chunk_lowres)

        # Update each encoded maps with the values for this chunk
        for idx, (chunk_pred_map, reference_map) in enumerate(zip(chunk_pred_maps, reference_maps)):
            ph, pw = (chunk_pred_map.shape[1] - (py0+py1)), \
                     (chunk_pred_map.shape[2] - (px0+px1))
            chunk_coded_map = code_fn(chunk_pred_map[:, py0:(py0+ph), px0:(px0+pw)],
                                        reference_map[:, y0:(y0+ph), x0:(x0+pw)])
            coded_maps[idx] = coded_maps[idx].at[:, y0:(y0+ph), x0:(x0+pw)].set(chunk_coded_map)

    return coded_maps


def encode_chunks(predictions_fn, encode_fn, highres, chunk=32, padding=0, progress_fn=None):
    # Assert the input is large enough
    validate_highres(highres)

    # Extract the lowres image from the highres image
    lowres = lowres_from_highres(highres)

    # Extract the plus values from the highres image
    gt_maps = maps_from_highres(highres)

    # Compute encoded maps in chunks
    encoded_maps = process_chunks(predictions_fn, encode_fn, lowres, gt_maps,
                                  chunk=chunk, padding=padding, progress_fn=progress_fn)

    return lowres, encoded_maps


def decode_chunks(predictions_fn, decode_fn, lowres, encoded_maps, chunk=32, padding=0, progress_fn=None):
    # Compute decoded maps in chunks
    decoded_maps = process_chunks(predictions_fn, decode_fn, lowres, encoded_maps,
                                  chunk=chunk, padding=padding, progress_fn=progress_fn)

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    return highres
