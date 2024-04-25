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


# Import kompressor image utilities
from .utils import \
    lowres_from_highres, maps_from_highres, highres_from_lowres_and_maps, \
    pad_neighborhood, pad_highres, pad_lowres, trim, pad_maps, trim_maps, \
    validate_highres, validate_lowres, validate_padding


def encode(predictions_fn, encode_fn, highres, padding=0):
    # Assert valid padding
    validate_padding(padding)

    # If highres has even spatial dimensions pad by 1
    highres, dims = pad_highres(highres)

    # Assert the input is large enough
    validate_highres(highres)

    # Extract the lowres image from the highres image
    lowres = lowres_from_highres(highres)
    validate_lowres(lowres)

    # Extract the plus values from the highres image
    gt_maps = maps_from_highres(highres)

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(pad_neighborhood(lowres, padding))

    # Compare the predictions to the true values for the pluses, trim the resulting maps if even padding was applied
    encoded_maps = trim_maps(jax.tree_map(encode_fn, pred_maps, gt_maps), dims)

    # Trim even padding off lowres if needed
    lowres = trim(lowres, dims)

    return lowres, (encoded_maps, dims)


def decode(predictions_fn, decode_fn, lowres, encoded, padding=0):
    # Unpack maps and dynamic padding
    encoded_maps, dims = encoded

    # Assert valid padding
    validate_padding(padding)

    # Assert the input is large enough
    validate_lowres(lowres)

    # Pad lowres using reflection if the original highres had even spatial dimensions
    lowres = pad_lowres(lowres, dims)

    # Apply even padding to encoded maps if needed
    encoded_maps = pad_maps(encoded_maps, dims)

    # Extract the predicted values from the lowres image
    pred_maps = predictions_fn(pad_neighborhood(lowres, padding))

    # Correct the predictions using the provided encoded maps
    decoded_maps = jax.tree_map(decode_fn, pred_maps, encoded_maps)

    # Reconstruct highres image from the corrected true values
    highres = highres_from_lowres_and_maps(lowres, decoded_maps)

    # Trim off even padding if needed
    return trim(highres, dims)
