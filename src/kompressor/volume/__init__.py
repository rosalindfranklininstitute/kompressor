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

# Import the common encode decode pairs for handling individual maps
from ..utils import \
    encode_values_raw, decode_values_raw, \
    encode_values_uint8, decode_values_uint8, \
    encode_values_uint16, decode_values_uint16, \
    encode_categorical, decode_categorical

# Import the 3D image utility functions
from .utils import \
    targets_from_highres, lowres_from_highres, \
    maps_from_predictions, maps_from_highres, \
    highres_from_lowres_and_maps, \
    features_from_lowres, pad

# Import the 3D image encode decode functions
from .encode_decode import \
    encode, decode, encode_chunks, decode_chunks