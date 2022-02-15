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


# Utility imports
import unittest
from itertools import product
import numpy as np
import jax.numpy as jnp

# Test imports
import kompressor as kom


class ImageMetricsTest(unittest.TestCase):

    def dummy_map(self, shape=(2, 4, 4, 3), max_value=256, dtype=jnp.uint8):
        return (jnp.arange(np.prod(shape)).reshape(shape) % max_value).astype(dtype)

    def test_imageio_rgb_bpp(self):
        """
        Test imageio_rgb_bpp on image tensor.
        """

        with self.subTest('Test single image bpp'):

            # Get a dummy map to test against
            dummy = self.dummy_map(shape=(16, 16, 3))

            # Apply the metric function
            bpp = kom.image.metrics.imageio_rgb_bpp(dummy, format='png')

            # Check the metric has the correct size and dtype
            self.assertEqual(np.float32, bpp.dtype)
            self.assertEqual(0, bpp.ndim)
            self.assertTrue(np.all(bpp < 24.0))

        with self.subTest('Test batch image bpp'):

            # Get a dummy map batch to test against
            dummy = self.dummy_map(shape=(2, 16, 16, 3))

            # Apply the metric function
            bpp = kom.image.metrics.imageio_rgb_bpp(dummy, format='png')

            # Check the metric has the correct size and dtype
            self.assertEqual(np.float32, bpp.dtype)
            self.assertEqual(1, bpp.ndim)
            self.assertEqual((2,), bpp.shape)
            self.assertTrue(np.all(bpp < 24.0))
