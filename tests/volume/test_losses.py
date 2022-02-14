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


class VolumeLossesTest(unittest.TestCase):

    def dummy_map(self, shape=(2, 4, 4, 4, 1), max_value=65536, dtype=jnp.uint16):
        return (jnp.arange(np.prod(shape)).reshape(shape) % max_value).astype(dtype)

    def test_mean_squared_error(self):
        """
        Test mean_squared_error on volume tensor.
        """

        # Get a dummy map to test against
        dummy = self.dummy_map()

        # Apply the loss function
        loss = kom.volume.losses.mean_squared_error((dummy + 1), dummy)

        # Check the loss has the correct size and dtype
        self.assertEqual(jnp.float32, loss.dtype)
        self.assertEqual(0, loss.ndim)
        self.assertTrue(np.allclose(1.0, loss))

    def test_mean_abs_error(self):
        """
        Test mean_squared_error on volume tensor.
        """

        # Get a dummy map to test against
        dummy = self.dummy_map()

        # Apply the loss function
        loss = kom.volume.losses.mean_abs_error((dummy + 1), dummy)

        # Check the loss has the correct size and dtype
        self.assertEqual(jnp.float32, loss.dtype)
        self.assertEqual(0, loss.ndim)
        self.assertTrue(np.allclose(1.0, loss))

    def test_mean_charbonnier_error(self):
        """
        Test mean_charbonnier_error on volume tensor.
        """

        # Get a dummy map to test against
        dummy = self.dummy_map()

        # Apply the loss function
        loss = kom.volume.losses.mean_charbonnier_error((dummy + 1), dummy, 1e-3)

        # Check the loss has the correct size and dtype
        self.assertEqual(jnp.float32, loss.dtype)
        self.assertEqual(0, loss.ndim)
        self.assertTrue(np.allclose(1.0, loss))

    def test_mean_total_variation(self):
        """
        Test mean_total_variation on volume tensor.
        """

        # Get a dummy map to test against
        dummy = jnp.ones_like(self.dummy_map())

        # Apply the loss function
        loss = kom.volume.losses.mean_total_variation(dummy)

        # Check the loss has the correct size and dtype
        self.assertEqual(jnp.float32, loss.dtype)
        self.assertEqual(0, loss.ndim)
        self.assertTrue(np.allclose(0.0, loss))