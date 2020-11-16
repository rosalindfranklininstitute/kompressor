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

# Test imports
import numpy as np
import jax
import jax.numpy as jnp
import kompressor as kom


class Utils2DTest(unittest.TestCase):

    def dummy_highres(self):
        shape = (2, 5, 5, 3)
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % 256).astype(jnp.uint8)
        return highres

    def test_targets_from_highres(self):

        highres = self.dummy_highres()

        targets = kom.utils_2d.targets_from_highres(highres)

        self.assertEqual(targets.dtype, highres.dtype)
        self.assertEqual(targets.ndim, highres.ndim + 1)
        self.assertTrue(np.allclose(targets.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            (highres.shape[2] - 1) // 2,
            5,
            *highres.shape[3:]
        ]))

    def test_lowres_from_highres(self):

        highres = self.dummy_highres()

        lowres = kom.utils_2d.lowres_from_highres(highres)

        self.assertEqual(lowres.dtype, highres.dtype)
        self.assertEqual(lowres.ndim, highres.ndim)
        self.assertTrue(np.allclose(lowres.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            ((highres.shape[2] - 1) // 2) + 1,
            *highres.shape[3:]
        ]))

    def test_maps_from_predictions(self):

        highres = self.dummy_highres()

        predictions = kom.utils_2d.targets_from_highres(highres)

        lrmap, udmap, cmap = kom.utils_2d.maps_from_predictions(predictions)

        self.assertEqual(lrmap.dtype, predictions.dtype)
        self.assertEqual(lrmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(lrmap.shape, [
            predictions.shape[0],
            predictions.shape[1],
            predictions.shape[2] + 1,
            *predictions.shape[4:]
        ]))

        self.assertEqual(udmap.dtype, predictions.dtype)
        self.assertEqual(udmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(udmap.shape, [
            predictions.shape[0],
            predictions.shape[1] + 1,
            predictions.shape[2],
            *predictions.shape[4:]
        ]))

        self.assertEqual(cmap.dtype, predictions.dtype)
        self.assertEqual(cmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(cmap.shape, [
            predictions.shape[0],
            predictions.shape[1],
            predictions.shape[2],
            *predictions.shape[4:]
        ]))

    def test_maps_from_highres(self):

        highres = self.dummy_highres()

        lrmap, udmap, cmap = kom.utils_2d.maps_from_highres(highres)

        self.assertEqual(lrmap.dtype, highres.dtype)
        self.assertEqual(lrmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(lrmap.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            ((highres.shape[2] - 1) // 2) + 1,
            *highres.shape[3:]
        ]))

        self.assertEqual(udmap.dtype, highres.dtype)
        self.assertEqual(udmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(udmap.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            (highres.shape[2] - 1) // 2,
            *highres.shape[3:]
        ]))

        self.assertEqual(cmap.dtype, highres.dtype)
        self.assertEqual(cmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(cmap.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            (highres.shape[2] - 1) // 2,
            *highres.shape[3:]
        ]))

    def test_highres_from_lowres_and_maps(self):

        highres = self.dummy_highres()
        lowres  = kom.utils_2d.lowres_from_highres(highres)
        maps    = kom.utils_2d.maps_from_highres(highres)

        reconstructed_highres = kom.utils_2d.highres_from_lowres_and_maps(lowres, maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

        highres     = self.dummy_highres()
        lowres      = kom.utils_2d.lowres_from_highres(highres)
        predictions = kom.utils_2d.targets_from_highres(highres)
        maps        = kom.utils_2d.maps_from_predictions(predictions)

        reconstructed_highres = kom.utils_2d.highres_from_lowres_and_maps(lowres, maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode(self):

        highres = self.dummy_highres()

        # Dummy predictor function just predicts all ones
        def predictions_fn(lowres):
            predictions = jnp.ones((lowres.shape[0],
                                    lowres.shape[1] - 1,
                                    lowres.shape[2] - 1,
                                    5, *lowres.shape[3:]), dtype=lowres.dtype)
            return kom.utils_2d.maps_from_predictions(predictions)

        encode_fn = kom.utils.encode_values_uint8
        decode_fn = kom.utils.decode_values_uint8

        lowres, maps          = kom.utils_2d.encode(predictions_fn, encode_fn, highres)
        reconstructed_highres = kom.utils_2d.decode(predictions_fn, decode_fn, lowres, maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode_categorical(self):

        highres = (self.dummy_highres() % 256).astype(jnp.uint8)

        # Dummy predictor function predicts same random logits every time
        def predictions_fn(lowres):

            shape = (lowres.shape[0],
                     lowres.shape[1] - 1,
                     lowres.shape[2] - 1,
                     5, *lowres.shape[3:], 256)

            key = jax.random.PRNGKey(1234)
            predictions = jax.nn.softmax(jax.random.uniform(key, shape, dtype=jnp.float32), axis=-1)

            return kom.utils_2d.maps_from_predictions(predictions)

        encode_fn = kom.utils.encode_categorical
        decode_fn = kom.utils.decode_categorical

        lowres, maps          = kom.utils_2d.encode(predictions_fn, encode_fn, highres)
        reconstructed_highres = kom.utils_2d.decode(predictions_fn, decode_fn, lowres, maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))
