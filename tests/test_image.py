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
from functools import partial
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp

# Test imports
import kompressor as kom


class ImageTest(unittest.TestCase):

    def dummy_highres(self):
        shape = (2, 5, 5, 3)
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % 256).astype(jnp.uint8)
        return highres

    def test_targets_from_highres(self):

        highres = self.dummy_highres()

        targets = kom.image.targets_from_highres(highres)

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

        lowres = kom.image.lowres_from_highres(highres)

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

        predictions = kom.image.targets_from_highres(highres)

        lrmap, udmap, cmap = kom.image.maps_from_predictions(predictions)

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

        lrmap, udmap, cmap = kom.image.maps_from_highres(highres)

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
        lowres  = kom.image.lowres_from_highres(highres)
        maps    = kom.image.maps_from_highres(highres)

        reconstructed_highres = kom.image.highres_from_lowres_and_maps(lowres, maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

        highres     = self.dummy_highres()
        lowres      = kom.image.lowres_from_highres(highres)
        predictions = kom.image.targets_from_highres(highres)
        maps        = kom.image.maps_from_predictions(predictions)

        reconstructed_highres = kom.image.highres_from_lowres_and_maps(lowres, maps)

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
            return kom.image.maps_from_predictions(predictions)

        encode_fn = kom.utils.encode_values_uint8
        decode_fn = kom.utils.decode_values_uint8

        lowres, maps          = kom.image.encode(predictions_fn, encode_fn, highres)
        reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, maps)

        lrmap, udmap, cmap = maps

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

            return kom.image.maps_from_predictions(predictions)

        encode_fn = kom.utils.encode_categorical
        decode_fn = kom.utils.decode_categorical

        lowres, maps          = kom.image.encode(predictions_fn, encode_fn, highres)
        lrmap, udmap, cmap = maps

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

        reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

    def encode_decode_chunks(self, encode_chunk, decode_chunk):

        encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks chunk={encode_chunk:d}')
        decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks chunk={decode_chunk:d}')

        shape = (2, 17, 17, 3)
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % 256).astype(jnp.uint8)

        # Dummy predictor function just predicts all ones
        def predictions_fn(lowres):
            predictions = jnp.ones((lowres.shape[0],
                                    lowres.shape[1] - 1,
                                    lowres.shape[2] - 1,
                                    5, *lowres.shape[3:]), dtype=lowres.dtype)
            return kom.image.maps_from_predictions(predictions)

        encode_fn = kom.utils.encode_values_uint8
        decode_fn = kom.utils.decode_values_uint8

        lowres, maps = kom.image.encode(predictions_fn, encode_fn, highres)
        lrmap, udmap, cmap = maps

        chunk_lowres, chunk_maps = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                           chunk=encode_chunk, progress_fn=encode_progress_fn)
        chunk_lrmap, chunk_udmap, chunk_cmap = chunk_maps

        self.assertEqual(chunk_lowres.dtype, lowres.dtype)
        self.assertEqual(chunk_lowres.ndim, lowres.ndim)
        self.assertTrue(np.allclose(chunk_lowres, lowres))

        self.assertEqual(chunk_lrmap.dtype, lrmap.dtype)
        self.assertEqual(chunk_lrmap.ndim, lrmap.ndim)
        self.assertTrue(np.allclose(chunk_lrmap, lrmap))

        self.assertEqual(chunk_udmap.dtype, udmap.dtype)
        self.assertEqual(chunk_udmap.ndim, udmap.ndim)
        self.assertTrue(np.allclose(chunk_udmap, udmap))

        self.assertEqual(chunk_cmap.dtype, cmap.dtype)
        self.assertEqual(chunk_cmap.ndim, cmap.ndim)
        self.assertTrue(np.allclose(chunk_cmap, cmap))

        chunk_highres = kom.image.decode_chunks(predictions_fn, decode_fn, chunk_lowres, chunk_maps,
                                                chunk=decode_chunk, progress_fn=decode_progress_fn)

        self.assertEqual(chunk_highres.dtype, highres.dtype)
        self.assertEqual(chunk_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(chunk_highres, highres))

    def test_encode_decode_chunks_2_2(self):
        self.encode_decode_chunks(encode_chunk=2, decode_chunk=2)

    def test_encode_decode_chunks_2_3(self):
        self.encode_decode_chunks(encode_chunk=2, decode_chunk=3)

    def test_encode_decode_chunks_3_2(self):
        self.encode_decode_chunks(encode_chunk=3, decode_chunk=2)

    def test_encode_decode_chunks_10_10(self):
        self.encode_decode_chunks(encode_chunk=10, decode_chunk=10)

    def test_encode_decode_chunks_2_10(self):
        self.encode_decode_chunks(encode_chunk=2, decode_chunk=10)

    def test_encode_decode_chunks_10_2(self):
        self.encode_decode_chunks(encode_chunk=10, decode_chunk=2)
