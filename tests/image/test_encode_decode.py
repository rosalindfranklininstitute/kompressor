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
from functools import partial
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp

# Test imports
import kompressor as kom


class ImageEncodeDecodeTest(unittest.TestCase):

    def dummy_highres(self, shape=(2, 17, 17, 3), max_value=256, dtype=jnp.uint8):
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % max_value).astype(dtype)
        return highres

    def dummy_predictions_fn(self, padding):

        # Dummy regression predictor function just predicts the average of the neighborhood features
        def predictions_fn(lowres):
            # Extract the features for each neighborhood
            features = kom.image.features_from_lowres(lowres, padding)
            # Dummy predictions are just the average values of the neighborhoods
            predictions = jnp.mean(features.astype(jnp.float32), axis=3, keepdims=True).astype(lowres.dtype)
            predictions = jnp.repeat(predictions, repeats=5, axis=3)
            # Extract the maps from the predictions
            return kom.image.maps_from_predictions(predictions)

        return predictions_fn

    def dummy_predictions_categorical_fn(self, padding, classes):

        # Dummy categorical predictor function just predicts a constant set of logits
        def predictions_fn(lowres):
            # Sample a constant set of random logits (for test consistency)
            key = jax.random.PRNGKey(1234)
            logits = jax.random.uniform(key, (1, 1, 1, 5, *lowres.shape[3:], classes))
            # Tile the same logits for every pixel and batch element
            predictions = jnp.tile(logits, (lowres.shape[0],
                                            (lowres.shape[1]-1) - (padding*2),
                                            (lowres.shape[2]-1) - (padding*2),
                                            1, *([1] * len(lowres.shape[3:])), 1))
            # Extract the maps from the predictions
            pred_maps = kom.image.maps_from_predictions(predictions)
            # Convert the predictions to dummy logits (one hot encodings)
            return [jax.nn.softmax(pred_map, axis=-1) for pred_map in pred_maps]

        return predictions_fn

    def test_encode_decode(self):
        """
        Test we can do an encode + decode cycle on an image processing the whole input at once using different paddings.
        """

        for padding in range(2):
            with self.subTest('Test with odd dimensions',
                              padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.image.encode_values_uint8
                decode_fn = kom.image.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire image at once
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Check that even padding was applied correctly
                eh, ew = dims
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)

                # Check that the lowres and maps are the correct sizes and dtypes
                lrmap, udmap, cmap = maps

                self.assertEqual(lowres.dtype, highres.dtype)
                self.assertEqual(lowres.ndim, highres.ndim)
                self.assertTrue(np.allclose(lowres.shape, [
                    highres.shape[0],
                    ((highres.shape[1] - 1) // 2) + 1,
                    ((highres.shape[2] - 1) // 2) + 1,
                    *highres.shape[3:]
                ]))

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

                # Decode the entire image at once
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for shape, padding in product([(16, 16), (16, 17), (17, 16)], range(2)):
            with self.subTest('Test with even dimensions',
                              shape=shape, padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.image.encode_values_uint8
                decode_fn = kom.image.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres(shape=(2, *shape, 3))

                # Encode the entire image at once
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Check that even padding was applied correctly
                eh, ew = dims
                self.assertEqual(eh, (shape[0] + 1) % 2)
                self.assertEqual(ew, (shape[1] + 1) % 2)

                # Decode the entire image at once
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode_categorical(self):
        """
        Test we can do an encode + decode cycle on an image processing the whole input at once using different paddings
        using a categorical predictor and encoding.
        """

        for padding in range(2):
            with self.subTest('Test with odd dimensions',
                              padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.image.encode_categorical
                decode_fn = kom.image.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire image at once
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Check that even padding was applied correctly
                eh, ew = dims
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)

                # Check that the lowres and maps are the correct sizes and dtypes
                lrmap, udmap, cmap = maps

                self.assertEqual(lowres.dtype, highres.dtype)
                self.assertEqual(lowres.ndim, highres.ndim)
                self.assertTrue(np.allclose(lowres.shape, [
                    highres.shape[0],
                    ((highres.shape[1] - 1) // 2) + 1,
                    ((highres.shape[2] - 1) // 2) + 1,
                    *highres.shape[3:]
                ]))

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

                # Decode the entire image at once
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for shape, padding in product([(16, 16), (16, 17)], range(2)):
            with self.subTest('Test with even dimensions',
                              shape=shape, padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.image.encode_categorical
                decode_fn = kom.image.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres(shape=(2, *shape, 3))

                # Encode the entire image at once
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Check that even padding was applied correctly
                eh, ew = dims
                self.assertEqual(eh, (shape[0] + 1) % 2)
                self.assertEqual(ew, (shape[1] + 1) % 2)

                # Decode the entire image at once
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_chunks(self):
        """
        Test we can encode an image processing the input in chunks and with different paddings.
        """

        for encode_chunk, padding in product([6, 11, (6, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              encode_chunk=encode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.image.encode_values_uint8
                decode_fn = kom.image.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                                       padding=padding)
                full_lrmap, full_udmap, full_cmap = full_maps

                # Encode the input in chunks
                lowres, (maps, dims) = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                               chunk=encode_chunk, padding=padding,
                                                               progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (eh, ew), (full_eh, full_ew) = dims, full_dims
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Check that processing in chunks gives the same results as processing all at once
                lrmap, udmap, cmap = maps

                self.assertEqual(lowres.dtype, full_lowres.dtype)
                self.assertEqual(lowres.ndim, full_lowres.ndim)
                self.assertTrue(np.allclose(lowres, full_lowres))

                self.assertEqual(lrmap.dtype, full_lrmap.dtype)
                self.assertEqual(lrmap.ndim, full_lrmap.ndim)
                self.assertTrue(np.allclose(lrmap, full_lrmap))

                self.assertEqual(udmap.dtype, full_udmap.dtype)
                self.assertEqual(udmap.ndim, full_udmap.ndim)
                self.assertTrue(np.allclose(udmap, full_udmap))

                self.assertEqual(cmap.dtype, full_cmap.dtype)
                self.assertEqual(cmap.ndim, full_cmap.ndim)
                self.assertTrue(np.allclose(cmap, full_cmap))

                # Decode the image in one pass
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for shape, encode_chunk, padding in product([(16, 16), (16, 17)], [6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              shape=shape, encode_chunk=encode_chunk, padding=padding):
                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.image.encode_values_uint8
                decode_fn = kom.image.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres(shape=(2, *shape, 3))

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                                       padding=padding)

                # Encode the input in chunks
                lowres, (maps, dims) = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                               chunk=encode_chunk, padding=padding,
                                                               progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (eh, ew), (full_eh, full_ew) = dims, full_dims
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Decode the image in one pass
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_decode_chunks(self):
        """
        Test we can decode an image processing the input in chunks and with different paddings.
        """

        for decode_chunk, padding in product([6, 11, (6, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.image.encode_values_uint8
                decode_fn = kom.image.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Decode the image in chunks
                reconstructed_highres = kom.image.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                chunk=decode_chunk, padding=padding,
                                                                progress_fn=decode_progress_fn)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for shape, decode_chunk, padding in product([(16, 16), (16, 17)], [6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              shape=shape, decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.image.encode_values_uint8
                decode_fn = kom.image.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres(shape=(2, *shape, 3))

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Decode the image in chunks
                reconstructed_highres = kom.image.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                chunk=decode_chunk, padding=padding,
                                                                progress_fn=decode_progress_fn)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_chunks_categorical(self):
        """
        Test we can encode an image processing the input in chunks and with different paddings
        with a categorical predictions function.
        """

        for encode_chunk, padding in product([6, 11, (6, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              encode_chunk=encode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks_categorical '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.image.encode_categorical
                decode_fn = kom.image.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                                       padding=padding)
                full_lrmap, full_udmap, full_cmap = full_maps

                # Encode the input in chunks
                lowres, (maps, dims) = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                               chunk=encode_chunk, padding=padding,
                                                               progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (eh, ew), (full_eh, full_ew) = dims, full_dims
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Check that processing in chunks gives the same results as processing all at once
                lrmap, udmap, cmap = maps

                self.assertEqual(lowres.dtype, full_lowres.dtype)
                self.assertEqual(lowres.ndim, full_lowres.ndim)
                self.assertTrue(np.allclose(lowres, full_lowres))

                self.assertEqual(lrmap.dtype, full_lrmap.dtype)
                self.assertEqual(lrmap.ndim, full_lrmap.ndim)
                self.assertTrue(np.allclose(lrmap, full_lrmap))

                self.assertEqual(udmap.dtype, full_udmap.dtype)
                self.assertEqual(udmap.ndim, full_udmap.ndim)
                self.assertTrue(np.allclose(udmap, full_udmap))

                self.assertEqual(cmap.dtype, full_cmap.dtype)
                self.assertEqual(cmap.ndim, full_cmap.ndim)
                self.assertTrue(np.allclose(cmap, full_cmap))

                # Decode the image in one pass
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for shape, encode_chunk, padding in product([(16, 16), (16, 17)], [6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              shape=shape, encode_chunk=encode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks_categorical '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.image.encode_categorical
                decode_fn = kom.image.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres(shape=(2, *shape, 3))

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                                       padding=padding)

                # Encode the input in chunks
                lowres, (maps, dims) = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                               chunk=encode_chunk, padding=padding,
                                                               progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (eh, ew), (full_eh, full_ew) = dims, full_dims
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Decode the image in one pass
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                         padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_decode_chunks_categorical(self):
        """
        Test we can decode an image processing the input in chunks and with different paddings
        with a categorical predictions function.
        """

        for decode_chunk, padding in product([6, 11, (6, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks_categorical '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.image.encode_categorical
                decode_fn = kom.image.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Decode the image in chunks
                reconstructed_highres = kom.image.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                chunk=decode_chunk, padding=padding,
                                                                progress_fn=decode_progress_fn)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for shape, decode_chunk, padding in product([(16, 16), (16, 17)], [6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              shape=shape, decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks_categorical '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.image.encode_categorical
                decode_fn = kom.image.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres(shape=(2, *shape, 3))

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.image.encode(predictions_fn, encode_fn, highres,
                                                        padding=padding)

                # Decode the image in chunks
                reconstructed_highres = kom.image.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                chunk=decode_chunk, padding=padding,
                                                                progress_fn=decode_progress_fn)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))
