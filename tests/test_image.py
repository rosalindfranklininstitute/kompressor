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


class ImageTest(unittest.TestCase):

    def dummy_highres(self):
        shape = (2, 17, 17, 3)
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % 256).astype(jnp.uint8)
        return highres

    def dummy_predictions_fn(self, padding):

        # Dummy regression predictor function just predicts the average of the neighborhood features
        def predictions_fn(lowres):
            # Extract the features for each neighborhood
            features = kom.image.features_from_lowres(lowres, padding)
            # Dummy predictions are just the average values of the neighborhoods
            predictions = jnp.mean(features.astype(jnp.float32), axis=3, keepdims=True).astype(jnp.uint8)
            predictions = jnp.repeat(predictions, repeats=5, axis=3)
            # Extract the maps from the predictions
            return kom.image.maps_from_predictions(predictions)

        return predictions_fn

    def dummy_predictions_categorical_fn(self, padding):

        # Dummy categorical predictor function just predicts a constant set of logits
        def predictions_fn(lowres):
            # Sample a constant set of random logits (for test consistency)
            key = jax.random.PRNGKey(1234)
            logits = jax.random.uniform(key, (1, 1, 1, 5, *lowres.shape[3:], 256))
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

    def test_targets_from_highres(self):
        """
        Test extracting [B, H, W, 5, ...] training targets from the highres images.
        """

        # Get a dummy highres image extract targets from
        highres = self.dummy_highres()

        # Extract the targets that a model would be trained to predict
        targets = kom.image.targets_from_highres(highres)

        # Check the predictions have the correct size and dtype
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
        """
        Test downsampling by skipping alternate pixels from the highres input to get the lowres output.
        """

        # Get a dummy highres image to downsample
        highres = self.dummy_highres()

        # Downsample the highres by skip sampling
        lowres = kom.image.lowres_from_highres(highres)

        # Check the lowres has the correct size and dtype
        self.assertEqual(lowres.dtype, highres.dtype)
        self.assertEqual(lowres.ndim, highres.ndim)
        self.assertTrue(np.allclose(lowres.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            ((highres.shape[2] - 1) // 2) + 1,
            *highres.shape[3:]
        ]))

    def test_maps_from_predictions(self):
        """
        Test extracting the LR, UD, and C maps from the prediction tensor, merging duplicates.
        """

        # Get a dummy highres image to extract the maps from
        highres = self.dummy_highres()

        # Extract the predictions from the highres image
        predictions = kom.image.targets_from_highres(highres)

        # Merge duplicate predictions together to get the maps
        lrmap, udmap, cmap = kom.image.maps_from_predictions(predictions)

        # Check the merged maps have the correct sizes and dtypes
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
        """
        Test extracting the LR, UD, and C maps directly from the highres inputs to use as the
        ground truths for encoding.
        """

        # Get a dummy highres image to extract the maps from
        highres = self.dummy_highres()

        # Extract the LR, UD, and C maps from the highres
        lrmap, udmap, cmap = kom.image.maps_from_highres(highres)

        # Check the extracted maps have the correct sizes and dtypes
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
        """
        Test reconstructing a highres image using an extracted lowres image and the extract ground truth maps.
        """

        with self.subTest('Reconstruct using maps_from_highres'):

            # Get a dummy highres image to perform the round trip on
            highres = self.dummy_highres()

            # Extract the lowres image and the maps
            lowres  = kom.image.lowres_from_highres(highres)
            maps    = kom.image.maps_from_highres(highres)

            # Reconstruct the highres image
            reconstructed_highres = kom.image.highres_from_lowres_and_maps(lowres, maps)

            # Check the reconstructed image is lossless
            self.assertEqual(reconstructed_highres.dtype, highres.dtype)
            self.assertEqual(reconstructed_highres.ndim, highres.ndim)
            self.assertTrue(np.allclose(reconstructed_highres, highres))

        with self.subTest('Reconstruct using targets_from_highres + maps_from_predictions'):

            # Get a dummy highres image to perform the round trip on
            highres     = self.dummy_highres()

            # Extract the lowres image and the maps
            lowres      = kom.image.lowres_from_highres(highres)
            predictions = kom.image.targets_from_highres(highres)
            maps        = kom.image.maps_from_predictions(predictions)

            # Reconstruct the highres image
            reconstructed_highres = kom.image.highres_from_lowres_and_maps(lowres, maps)

            # Check the reconstructed image is lossless
            self.assertEqual(reconstructed_highres.dtype, highres.dtype)
            self.assertEqual(reconstructed_highres.ndim, highres.ndim)
            self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_features_from_lowres(self):
        """
        Test we can extract a [B, H, W, N, ...] tensor of neighbor features from a lowres.
        """

        for padding in range(4):
            with self.subTest(padding=padding):

                # Get a dummy lowres image to extract features from
                lowres  = kom.image.lowres_from_highres(self.dummy_highres())

                # Apply the correct padding to the lowres and then extract the features from the padded neighborhoods
                padded_lowres = jnp.pad(lowres, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='reflect')
                features      = kom.image.features_from_lowres(padded_lowres, padding)

                # Check the extract features have the correct shape and dtype
                self.assertEqual(features.dtype, lowres.dtype)
                self.assertEqual(features.ndim, lowres.ndim + 1)
                self.assertTrue(np.allclose(features.shape, [
                    lowres.shape[0],
                    lowres.shape[1] - 1,
                    lowres.shape[2] - 1,
                    ((padding*2)+2)**2,
                    *lowres.shape[3:]
                ]))

    def test_chunk_from_lowres(self):
        """
        Test we can extract chunks from a lowres image, padding should use real pixels from the rest of the image where
        available and use symmetric padding where not.
        """

        for y, x, chunk, padding in product([0, 1, 7], [0, 1, 7], [2, 7, 10], range(4)):
            with self.subTest(y=y, x=x, chunk=chunk, padding=padding):

                # Get a dummy lowres image to extract features from
                lowres = kom.image.lowres_from_highres(self.dummy_highres())

                # Extract the desired chunk with the requested padding
                chunk_lowres = kom.image.chunk_from_lowres(lowres,
                                                           y=(y, min(lowres.shape[1], y+chunk)),
                                                           x=(x, min(lowres.shape[2], x+chunk)),
                                                           padding=padding)

                # Check the extract features have the correct shape and dtype
                self.assertEqual(chunk_lowres.dtype, lowres.dtype)
                self.assertEqual(chunk_lowres.ndim, lowres.ndim)
                self.assertTrue(np.allclose(chunk_lowres.shape, [
                    lowres.shape[0],
                    (min(lowres.shape[1], y + chunk) - y) + (padding * 2),
                    (min(lowres.shape[2], x + chunk) - x) + (padding * 2),
                    *lowres.shape[3:]
                ]))

    def test_encode_decode(self):
        """
        Test we can do an encode + decode cycle on an image processing the whole input at once using different paddings.
        """

        for padding in range(4):
            with self.subTest(padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.utils.encode_values_uint8
                decode_fn = kom.utils.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire image at once
                lowres, maps = kom.image.encode(predictions_fn, encode_fn, highres, padding=padding)

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
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, maps, padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode_categorical(self):
        """
        Test we can do an encode + decode cycle on an image processing the whole input at once using different paddings
        using a categorical predictor and encoding.
        """

        for padding in range(4):
            with self.subTest(padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding)
                encode_fn = kom.utils.encode_categorical
                decode_fn = kom.utils.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire image at once
                lowres, maps = kom.image.encode(predictions_fn, encode_fn, highres, padding=padding)

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
                reconstructed_highres = kom.image.decode(predictions_fn, decode_fn, lowres, maps, padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode_chunks(self):
        """
        Test we can do an encode + decode cycle on an image processing the input in chunks and with different paddings.
        """

        for encode_chunk, decode_chunk, padding in product([2, 3, 10], [2, 3, 10], range(4)):
            with self.subTest(encode_chunk=encode_chunk, decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')
                decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.utils.encode_values_uint8
                decode_fn = kom.utils.decode_values_uint8

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                full_lowres, full_maps = kom.image.encode(predictions_fn, encode_fn, highres, padding=padding)
                full_lrmap, full_udmap, full_cmap = full_maps

                # Encode the input in chunks
                lowres, maps = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                       chunk=encode_chunk, padding=padding,
                                                       progress_fn=encode_progress_fn)

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
                full_highres = kom.image.decode(predictions_fn, decode_fn, lowres, maps, padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(full_highres.dtype, highres.dtype)
                self.assertEqual(full_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(full_highres, highres))

                # Decode the image in chunks
                chunk_highres = kom.image.decode_chunks(predictions_fn, decode_fn, lowres, maps,
                                                        chunk=decode_chunk, padding=padding,
                                                        progress_fn=decode_progress_fn)

                # Check the decoded image is lossless
                self.assertEqual(chunk_highres.dtype, highres.dtype)
                self.assertEqual(chunk_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(chunk_highres, highres))

    def test_encode_decode_chunks_categorical(self):
        """
        Test we can do an encode + decode cycle on an image processing the input in chunks and with different paddings
        with a categorical predictions function.
        """

        for encode_chunk, decode_chunk, padding in product([2, 3, 10], [2, 3, 10], range(4)):
            with self.subTest(encode_chunk=encode_chunk, decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.image.encode_chunks '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')
                decode_progress_fn = partial(tqdm, desc=f'kom.image.decode_chunks '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding)
                encode_fn = kom.utils.encode_categorical
                decode_fn = kom.utils.decode_categorical

                # Get a dummy highres image to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                full_lowres, full_maps = kom.image.encode(predictions_fn, encode_fn, highres, padding=padding)
                full_lrmap, full_udmap, full_cmap = full_maps

                # Encode the input in chunks
                lowres, maps = kom.image.encode_chunks(predictions_fn, encode_fn, highres,
                                                       chunk=encode_chunk, padding=padding,
                                                       progress_fn=encode_progress_fn)

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
                full_highres = kom.image.decode(predictions_fn, decode_fn, lowres, maps, padding=padding)

                # Check the decoded image is lossless
                self.assertEqual(full_highres.dtype, highres.dtype)
                self.assertEqual(full_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(full_highres, highres))

                # Decode the image in chunks
                chunk_highres = kom.image.decode_chunks(predictions_fn, decode_fn, lowres, maps,
                                                        chunk=decode_chunk, padding=padding,
                                                        progress_fn=decode_progress_fn)

                # Check the decoded image is lossless
                self.assertEqual(chunk_highres.dtype, highres.dtype)
                self.assertEqual(chunk_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(chunk_highres, highres))
