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


class ImageUtilsTest(unittest.TestCase):

    def dummy_highres(self, shape=(2, 17, 17, 3), max_value=256, dtype=jnp.uint8):
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % max_value).astype(dtype)
        return highres

    def test_targets_from_highres(self):
        """
        Test extracting [B, H, W, 5, ...] training targets from the highres images.
        """

        # Get a dummy highres image to extract targets from
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
        Test reconstructing a highres image using an extracted lowres image and the extracted ground truth maps.
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
                spatial_padding = [(padding, padding)] * 2
                data_padding    = ((0, 0),) * len(lowres.shape[3:])
                features = kom.image.features_from_lowres(jnp.pad(lowres, ((0, 0), *spatial_padding, *data_padding),
                                                                  mode='symmetric'), padding)

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