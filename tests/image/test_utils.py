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

    def test_pad(self):
        """
        Test we can pad a lowres image.
        """

        for padding in range(4):
            with self.subTest(padding=padding):

                # Get a dummy lowres image to extract features from
                lowres  = kom.image.lowres_from_highres(self.dummy_highres())

                # Apply the padding to the lowres
                padded_lowres = kom.image.pad(lowres, padding)

                # Check the extract features have the correct shape and dtype
                self.assertEqual(padded_lowres.dtype, lowres.dtype)
                self.assertEqual(padded_lowres.ndim, lowres.ndim)
                self.assertTrue(np.allclose(padded_lowres.shape, [
                    lowres.shape[0],
                    lowres.shape[1] + (padding*2),
                    lowres.shape[2] + (padding*2),
                    *lowres.shape[3:]
                ]))

    def test_validate_highres(self):
        """
        Validate bad highres shapes raises exception.
        """

        with self.subTest('Input with no explicit data dimensions throws exception'):
            self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_highres,
                                   self.dummy_highres(shape=(2, 3, 3, 1))[..., 0])

        with self.subTest('Input with degenerate dimensions throws exception'):
            self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_highres,
                                   self.dummy_highres(shape=(0, 3, 3, 1)))

        for shape in [(0, 0), (1, 1), (2, 2), (2, 3), (3, 2)]:
            with self.subTest('Spatial dimensions less than or equal to 2 throw exception',
                              shape=shape):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_highres,
                                       self.dummy_highres(shape=(2, *shape, 3)))

        for shape in [(4, 4), (6, 6)]:
            with self.subTest('Spatial dimensions divisible by 2 throw exception',
                              shape=shape):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_highres,
                                       self.dummy_highres(shape=(2, *shape, 3)))

        for shape in [(3, 3), (3, 5), (5, 3)]:
            with self.subTest('Spatial dimensions greater than or equal to 3 return shape tuple',
                              shape=shape):
                hh, hw = kom.image.utils.validate_highres(self.dummy_highres(shape=(2, *shape, 3)))

    def test_validate_lowres(self):
        """
        Validate bad lowres shapes raises exception.
        """

        with self.subTest('Input with no explicit data dimensions throws exception'):
            self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_highres,
                                   self.dummy_highres(shape=(2, 2, 2, 1))[..., 0])

        with self.subTest('Input with degenerate dimensions throws exception'):
            self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_highres,
                                   self.dummy_highres(shape=(0, 2, 2, 1)))

        for shape in [(0, 0), (1, 1), (1, 2), (2, 1)]:
            with self.subTest('Spatial dimensions less than or equal to 1 throw exception',
                              shape=shape):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_lowres,
                                       self.dummy_highres(shape=(2, *shape, 3)))

        for shape in [(2, 2), (2, 3), (3, 2)]:
            with self.subTest('Spatial dimensions greater than or equal to 2 return shape tuple',
                              shape=shape):
                lh, lw = kom.image.utils.validate_lowres(self.dummy_highres(shape=(2, *shape, 3)))

    def test_validate_padding(self):
        """
        Validate that negative padding raises exception.
        """

        for padding in [None]:
            with self.subTest('Non int padding throws exception',
                              padding=padding):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_padding, padding)

        for padding in range(-4, 0):
            with self.subTest('Negative padding should throw exception',
                              padding=padding):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_padding, padding)

        for padding in range(0, 4):
            with self.subTest('Positive padding should return without error',
                              padding=padding):
                kom.image.utils.validate_padding(padding)

    def test_validate_chunk(self):
        """
        Validate that invalid chunk sizes throw exceptions, and valid ones are normalized into tuples.
        """

        for chunk in [(4,), (4, 4, 4)]:
            with self.subTest('Bad shaped tuple throws exception',
                              chunk=chunk):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_chunk, chunk)

        for chunk in [None, (None, 4), (4, None)]:
            with self.subTest('Non int chunk throws exception',
                              chunk=chunk):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_chunk, chunk)

        for chunk in [3, (3, 4), (4, 3)]:
            with self.subTest('Chunk less than 4 in any dimension should throw exception',
                              chunk=chunk):
                self.assertRaisesRegex(Exception, '.*', kom.image.utils.validate_chunk, chunk)

        for chunk in [4, 5, (4, 4), (4, 5), (5, 4)]:
            with self.subTest('Chunk greater than or equal to 4 in all dimensions should return a tuple of chunk sizes',
                              chunk=chunk):
                ch, cw = kom.image.utils.validate_chunk(chunk)
