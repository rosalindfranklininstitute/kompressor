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


class VolumeUtilsTest(unittest.TestCase):

    def dummy_highres(self, shape=(2, 17, 17, 17, 1), max_value=65536, dtype=jnp.uint16):
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % max_value).astype(dtype)
        return highres

    def test_targets_from_highres(self):
        """
        Test extracting [B, D, H, W, 19, ...] training targets from the highres volumes.
        """

        # Get a dummy highres volume to extract targets from
        highres = self.dummy_highres()

        # Extract the targets that a model would be trained to predict
        targets = kom.volume.targets_from_highres(highres)

        # Check the predictions have the correct size and dtype
        self.assertEqual(targets.dtype, highres.dtype)
        self.assertEqual(targets.ndim, highres.ndim + 1)
        self.assertTrue(np.allclose(targets.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            (highres.shape[2] - 1) // 2,
            (highres.shape[3] - 1) // 2,
            19,
            *highres.shape[4:]
        ]))

    def test_lowres_from_highres(self):
        """
        Test downsampling by skipping alternate voxels from the highres input to get the lowres output.
        """

        # Get a dummy highres volume to downsample
        highres = self.dummy_highres()

        # Downsample the highres by skip sampling
        lowres = kom.volume.lowres_from_highres(highres)

        # Check the lowres has the correct size and dtype
        self.assertEqual(lowres.dtype, highres.dtype)
        self.assertEqual(lowres.ndim, highres.ndim)
        self.assertTrue(np.allclose(lowres.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            ((highres.shape[2] - 1) // 2) + 1,
            ((highres.shape[3] - 1) // 2) + 1,
            *highres.shape[4:]
        ]))

    def test_maps_from_predictions(self):
        """
        Test extracting the LR, UD, FB, C, Z, Y, and X maps from the prediction tensor, merging duplicates.
        """

        # Get a dummy highres volume to extract the maps from
        highres = self.dummy_highres()

        # Extract the predictions from the highres volume
        predictions = kom.volume.targets_from_highres(highres)

        # Merge duplicate predictions together to get the maps
        lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = kom.volume.maps_from_predictions(predictions)

        # Check the merged maps have the correct sizes and dtypes
        self.assertEqual(lrmap.dtype, predictions.dtype)
        self.assertEqual(lrmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(lrmap.shape, [
            predictions.shape[0],
            predictions.shape[1],
            predictions.shape[2],
            predictions.shape[3] + 1,
            *predictions.shape[5:]
        ]))

        self.assertEqual(udmap.dtype, predictions.dtype)
        self.assertEqual(udmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(udmap.shape, [
            predictions.shape[0],
            predictions.shape[1],
            predictions.shape[2] + 1,
            predictions.shape[3],
            *predictions.shape[5:]
        ]))

        self.assertEqual(fbmap.dtype, predictions.dtype)
        self.assertEqual(fbmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(fbmap.shape, [
            predictions.shape[0],
            predictions.shape[1] + 1,
            predictions.shape[2],
            predictions.shape[3],
            *predictions.shape[5:]
        ]))

        self.assertEqual(cmap.dtype, predictions.dtype)
        self.assertEqual(cmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(cmap.shape, [
            predictions.shape[0],
            predictions.shape[1],
            predictions.shape[2],
            predictions.shape[3],
            *predictions.shape[5:]
        ]))

        self.assertEqual(zmap.dtype, predictions.dtype)
        self.assertEqual(zmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(zmap.shape, [
            predictions.shape[0],
            predictions.shape[1],
            predictions.shape[2] + 1,
            predictions.shape[3] + 1,
            *predictions.shape[5:]
        ]))

        self.assertEqual(ymap.dtype, predictions.dtype)
        self.assertEqual(ymap.ndim, highres.ndim)
        self.assertTrue(np.allclose(ymap.shape, [
            predictions.shape[0],
            predictions.shape[1] + 1,
            predictions.shape[2],
            predictions.shape[3] + 1,
            *predictions.shape[5:]
        ]))

        self.assertEqual(xmap.dtype, predictions.dtype)
        self.assertEqual(xmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(xmap.shape, [
            predictions.shape[0],
            predictions.shape[1] + 1,
            predictions.shape[2] + 1,
            predictions.shape[3],
            *predictions.shape[5:]
        ]))

    def test_maps_from_highres(self):
        """
        Test extracting the LR, UD, FB, C, Z, Y, and X maps directly from the highres inputs to use as the
        ground truths for encoding.
        """

        # Get a dummy highres volume to extract the maps from
        highres = self.dummy_highres()

        # Extract the LR, UD, FB, C, Z, Y, and X maps from the highres
        lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = kom.volume.maps_from_highres(highres)

        # Check the extracted maps have the correct sizes and dtypes
        self.assertEqual(lrmap.dtype, highres.dtype)
        self.assertEqual(lrmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(lrmap.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            (highres.shape[2] - 1) // 2,
            ((highres.shape[3] - 1) // 2) + 1,
            *highres.shape[4:]
        ]))

        self.assertEqual(udmap.dtype, highres.dtype)
        self.assertEqual(udmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(udmap.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            ((highres.shape[2] - 1) // 2) + 1,
            (highres.shape[3] - 1) // 2,
            *highres.shape[4:]
        ]))

        self.assertEqual(fbmap.dtype, highres.dtype)
        self.assertEqual(fbmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(fbmap.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            (highres.shape[2] - 1) // 2,
            (highres.shape[3] - 1) // 2,
            *highres.shape[4:]
        ]))

        self.assertEqual(cmap.dtype, highres.dtype)
        self.assertEqual(cmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(cmap.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            (highres.shape[2] - 1) // 2,
            (highres.shape[3] - 1) // 2,
            *highres.shape[4:]
        ]))

        self.assertEqual(zmap.dtype, highres.dtype)
        self.assertEqual(zmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(zmap.shape, [
            highres.shape[0],
            (highres.shape[1] - 1) // 2,
            ((highres.shape[2] - 1) // 2) + 1,
            ((highres.shape[3] - 1) // 2) + 1,
            *highres.shape[4:]
        ]))

        self.assertEqual(ymap.dtype, highres.dtype)
        self.assertEqual(ymap.ndim, highres.ndim)
        self.assertTrue(np.allclose(ymap.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            (highres.shape[2] - 1) // 2,
            ((highres.shape[3] - 1) // 2) + 1,
            *highres.shape[4:]
        ]))

        self.assertEqual(xmap.dtype, highres.dtype)
        self.assertEqual(xmap.ndim, highres.ndim)
        self.assertTrue(np.allclose(xmap.shape, [
            highres.shape[0],
            ((highres.shape[1] - 1) // 2) + 1,
            ((highres.shape[2] - 1) // 2) + 1,
            (highres.shape[3] - 1) // 2,
            *highres.shape[4:]
        ]))

    def test_highres_from_lowres_and_maps(self):
        """
        Test reconstructing a highres volume using an extracted lowres volume and the extracted ground truth maps.
        """

        with self.subTest('Reconstruct using maps_from_highres'):

            # Get a dummy highres volume to perform the round trip on
            highres = self.dummy_highres()

            # Extract the lowres volume and the maps
            lowres  = kom.volume.lowres_from_highres(highres)
            maps    = kom.volume.maps_from_highres(highres)

            # Reconstruct the highres volume
            reconstructed_highres = kom.volume.highres_from_lowres_and_maps(lowres, maps)

            # Check the reconstructed volume is lossless
            self.assertEqual(reconstructed_highres.dtype, highres.dtype)
            self.assertEqual(reconstructed_highres.ndim, highres.ndim)
            self.assertTrue(np.allclose(reconstructed_highres, highres))

        with self.subTest('Reconstruct using targets_from_highres + maps_from_predictions'):

            # Get a dummy highres volume to perform the round trip on
            highres     = self.dummy_highres()

            # Extract the lowres volume and the maps
            lowres      = kom.volume.lowres_from_highres(highres)
            predictions = kom.volume.targets_from_highres(highres)
            maps        = kom.volume.maps_from_predictions(predictions)

            # Reconstruct the highres volume
            reconstructed_highres = kom.volume.highres_from_lowres_and_maps(lowres, maps)

            # Check the reconstructed volume is lossless
            self.assertEqual(reconstructed_highres.dtype, highres.dtype)
            self.assertEqual(reconstructed_highres.ndim, highres.ndim)
            self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_features_from_lowres(self):
        """
        Test we can extract a [B, D, H, W, N, ...] tensor of neighbor features from a lowres.
        """

        for padding in range(4):
            with self.subTest(padding=padding):

                # Get a dummy lowres volume to extract features from
                lowres  = kom.volume.lowres_from_highres(self.dummy_highres())

                # Apply the correct padding to the lowres and then extract the features from the padded neighborhoods
                spatial_padding = [(padding, padding)] * 3
                data_padding = ((0, 0),) * len(lowres.shape[4:])
                features = kom.volume.features_from_lowres(jnp.pad(lowres, ((0, 0), *spatial_padding, *data_padding),
                                                                   mode='symmetric'), padding)

                # Check the extract features have the correct shape and dtype
                self.assertEqual(features.dtype, lowres.dtype)
                self.assertEqual(features.ndim, lowres.ndim + 1)
                self.assertTrue(np.allclose(features.shape, [
                    lowres.shape[0],
                    lowres.shape[1] - 1,
                    lowres.shape[2] - 1,
                    lowres.shape[3] - 1,
                    ((padding*2)+2)**3,
                    *lowres.shape[4:]
                ]))

    def test_chunk_from_lowres(self):
        """
        Test we can extract chunks from a lowres volume, padding should use real pixels from the rest of the volume
        where available and use symmetric padding where not.
        """

        for z, y, x, chunk, padding in product([0, 1, 7], [0, 1, 7], [0, 1, 7], [2, 7, 10], range(4)):
            with self.subTest(z=z, y=y, x=x, chunk=chunk, padding=padding):

                # Get a dummy lowres volume to extract features from
                lowres = kom.volume.lowres_from_highres(self.dummy_highres())

                # Extract the desired chunk with the requested padding
                chunk_lowres = kom.volume.chunk_from_lowres(lowres,
                                                            z=(z, min(lowres.shape[1], z+chunk)),
                                                            y=(y, min(lowres.shape[2], y+chunk)),
                                                            x=(x, min(lowres.shape[3], x+chunk)),
                                                            padding=padding)

                # Check the extract features have the correct shape and dtype
                self.assertEqual(chunk_lowres.dtype, lowres.dtype)
                self.assertEqual(chunk_lowres.ndim, lowres.ndim)
                self.assertTrue(np.allclose(chunk_lowres.shape, [
                    lowres.shape[0],
                    (min(lowres.shape[1], z + chunk) - z) + (padding * 2),
                    (min(lowres.shape[2], y + chunk) - y) + (padding * 2),
                    (min(lowres.shape[3], x + chunk) - x) + (padding * 2),
                    *lowres.shape[4:]
                ]))
