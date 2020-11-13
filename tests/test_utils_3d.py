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
import jax.numpy as jnp
import kompressor as kom


class Utils3DTest(unittest.TestCase):

    def dummy_highres(self):
        shape = (2, 5, 5, 5, 3)
        highres = jnp.arange(np.prod(shape)).reshape(shape)
        return highres

    def test_neighbor_enum(self):

        neighbors = [
            kom.utils_3d.Neighbors.L,
            kom.utils_3d.Neighbors.R,
            kom.utils_3d.Neighbors.U,
            kom.utils_3d.Neighbors.D,
            kom.utils_3d.Neighbors.F,
            kom.utils_3d.Neighbors.B,
            kom.utils_3d.Neighbors.C,
            kom.utils_3d.Neighbors.Z0,
            kom.utils_3d.Neighbors.Z1,
            kom.utils_3d.Neighbors.Z2,
            kom.utils_3d.Neighbors.Z3,
            kom.utils_3d.Neighbors.Y0,
            kom.utils_3d.Neighbors.Y1,
            kom.utils_3d.Neighbors.Y2,
            kom.utils_3d.Neighbors.Y3,
            kom.utils_3d.Neighbors.X0,
            kom.utils_3d.Neighbors.X1,
            kom.utils_3d.Neighbors.X2,
            kom.utils_3d.Neighbors.X3
        ]

        self.assertTrue(np.allclose(neighbors, np.arange(len(neighbors))))

    def test_targets_from_highres(self):

        highres = self.dummy_highres()

        targets = kom.utils_3d.targets_from_highres(highres)

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

        highres = self.dummy_highres()

        lowres = kom.utils_3d.lowres_from_highres(highres)

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

        highres = self.dummy_highres()

        predictions = kom.utils_3d.targets_from_highres(highres)

        lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = kom.utils_3d.maps_from_predictions(predictions)

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

        highres = self.dummy_highres()

        lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = kom.utils_3d.maps_from_highres(highres)

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

        highres = self.dummy_highres()
        lowres  = kom.utils_3d.lowres_from_highres(highres)
        maps    = kom.utils_3d.maps_from_highres(highres)

        reconstructed_highres = kom.utils_3d.highres_from_lowres_and_maps(lowres, *maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

        highres     = self.dummy_highres()
        lowres      = kom.utils_3d.lowres_from_highres(highres)
        predictions = kom.utils_3d.targets_from_highres(highres)
        maps        = kom.utils_3d.maps_from_predictions(predictions)

        reconstructed_highres = kom.utils_3d.highres_from_lowres_and_maps(lowres, *maps)

        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))
