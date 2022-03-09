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


class VolumeEncodeDecodeTest(unittest.TestCase):

    def dummy_highres(self, shape=(2, 17, 17, 17, 1), max_value=65536, dtype=jnp.uint16):
        highres = (jnp.arange(np.prod(shape)).reshape(shape) % max_value).astype(dtype)
        return highres

    def dummy_predictions_fn(self, padding):

        # Dummy regression predictor function just predicts the average of the neighborhood features
        def predictions_fn(lowres):
            # Extract the features for each neighborhood
            features = kom.volume.features_from_lowres(lowres, padding)
            # Dummy predictions are just the average values of the neighborhoods
            predictions = jnp.mean(features.astype(jnp.float32), axis=4, keepdims=True).astype(lowres.dtype)
            predictions = jnp.repeat(predictions, repeats=19, axis=4)
            # Extract the maps from the predictions
            return kom.volume.maps_from_predictions(predictions)

        return predictions_fn

    def dummy_predictions_categorical_fn(self, padding, classes):

        # Dummy categorical predictor function just predicts a constant set of logits
        def predictions_fn(lowres):
            # Sample a constant set of random logits (for test consistency)
            key = jax.random.PRNGKey(1234)
            logits = jax.random.uniform(key, (1, 1, 1, 1, 19, *lowres.shape[4:], classes))
            # Tile the same logits for every voxel and batch element
            predictions = jnp.tile(logits, (lowres.shape[0],
                                            (lowres.shape[1]-1) - (padding*2),
                                            (lowres.shape[2]-1) - (padding*2),
                                            (lowres.shape[3]-1) - (padding*2),
                                            1, *([1] * len(lowres.shape[4:])), 1))
            # Extract the maps from the predictions
            pred_maps = kom.volume.maps_from_predictions(predictions)
            # Convert the predictions to dummy logits (one hot encodings)
            return jax.tree_map(partial(jax.nn.softmax, axis=-1), pred_maps)

        return predictions_fn

    def test_encode_decode(self):
        """
        Test we can do an encode + decode cycle on an volume processing the whole input at once using different paddings.
        """

        for padding in range(2):
            with self.subTest('Test with odd dimensions',
                              padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.mapping.uint16.encode_values
                decode_fn = kom.mapping.uint16.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres()

                # Encode the entire volume at once
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                         padding=padding)

                # Check that even padding was applied correctly
                ed, eh, ew = dims
                self.assertEqual(ed, 0)
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)

                # Check that the lowres and maps are the correct sizes and dtypes
                self.assertEqual(len(maps), 7)
                lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps['lrmap'], maps['udmap'], maps['fbmap'], maps['cmap'], \
                                                              maps['zmap'], maps['ymap'], maps['xmap']

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

                # Decode the entire volume at once
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))
                
        for padding in range(2):
            with self.subTest('Test with even dimensions',
                              padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.mapping.uint16.encode_values
                decode_fn = kom.mapping.uint16.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(shape=(2, 16, 16, 16, 1))

                # Encode the entire volume at once
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                         padding=padding)

                # Check that even padding was applied correctly
                ed, eh, ew = dims
                self.assertEqual(ed, 1)
                self.assertEqual(eh, 1)
                self.assertEqual(ew, 1)

                # Decode the entire volume at once
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode_categorical(self):
        """
        Test we can do an encode + decode cycle on an volume processing the whole input at once using different
        paddings using a categorical predictor and encoding.
        """

        for padding in range(2):
            with self.subTest('Test with odd dimensions',
                              padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.mapping.categorical.encode_values
                decode_fn = kom.mapping.categorical.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(max_value=256, dtype=jnp.uint8)

                # Encode the entire volume at once
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                         padding=padding)

                # Check that even padding was applied correctly
                ed, eh, ew = dims
                self.assertEqual(ed, 0)
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)

                # Check that the lowres and maps are the correct sizes and dtypes
                self.assertEqual(len(maps), 7)
                lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps['lrmap'], maps['udmap'], maps['fbmap'], maps['cmap'], \
                                                              maps['zmap'], maps['ymap'], maps['xmap']

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

                # Decode the entire volume at once
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))
                
        for padding in range(2):
            with self.subTest('Test with even dimensions',
                              padding=padding):

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.mapping.categorical.encode_values
                decode_fn = kom.mapping.categorical.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(shape=(2, 16, 16, 16, 1), max_value=256, dtype=jnp.uint8)

                # Encode the entire volume at once
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                         padding=padding)

                # Check that even padding was applied correctly
                ed, eh, ew = dims
                self.assertEqual(ed, 1)
                self.assertEqual(eh, 1)
                self.assertEqual(ew, 1)

                # Decode the entire volume at once
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_decode_raw(self):
        """
        Test we can do an encode + decode cycle on an volume processing the whole input at once using different
        paddings using a regression predictor and raw encoding.
        """

        padding = 0

        # Make a prediction function for this test
        predictions_fn = self.dummy_predictions_fn(padding=padding)
        encode_fn = kom.mapping.raw.encode_values
        decode_fn = kom.mapping.raw.decode_values

        # Get a dummy highres volume to encode + decode
        highres = jnp.int32(self.dummy_highres())

        # Encode the entire volume at once
        lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                 padding=padding)

        # Check that even padding was applied correctly
        ed, eh, ew = dims
        self.assertEqual(ed, 0)
        self.assertEqual(eh, 0)
        self.assertEqual(ew, 0)

        # Check that the lowres and maps are the correct sizes and dtypes
        self.assertEqual(len(maps), 7)
        lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps['lrmap'], maps['udmap'], maps['fbmap'], maps['cmap'], \
                                                      maps['zmap'], maps['ymap'], maps['xmap']

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

        # Decode the entire volume at once
        reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                  padding=padding)

        # Check the decoded volume is lossless
        self.assertEqual(reconstructed_highres.dtype, highres.dtype)
        self.assertEqual(reconstructed_highres.ndim, highres.ndim)
        self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_chunks(self):
        """
        Test we can encode a volume processing the input in chunks and with different paddings.
        """

        for encode_chunk, padding in product([6, 11, (6, 11, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              encode_chunk=encode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.volume.encode_chunks '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.mapping.uint16.encode_values
                decode_fn = kom.mapping.uint16.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                                        padding=padding)
                self.assertEqual(len(full_maps), 7)
                full_lrmap, full_udmap, full_fbmap, full_cmap, full_zmap, full_ymap, full_xmap = \
                    full_maps['lrmap'], full_maps['udmap'], full_maps['fbmap'], full_maps['cmap'], \
                    full_maps['zmap'],  full_maps['ymap'], full_maps['xmap']

                # Encode the input in chunks
                lowres, (maps, dims) = kom.volume.encode_chunks(predictions_fn, encode_fn, highres,
                                                                chunk=encode_chunk, padding=padding,
                                                                progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (ed, eh, ew), (full_ed, full_eh, full_ew) = dims, full_dims
                self.assertEqual(ed, 0)
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)
                self.assertEqual(ed, full_ed)
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Check that processing in chunks gives the same results as processing all at once
                self.assertEqual(len(maps), 7)
                lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps['lrmap'], maps['udmap'], maps['fbmap'], maps['cmap'], \
                                                              maps['zmap'], maps['ymap'], maps['xmap']

                self.assertEqual(lowres.dtype, full_lowres.dtype)
                self.assertEqual(lowres.ndim, full_lowres.ndim)
                self.assertTrue(np.allclose(lowres, full_lowres))

                self.assertEqual(lrmap.dtype, full_lrmap.dtype)
                self.assertEqual(lrmap.ndim, full_lrmap.ndim)
                self.assertTrue(np.allclose(lrmap, full_lrmap))

                self.assertEqual(udmap.dtype, full_udmap.dtype)
                self.assertEqual(udmap.ndim, full_udmap.ndim)
                self.assertTrue(np.allclose(udmap, full_udmap))

                self.assertEqual(fbmap.dtype, full_fbmap.dtype)
                self.assertEqual(fbmap.ndim, full_fbmap.ndim)
                self.assertTrue(np.allclose(fbmap, full_fbmap))

                self.assertEqual(cmap.dtype, full_cmap.dtype)
                self.assertEqual(cmap.ndim, full_cmap.ndim)
                self.assertTrue(np.allclose(cmap, full_cmap))

                self.assertEqual(zmap.dtype, full_zmap.dtype)
                self.assertEqual(zmap.ndim, full_zmap.ndim)
                self.assertTrue(np.allclose(zmap, full_zmap))

                self.assertEqual(ymap.dtype, full_ymap.dtype)
                self.assertEqual(ymap.ndim, full_ymap.ndim)
                self.assertTrue(np.allclose(ymap, full_ymap))

                self.assertEqual(xmap.dtype, full_xmap.dtype)
                self.assertEqual(xmap.ndim, full_xmap.ndim)
                self.assertTrue(np.allclose(xmap, full_xmap))

                # Decode the volume in one pass
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))
                
        for encode_chunk, padding in product([6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              encode_chunk=encode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.volume.encode_chunks '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.mapping.uint16.encode_values
                decode_fn = kom.mapping.uint16.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(shape=(2, 16, 16, 16, 1))

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                                        padding=padding)

                # Encode the input in chunks
                lowres, (maps, dims) = kom.volume.encode_chunks(predictions_fn, encode_fn, highres,
                                                                chunk=encode_chunk, padding=padding,
                                                                progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (ed, eh, ew), (full_ed, full_eh, full_ew) = dims, full_dims
                self.assertEqual(ed, full_ed)
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Decode the volume in one pass
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_decode_chunks(self):
        """
        Test we can decode a volume processing the input in chunks and with different paddings.
        """

        for decode_chunk, padding in product([6, 11, (6, 11, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.volume.decode_chunks '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.mapping.uint16.encode_values
                decode_fn = kom.mapping.uint16.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres()

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres, padding=padding)

                # Decode the volume in chunks
                reconstructed_highres = kom.volume.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                 chunk=decode_chunk, padding=padding,
                                                                 progress_fn=decode_progress_fn)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))
                
        for decode_chunk, padding in product([6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.volume.decode_chunks '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_fn(padding=padding)
                encode_fn = kom.mapping.uint16.encode_values
                decode_fn = kom.mapping.uint16.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(shape=(2, 16, 16, 16, 1))

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                         padding=padding)

                # Decode the volume in chunks
                reconstructed_highres = kom.volume.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                 chunk=decode_chunk, padding=padding,
                                                                 progress_fn=decode_progress_fn)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_encode_chunks_categorical(self):
        """
        Test we can encode a volume processing the input in chunks and with different paddings
        with a categorical predictions function.
        """

        for encode_chunk, padding in product([6, 11, (6, 11, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              encode_chunk=encode_chunk, padding=padding):

                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.volume.encode_chunks_categorical '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.mapping.categorical.encode_values
                decode_fn = kom.mapping.categorical.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(max_value=256, dtype=jnp.uint8)

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                                        padding=padding)
                self.assertEqual(len(full_maps), 7)
                full_lrmap, full_udmap, full_fbmap, full_cmap, full_zmap, full_ymap, full_xmap = \
                    full_maps['lrmap'], full_maps['udmap'], full_maps['fbmap'], full_maps['cmap'], \
                    full_maps['zmap'], full_maps['ymap'], full_maps['xmap']

                # Encode the input in chunks
                lowres, (maps, dims) = kom.volume.encode_chunks(predictions_fn, encode_fn, highres,
                                                                chunk=encode_chunk, padding=padding,
                                                                progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (ed, eh, ew), (full_ed, full_eh, full_ew) = dims, full_dims
                self.assertEqual(ed, 0)
                self.assertEqual(eh, 0)
                self.assertEqual(ew, 0)
                self.assertEqual(ed, full_ed)
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Check that processing in chunks gives the same results as processing all at once
                self.assertEqual(len(maps), 7)
                lrmap, udmap, fbmap, cmap, zmap, ymap, xmap = maps['lrmap'], maps['udmap'], maps['fbmap'], maps['cmap'], \
                                                              maps['zmap'], maps['ymap'], maps['xmap']

                self.assertEqual(lowres.dtype, full_lowres.dtype)
                self.assertEqual(lowres.ndim, full_lowres.ndim)
                self.assertTrue(np.allclose(lowres, full_lowres))

                self.assertEqual(lrmap.dtype, full_lrmap.dtype)
                self.assertEqual(lrmap.ndim, full_lrmap.ndim)
                self.assertTrue(np.allclose(lrmap, full_lrmap))

                self.assertEqual(udmap.dtype, full_udmap.dtype)
                self.assertEqual(udmap.ndim, full_udmap.ndim)
                self.assertTrue(np.allclose(udmap, full_udmap))

                self.assertEqual(fbmap.dtype, full_fbmap.dtype)
                self.assertEqual(fbmap.ndim, full_fbmap.ndim)
                self.assertTrue(np.allclose(fbmap, full_fbmap))

                self.assertEqual(cmap.dtype, full_cmap.dtype)
                self.assertEqual(cmap.ndim, full_cmap.ndim)
                self.assertTrue(np.allclose(cmap, full_cmap))

                self.assertEqual(zmap.dtype, full_zmap.dtype)
                self.assertEqual(zmap.ndim, full_zmap.ndim)
                self.assertTrue(np.allclose(zmap, full_zmap))

                self.assertEqual(ymap.dtype, full_ymap.dtype)
                self.assertEqual(ymap.ndim, full_ymap.ndim)
                self.assertTrue(np.allclose(ymap, full_ymap))

                self.assertEqual(xmap.dtype, full_xmap.dtype)
                self.assertEqual(xmap.ndim, full_xmap.ndim)
                self.assertTrue(np.allclose(xmap, full_xmap))

                # Decode the volume in one pass
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for encode_chunk, padding in product([6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              encode_chunk=encode_chunk, padding=padding):
                # Make logging functions for this test
                encode_progress_fn = partial(tqdm, desc=f'kom.volume.encode_chunks_categorical '
                                                        f'encode_chunk={encode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.mapping.categorical.encode_values
                decode_fn = kom.mapping.categorical.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(shape=(2, 16, 16, 16, 1), max_value=256, dtype=jnp.uint8)

                # Encode the entire input at once to check for consistency
                full_lowres, (full_maps, full_dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                                        padding=padding)

                # Encode the input in chunks
                lowres, (maps, dims) = kom.volume.encode_chunks(predictions_fn, encode_fn, highres,
                                                                chunk=encode_chunk, padding=padding,
                                                                progress_fn=encode_progress_fn)

                # Check that even padding was applied correctly
                (ed, eh, ew), (full_ed, full_eh, full_ew) = dims, full_dims
                self.assertEqual(ed, full_ed)
                self.assertEqual(eh, full_eh)
                self.assertEqual(ew, full_ew)

                # Decode the volume in one pass
                reconstructed_highres = kom.volume.decode(predictions_fn, decode_fn, lowres, (maps, dims),
                                                          padding=padding)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

    def test_decode_chunks_categorical(self):
        """
        Test we can decode a volume processing the input in chunks and with different paddings
        with a categorical predictions function.
        """

        for decode_chunk, padding in product([6, 11, (6, 11, 11)], range(2)):
            with self.subTest('Test with odd dimensions',
                              decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.volume.decode_chunks_categorical '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.mapping.categorical.encode_values
                decode_fn = kom.mapping.categorical.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(max_value=256, dtype=jnp.uint8)

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres, padding=padding)

                # Decode the volume in chunks
                reconstructed_highres = kom.volume.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                 chunk=decode_chunk, padding=padding,
                                                                 progress_fn=decode_progress_fn)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))

        for decode_chunk, padding in product([6, 11], range(2)):
            with self.subTest('Test with even dimensions',
                              decode_chunk=decode_chunk, padding=padding):

                # Make logging functions for this test
                decode_progress_fn = partial(tqdm, desc=f'kom.volume.decode_chunks_categorical '
                                                        f'decode_chunk={decode_chunk}, padding={padding}')

                # Make a prediction function for this test
                predictions_fn = self.dummy_predictions_categorical_fn(padding=padding, classes=256)
                encode_fn = kom.mapping.categorical.encode_values
                decode_fn = kom.mapping.categorical.decode_values

                # Get a dummy highres volume to encode + decode
                highres = self.dummy_highres(shape=(2, 16, 16, 16, 1), max_value=256, dtype=jnp.uint8)

                # Encode the entire input at once to check for consistency
                lowres, (maps, dims) = kom.volume.encode(predictions_fn, encode_fn, highres,
                                                         padding=padding)

                # Decode the volume in chunks
                reconstructed_highres = kom.volume.decode_chunks(predictions_fn, decode_fn, lowres, (maps, dims),
                                                                 chunk=decode_chunk, padding=padding,
                                                                 progress_fn=decode_progress_fn)

                # Check the decoded volume is lossless
                self.assertEqual(reconstructed_highres.dtype, highres.dtype)
                self.assertEqual(reconstructed_highres.ndim, highres.ndim)
                self.assertTrue(np.allclose(reconstructed_highres, highres))
