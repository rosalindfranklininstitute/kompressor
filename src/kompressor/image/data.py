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


import tensorflow as tf


from .utils import \
    validate_chunk


def transform_extract_level_from_highres(level):

    assert level >= 0
    skip = 2 ** level

    def fn(highres):
        # Downsample by skip sampling
        highres = highres[::skip, ::skip]

        # Determine the size of the input and the padding to apply
        ph, pw = (tf.shape(highres)[0] + 1) % 2, (tf.shape(highres)[1] + 1) % 2

        # Pad highres using reflect to match lowres padded with symmetric
        data_padding = ((0, 0),) * (tf.rank(highres) - 2)
        return tf.pad(highres, ((0, ph), (0, pw), *data_padding), mode='REFLECT')

    return fn


def transform_random_crop_highres(chunk):

    # Assert chunk size is valid
    ch, cw = validate_chunk(chunk)

    def fn(highres):
        return tf.image.random_crop(highres, (ch, cw, 3))

    return fn


def transform_lowres_and_targets_from_highres(padding):

    assert padding >= 0

    def fn(highres):
        highres = tf.expand_dims(highres, axis=0)

        # Downsample by skip sampling
        lowres = highres[:, ::2, ::2]

        # Pad only the 2 spatial dimensions
        spatial_padding = ((padding, padding),) * 2
        data_padding = ((0, 0),) * (tf.rank(lowres) - 3)
        lowres = tf.pad(lowres, ((0, 0), *spatial_padding, *data_padding), mode='SYMMETRIC')

        # Slice out each value of the pluses
        lmap = highres[:,  1::2, :-1:2]
        rmap = highres[:,  1::2,  2::2]
        umap = highres[:, :-1:2,  1::2]
        dmap = highres[:,  2::2,  1::2]
        cmap = highres[:,  1::2,  1::2]

        # Stack the vectors LRUDC order with dim [B,H,W,5,...]
        targets = tf.stack([lmap, rmap, umap, dmap, cmap], axis=3)

        return dict(lowres=tf.cast(lowres[0], tf.float32) / 256.,
                    targets=tf.cast(targets[0], tf.float32) / 256.)

    return fn


def random_chunk_dataset(dataset, padding=0, chunk=64, chunks_per_sample=1, chunks_shuffle_buffer=None, levels=1):

    assert padding >= 0
    assert levels > 0

    # Construct separate datasets for each level requested, each dataset outputs the same chunk size
    ds_levels = list()
    for level in range(levels):

        # From the input dataset of consistently sized highres images for level=0
        ds = dataset

        # Extract the skip-sampled highres image for the given level and apply padding if needed
        ds = ds.map(transform_extract_level_from_highres(level), num_parallel_calls=tf.data.AUTOTUNE)

        # Repeat the same image multiple times to allow different random chunks to be sampled from it
        if chunks_per_sample > 1:
            ds = ds.flat_map(lambda highres: tf.data.Dataset.from_tensors(highres).repeat(chunks_per_sample))

        # Extract a random chunk from the highres image
        ds = ds.map(transform_random_crop_highres(chunk), num_parallel_calls=tf.data.AUTOTUNE)

        ds_levels.append(ds)

    # Interleave the streams of skip-sampled, padded, and cropped highres images from each level
    ds = tf.data.Dataset.sample_from_datasets(ds_levels)

    # Shuffle the stream of highres images from each level
    if (chunks_shuffle_buffer is not None) and (chunks_shuffle_buffer > 0):
        ds = ds.shuffle(chunks_shuffle_buffer, reshuffle_each_iteration=True)

    # From the highres images extract the lowres inputs and prediction targets
    ds = ds.map(transform_lowres_and_targets_from_highres(padding), num_parallel_calls=tf.data.AUTOTUNE)

    return ds
