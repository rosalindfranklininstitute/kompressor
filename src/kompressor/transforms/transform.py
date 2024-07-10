from typing import Optional, Tuple, Union
import numpy as np


def validate_chunk(chunk):
    # Assert valid chunk size
    if isinstance(chunk, int):
        assert chunk > 3
        ch, cw = (chunk,) * 2
    elif isinstance(chunk, tuple):
        ch, cw = chunk
        assert ch > 3
        assert cw > 3
    else:
        raise AssertionError("chunk must be int or tuple(int, int)")
    return ch, cw


class RandomChunk(object):
    def __init__(self, chunk: Union[int, tuple]):
        self.chunk_height, self.chunk_width = validate_chunk(chunk)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: frame of data to take chunks out of. frame Shape ["Height", "Width", "Channels"]
        Returns:
        """
        y = np.random.randint(0, frame.shape[0] - self.chunk_height)
        x = np.random.randint(0, frame.shape[1] - self.chunk_width)
        chunk_frame = frame[y : y + self.chunk_height, x : x + self.chunk_width, :]
        return chunk_frame


class ExtractLevelFromHighres(object):
    """Extract Downsampled Low-Resolution batch from a high-resolution batch."""

    def __init__(self, level: int):
        """Initializes a new ExtractLevelFromHighre
        Args:

            level: level to downsample: results in a skip size of 2**level, i.e. an
            image that is 128x128 with level=2,3,4 the skip size is 4,8,16
            resulting image sizes of 32x32,16x16,8x8 respectively
        """
        assert level >= 0

        self.level = level
        self.skip = 2**level

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Batch to downsample from a high-resolution image.
        Args:
            frame: High resolution batch to downsample. Shape [height, width, channel]
        Return:
            np.ndarray: Downsampled batch: Shape [height, width, channel]
        """
        highres_frame = frame
        # Downsample by skip sampling
        lowres_frame = highres_frame[:: self.skip, :: self.skip]
        ph, pw = (np.shape(lowres_frame)[0] + 1) % 2, (
            np.shape(lowres_frame)[1] + 1
        ) % 2
        lowres_frame = np.pad(lowres_frame, ((0, ph), (0, pw), (0, 0)), mode="reflect")

        return lowres_frame


class LowresAndTargetsFromHighres(object):
    """Extract Low-Resolution and Targets from a high-resolution."""

    def __init__(self, padding: int, bit_depth: int):
        assert padding >= 0
        self.padding = padding
        self.bit_depth = bit_depth

    def __call__(self, highres) -> dict:
        """Downsample from a high-resolution.

        Args: highres: high-resolution images to downsample
        in the form [height, width, channels]
        """

        # Downsample by skip sampling
        lowres = highres[::2, ::2, :]

        # Pad the 2 spatial dimensions Height and Width
        lowres = np.pad(
            lowres,
            (
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
            mode="symmetric",
        )

        # Slice out each value of the pluses
        lmap = highres[1::2, :-1:2, :]
        rmap = highres[1::2, 2::2, :]
        umap = highres[:-1:2, 1::2, :]
        dmap = highres[2::2, 1::2, :]
        cmap = highres[1::2, 1::2, :]

        # Stack the vectors LRUDC order with dim [H,W,5,...]
        targets = np.stack([lmap, rmap, umap, dmap, cmap], axis=2)
        lowres[0] = lowres[0].astype(np.float32) / np.float32(2**self.bit_depth)
        targets[0] = targets[0].astype(np.float32) / np.float32(2**self.bit_depth)
        return dict(lowres=lowres, targets=targets)


class RandomChunkDataset(object):
    def __init__(
        self,
        padding,
        chunk_size: Optional[Tuple[int, ...]] = None,
        number_of_chunks: int = 1,
        bit_depth: int = 16,
        levels: int = 0,
    ):
        assert padding >= 0
        assert levels >= 0

        self.padding = padding
        self.chunk_size = chunk_size
        self.number_of_chunks = number_of_chunks
        self.bit_depth = bit_depth
        self.levels = levels

    def __call__(self, batch):
        ds = batch
        ds = ExtractLevelFromHighres(self.levels)(ds)

        # Generate Random Chunks
        if self.chunk_size:
            ds = RandomChunk(self.chunk_size)(ds)

        return LowresAndTargetsFromHighres(self.padding, self.bit_depth)(ds)
