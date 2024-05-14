import mrcfile
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np

Paths = str
Frame = int
DataPathFrame = Tuple[Paths, Frame]


def get_data_paths_and_frames(files: List[str]) -> List[DataPathFrame]:
    """Gets the data paths and frames from the list of files provided.

    Args:
      files: The files to be compressed.

    Returns:
        A list of paths and frames.

    Example:
        If we have an MRC file file shape 1,2,3 at /tmp/0.mrc then::

          data = get_data_paths_and_frames("/tmp/0.mrc)

        data will be:
          [("/tmp/0.mrc",0), ("/tmp/0.mrc",1),("/tmp/0.mrc",2)]
    """
    data_paths = []
    for file in files:
        file = Path(file)
        assert file.is_file(), f"{file} is not a file."
        for frame in range(mrcfile.mmap(file).data.shape[-1]):
            data_paths.append((file, frame))
    return data_paths


def decode_mrc_data_path(data_paths: DataPathFrame) -> np.array:
    """
    Decode the MRC data path returning the slice of the data speficifed.

    Args:
        data_paths: A List containing the [Path, Frame] tuples

    Returns:
        A Numpy array containing the decoded data along with an additional axis to create Height, Width, Channel.

    Examples:
        If we have an Data path array of ("/tmp/0.mrc",0) with shape (5,5) this will
        return the numpy array with the shape (5,5,1)
    """
    data = np.asarray(mrcfile.mmap(data_paths[0]).data[..., data_paths[1]])
    if data.ndim == 2:
        return np.expand_dims(data, axis=-1)
    return data
