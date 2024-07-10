import typing
from pathlib import Path
from typing import List, Tuple

import mrcfile
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

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
        If we have an MRC file shape 1,2,3 at /tmp/0.mrc then::

          data = get_data_paths_and_frames("/tmp/0.mrc")

        data will be:
          [("/tmp/0.mrc",0), ("/tmp/0.mrc",1),("/tmp/0.mrc",2)]
    """
    data_paths = []
    for file in files:
        assert Path(file).is_file(), f"{file} is not a file."
        frames = np.max(mrcfile.mmap(file).data.shape)
        for frame in range(frames):
            data_paths.append((file, frame))
    return data_paths


class MRCFileDataset(Dataset):
    def __init__(
        self,
        dataset: List[str],
        transform: typing.Optional[torchvision.transforms.Compose] = None,
    ):
        self.dataset = dataset

        self.data_file = mrcfile.mmap(dataset[0][0], mode="r")
        self.transform = transform

    def _decode_mrc_data_path(self, idx: int) -> np.array:
        """
        Decode the MRC data path returning the slice of the data speficifed.

        Args:
            idx: The index of the MRC frame to decode and get

        Returns: A Numpy array containing the decoded data along with an additional
        axis to create Height, Width, Channel.

        Examples:
            If we have a Data path array of ("/tmp/0.mrc",0) with shape (5,5) this will
            return the numpy array with the shape (5,5,1)
        """
        frame_index = np.argmax(self.data_file.data.shape)
        if len(self.data_file.data.shape) == 3:
            if frame_index == 0:
                data = np.asarray(self.data_file.data[self.dataset[idx][1], ...])
            elif frame_index == 1:
                data = np.asarray(self.data_file.data[:, self.dataset[idx][1], ...])
            elif frame_index == 2:
                data = np.asarray(self.data_file.data[..., self.dataset[idx][1]])
        return np.expand_dims(data, axis=data.ndim)

    def __len__(self) -> int:
        """Get the length of the dataset.
        Returns:
            int: Length of the dataset
        """
        return len(self.dataset)

    def __getitem__(
        self, idx: typing.Union[int, slice]
    ) -> np.ndarray[typing.Literal[4]]:
        """Get a sample from the dataset2 at idx position.
        Args:
            idx: idx to get the sample.
        Return:
            Frame with Shape: Height,Width,Channel
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self._decode_mrc_data_path(idx)
        if self.transform:
            frame = self.transform(frame)
        return frame
