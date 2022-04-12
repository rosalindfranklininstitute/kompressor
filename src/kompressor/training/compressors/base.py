from abc import abstractmethod, ABC
import kompressor as kom


class BaseCompressor(ABC):

    def __init__(self, encode_fn, decode_fn, padding):
        self.encode_fn, self.decode_fn = encode_fn, decode_fn
        self.padding = padding

    @abstractmethod
    def _predictions_fn(self):
        raise NotImplementedError()

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, file_name):
        raise NotImplementedError()

    @abstractmethod
    def load_model(self, file_name):
        raise NotImplementedError()

    def encode(self, highres, levels=1, chunk=None, progress_fn=None, debug=False):

        assert levels > 0

        predictions_fn = self._predictions_fn()

        maps = list()
        for level in range(levels):

            if chunk is None:
                lowres, maps_dims = kom.image.encode(predictions_fn, self.encode_fn, highres, padding=self.padding)
            else:
                lowres, maps_dims = kom.image.encode_chunks(predictions_fn, self.encode_fn, highres,
                                                            padding=self.padding, chunk=chunk, progress_fn=progress_fn)

            if debug:
                maps.append((lowres, maps_dims, highres))
            else:
                maps.append(maps_dims)

            highres = lowres

        return lowres, maps

    def decode(self, lowres, maps, chunk=None, progress_fn=None, debug=False):

        assert len(maps) > 0

        predictions_fn = self._predictions_fn()

        for maps_dims in reversed(maps):

            if debug:
                _, maps_dims, _ = maps_dims

            if chunk is None:
                highres = kom.image.decode(predictions_fn, self.decode_fn, lowres, maps_dims, padding=self.padding)
            else:
                highres = kom.image.decode_chunks(predictions_fn, self.decode_fn, lowres, maps_dims,
                                                  padding=self.padding, chunk=chunk, progress_fn=progress_fn)

            lowres = highres

        return highres