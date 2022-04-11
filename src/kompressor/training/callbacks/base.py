from abc import abstractmethod, ABC


class Callback(ABC):

    def on_step_start(self, *args, **kargs):
        pass

    @abstractmethod
    def on_step_end(self, *args, **kargs):
        pass