from abc import abstractmethod, ABCMeta


class Callback(ABCMeta):

    @abstractmethod
    def on_step_start(self, *args, **kargs):
        pass

    @abstractmethod
    def on_step_end(self, *args, **kargs):
        pass