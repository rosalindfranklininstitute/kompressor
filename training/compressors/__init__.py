"""Helper functions to store and get the selected model"""
from collections import namedtuple
from training.utils import named_module

_current = None
AVAILABLE = ['haiku']
Compressor = namedtuple('Compressor', ['name', "Compressor"])


def use_compressor(compressor):
    compressor = compressor.lower()
    global _current
    if compressor in AVAILABLE:
        module_ = named_module('training.compressors.{}'.format(compressor))
        if hasattr(module_, '_compressor'):
            _current = module_._compressor
        else:
            raise Exception(
                'Module `{}` is not a proper model.'.format(compressor))
    else:
        raise Exception('Model `{}` not available! Choose from: {}'
                        .format(compressor, AVAILABLE))


def get_compressor(name=None):
    if name is not None:
        use_compressor(name)
    if _current is None:
        use_compressor(AVAILABLE[0])
    return _current
