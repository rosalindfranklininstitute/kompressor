"""Helper functions to store and get the selected model"""
from collections import namedtuple
from kompressor.training.utils import named_module

_current = None
AVAILABLE = ['cnn', 'mlp']
Model = namedtuple('Model', ['name', "Model"])

def use_model(model):
    model = model.lower()
    global _current
    if model in AVAILABLE:
        module_ = named_module('kompressor.training.models.{}'.format(model))
        if hasattr(module_, '_model'):
            _current = module_._model
        else:
            raise Exception(
                'Module `{}` is not a proper model.'.format(model))
    else:
        raise Exception('Model `{}` not available! Choose from: {}'
                        .format(model, AVAILABLE))


def get_model(name=None):
    if name is not None:
        use_model(name)
    if _current is None:
        use_model(AVAILABLE[0])
    return _current
