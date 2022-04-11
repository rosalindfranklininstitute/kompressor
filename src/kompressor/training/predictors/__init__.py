"""Helper functions to store and get the selected predictor """
from collections import namedtuple
from kompressor.training.utils import named_module

_current = None
AVAILABLE = ['regression']
Predictor = namedtuple('Predictor', ['name', "Predictor"])

def use_predictor(predictor):
    predictor = predictor.lower()
    global _current
    if predictor in AVAILABLE:
        module_ = named_module('kompressor.training.predictors.{}'.format(predictor))
        if hasattr(module_, '_predictor'):
            _current = module_._predictor
        else:
            raise Exception(
                'Module `{}` is not a proper model.'.format(predictor))
    else:
        raise Exception('Model `{}` not available! Choose from: {}'
                        .format(predictor, AVAILABLE))


def get_predictor(name=None):
    if name is not None:
        use_predictor(name)
    if _current is None:
        use_predictor(AVAILABLE[0])
    return _current
