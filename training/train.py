import argparse
from callbacks.metrics import MetricsCallback
from kompressors.haiku import HaikuKompressor
from models import mlp, cnn
from predictors import regression


def __option_parser():
    """
    Option parser for command line arguments.
    """
    parser = argparse.ArgumentParser(description="Setup training")
    parser.add_argument(
        "-l", "--level", type=int, default=1, help=""
    )
