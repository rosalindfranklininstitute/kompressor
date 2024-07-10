import yaml
import os


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return open(arg, 'r')  # return an open file handle


def get_config(path):
    """
    Returns the config dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config
