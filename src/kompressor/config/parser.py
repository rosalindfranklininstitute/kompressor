import yaml
import os


def is_valid_file(parser, file_path):
    """
    Check if valid file is passed.
    Args:
        parser: argument parser
        file_path: path to check

    Returns:
        File path if it exists.
    """
    if not os.path.exists(file_path):
        parser.error(f"File {file_path} does not exist. Please provide a valid path.")
    else:
        return file_path


def get_config(path):
    """
    Returns the config dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config
