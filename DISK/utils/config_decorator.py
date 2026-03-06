import yaml
import argparse
from typing import List, TypeVar, Generic
import os

# Define a type variable to represent the generic type
I = TypeVar('I', bound=int)

class IntList(list, Generic[I]):
    """A custom list that only allows strings."""

    def __init__(self, *args: I) -> None:
        # Call the parent list's constructor with the filtered strings
        super().__init__(s for s in args if isinstance(s, int))

    def append(self, item: I) -> None:
        if not isinstance(item, int):
            raise TypeError("Only strings can be added to IntList")
        super().append(item)

    def extend(self, iterable) -> None:
        if not all(isinstance(item, int) for item in iterable):
            raise TypeError("Only strings can be added to IntList")
        super().extend(iterable)

    def __repr__(self) -> str:
        return f'IntList({super().__repr__()})'

# Define a type variable to represent the generic type
S = TypeVar('S', bound=str)

class StringList(list, Generic[S]):
    """A custom list that only allows strings."""

    def __init__(self, *args: S) -> None:
        # Call the parent list's constructor with the filtered strings
        super().__init__(s for s in args if isinstance(s, str))

    def append(self, item: S) -> None:
        if not isinstance(item, str):
            raise TypeError("Only strings can be added to StringList")
        super().append(item)

    def extend(self, iterable) -> None:
        if not all(isinstance(item, str) for item in iterable):
            raise TypeError("Only strings can be added to StringList")
        super().extend(iterable)

    def __repr__(self) -> str:
        return f'StringList({super().__repr__()})'

# Read the configuration
def read_config(file_path):
    # config = configparser.ConfigParser()
    # config.read(file_path)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_directory, file_path), 'r') as file:
        config = yaml.safe_load(file)
    return config

# Using a decorator to read the config with a customizable path
def config_reader(config_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = read_config(config_path)
            return func(config, *args, **kwargs)
        return wrapper
    return decorator


def parse_command_line_args(config):
    parser = argparse.ArgumentParser(description="Run application with configuration.")

    dict_keys = []
    # Dynamically add arguments based on config keys
    for k,v in config.items():
        if 'type' in k or 'help' in k:
            continue
        if type(v) == dict:
            for kk, vv in v.items():
                if 'type' in kk or 'help' in kk:
                    continue
                if type(vv) == dict:
                    for kkk, vvv in v.items():
                        if 'type' in kkk or 'help' in kkk:
                            continue

                        type_key = f"{kkk}_type"
                        help_key = f"{kkk}_help"
                        expected_type = eval(v.get(type_key, 'str'))  # Default to 'str'
                        if expected_type == StringList:
                            nargs = '*'
                        else:
                            nargs = '?'
                        help_message = v.get(help_key, '')  # Default to 'str'
                        parser.add_argument(f'--{k}.{kk}.{kkk}', type=expected_type, nargs=nargs,
                                            help=f'Override {kkk}. {help_message}', default=vvv)
                else:
                    type_key = f"{kk}_type"
                    help_key = f"{kk}_help"
                    expected_type = eval(config[k].get(type_key, 'str'))  # Default to 'str'
                    dict_keys.append(k)
                    if expected_type == StringList:
                        nargs = '*'
                    else:
                        nargs = '?'
                    help_message = config[k].get(help_key, '')  # Default to 'str'
                    parser.add_argument(f'--{k}-{kk}', type=expected_type, nargs=nargs,
                                        help=f'Override {kk}. {help_message}', default=vv)

        else:
            type_key = f"{k}_type"
            help_key = f"{k}_help"
            expected_type = eval(config.get(type_key, 'str'))  # Default to 'str'
            if expected_type == StringList:
                nargs = '*'
                expected_type = str
            else:
                nargs = '?'
            help_message = config.get(help_key, '')  # Default to 'str'
            parser.add_argument(f'--{k}', type=expected_type, nargs=nargs,
                                help=f'Override {k}. {help_message}', default=v)

    args = parser.parse_args()
    return args