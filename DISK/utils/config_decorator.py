import yaml
import argparse
from typing import List, TypeVar, Generic
import os
import sys

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

def test_boolean_variable(var, varname):
    if var is not None:
        if type(var) == str:
            seq = var[0].upper() + var[1:]
            if eval(seq) not in [True, False]:
                print("\n❌ sequential should be a "
                      f"bool. Got {var}")
                sys.exit(1)
            else:
                output = eval(seq)
        elif type(var) == bool:
            output = var
        else:
            print(f"\n❌ {varname} should be a "
                  f"bool. Got {var}")
            sys.exit(1)
    else:
        print(f"\n❌ {varname} should be a "
              f"bool. Got {var}")
        sys.exit(1)
    return output


class IntDefault(argparse.Action):
    """Custom action to handle integer arguments allowing _DEFAULT_."""

    def __call__(self, parser, namespace, orig_value, option_string=None):
        if orig_value == '_DEFAULT_':
            value = '_DEFAULT_'
        else:
            # Convert to integer
            value = int(orig_value)

        # Set the value in the namespace
        setattr(namespace, self.dest, value)

class FloatDefault(argparse.Action):
    """Custom action to handle integer arguments allowing _DEFAULT_."""

    def __call__(self, parser, namespace, orig_value, option_string=None):
        if orig_value == '_DEFAULT_':
            value = '_DEFAULT_'
        else:
            # Convert to integer
            value = float(orig_value)

        # Set the value in the namespace
        setattr(namespace, self.dest, value)


class NoAction(argparse.Action):

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)

def single_add_argument(parser, key, value, parent_value, full_name):
    type_key = f"{key}_type"
    help_key = f"{key}_help"
    expected_type = eval(parent_value.get(type_key, 'str'))  # Default to 'str'
    custom_action = 'store'
    if expected_type == StringList:
        nargs = '*'
        expected_type = str
    elif expected_type == IntList:
        nargs = '*'
        expected_type = int
    else:
        nargs = '?'

        if expected_type == bool:
            expected_type = str
            custom_action = 'store'
        elif expected_type == int:
            expected_type = str
            custom_action = IntDefault
        elif expected_type == float:
            expected_type = str
            custom_action = FloatDefault

    help_message = parent_value.get(help_key, '')  # Default to 'str'
    parser.add_argument(f'--{full_name}', type=expected_type, nargs=nargs,
                        action=custom_action,
                        help=f'Override {key}. {help_message}', default=value)
    return parser

def parse_command_line_args(config, desc=''):
    parser = argparse.ArgumentParser(description=desc)

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
                    for kkk, vvv in vv.items():
                        if 'type' in kkk or 'help' in kkk:
                            continue

                    parser = single_add_argument(parser, kkk, vvv, vv, f'{k}-{kk}-{kkk}')
                else:
                    dict_keys.append(k)
                    parser = single_add_argument(parser, kk, vv, v, f'{k}-{kk}')

        else:
             parser = single_add_argument(parser, k, v, config, f'{k}')

    args = parser.parse_args()
    return args