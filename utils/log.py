import random
import string
import sys
from dask.distributed import print


def generate_random_string():
    """Generate a random string of length 10"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))


def print_info(*args, add_label: bool = True, **kwargs):
    """Prints an information message"""
    message = ' '.join(map(str, args))
    label = '[INFO]\t ' if add_label else ''
    print(f"\033[94m{label}{message}\033[0m", **kwargs)


def print_success(*args, add_label: bool = True, **kwargs):
    """Prints a success message"""
    message = ' '.join(map(str, args))
    label = '[SUCCESS]\t ' if add_label else ''
    print(f"\033[92m{label}{message}\033[0m", **kwargs)


def print_alert(*args, add_label: bool = True, **kwargs):
    """Prints an alert message"""
    message = ' '.join(map(str, args))
    label = '[ALERT]\t ' if add_label else ''
    print(f"\033[93m{label}{message}\033[0m", file=sys.stderr, **kwargs)


def print_error(*args, add_label: bool = True, **kwargs):
    """Prints an error message"""
    message = ' '.join(map(str, args))
    label = '[ERROR]\t ' if add_label else ''
    print(f"\033[91m{label}{message}\033[0m", file=sys.stderr, **kwargs)
