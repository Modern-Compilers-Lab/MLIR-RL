from datetime import datetime
import random
import string
import sys
from dask.distributed import print
import pytz


def generate_random_string():
    """Generate a random string of length 10"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))


def time_log():
    now = datetime.now(pytz.timezone('Africa/Algiers'))
    return now.strftime("%m-%d %H:%M")


def print_info(*args, add_label: bool = True, **kwargs):
    """Prints an information message"""
    message = ' '.join(map(str, args))
    label = f'{time_log()} - [INFO]    ' if add_label else ''
    for line in message.split('\n'):
        print(f"\033[94m{label}{line}\033[0m", **kwargs)


def print_success(*args, add_label: bool = True, **kwargs):
    """Prints a success message"""
    message = ' '.join(map(str, args))
    label = f'{time_log()} - [SUCCESS]    ' if add_label else ''
    for line in message.split('\n'):
        print(f"\033[92m{label}{line}\033[0m", **kwargs)


def print_alert(*args, add_label: bool = True, **kwargs):
    """Prints an alert message"""
    message = ' '.join(map(str, args))
    label = f'{time_log()} - [ALERT]    ' if add_label else ''
    for line in message.split('\n'):
        print(f"\033[93m{label}{line}\033[0m", file=sys.stderr, **kwargs)


def print_error(*args, add_label: bool = True, with_barrier: bool = True, **kwargs):
    """Prints an error message"""
    message = ' '.join(map(str, args))
    if with_barrier:
        message = '\n----------------------------------------\n' + message + '\n----------------------------------------\n'
    label = f'{time_log()} - [ERROR]    ' if add_label else ''
    for line in message.split('\n'):
        print(f"\033[91m{label}{line}\033[0m", file=sys.stderr, **kwargs)
