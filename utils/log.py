import random
import string


def generate_random_string():
    """Generate a random string of length 10"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))


def print_info(*args):
    """Prints an information message"""
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")


def print_success(*args):
    """Prints a success message"""
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")


def print_alert(*args):
    """Prints an alert message"""
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")


def print_error(*args):
    """Prints an error message"""
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")
