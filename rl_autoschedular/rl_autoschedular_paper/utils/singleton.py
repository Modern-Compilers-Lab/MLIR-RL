"""Singleton metaclass for ensuring single instance of classes.

This module provides a metaclass that enforces the singleton pattern,
ensuring only one instance of a class is created throughout the application lifetime.
"""


class Singleton(type):
    """Meta class to create a singleton instance of a class"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
