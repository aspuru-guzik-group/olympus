#!/usr/bin/env python

from olympus import Logger

from . import get_noises_list
from . import import_noise
from . import AbstractNoise


def Noise(kind='GaussianNoise', scale=1):
    """Convenience function to access noise objects. It returns a certain noise object
    with defaults arguments.

    Args:
        kind (str or AbstractNoise): Keyword identifying one of the noise objects available in Olympus. Alternatively,
            you can pass a custom noise class that is a subclass of AbstractNoise.

    Returns:
        Noise (object): An instance of the chosen noise.
    """
    _validate_noise_kind(kind)
    # if a string is passed, then load the corresponding wrapper
    if type(kind) == str:
        noise = import_noise(kind)
        noise = noise(scale=scale)
    # if an instance of a planner is passed, simply return the same instance
    elif isinstance(kind, AbstractNoise):
        noise = kind
    # if a custom class is passed, then that is the 'wrapper'
    elif issubclass(kind, AbstractNoise):
        noise = kind()

    return noise


def _validate_noise_kind(kind):
    # if we received a string
    if type(kind) == str:
        avail_noises = get_noises_list()
        if kind not in avail_noises:
            message = ('Noise "{0}" not available in Olympus. Please choose '
                       'from one of the available noise objects: {1}'.format(kind, ', '.join(avail_noises)))
            Logger.log(message, 'FATAL')

    # if we get an instance of a noise class
    elif isinstance(kind, AbstractNoise):
        # make sure it has the necessary methods
        for method in ['_add_noise']:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f'The object {kind} does not implement the necessary method "{method}"'
                Logger.log(message, 'FATAL')

    # if we received a custom planner class
    elif issubclass(kind, AbstractNoise):
        # make sure it has the necessary methods
        for method in ['_add_noise']:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f'The object {kind} does not implement the necessary method "{method}"'
                Logger.log(message, 'FATAL')

    # if we do not know what was passed raise an error
    else:
        message = 'Could not initialize Noise: the argument "kind" is neither a string or AbstractNoise subclass'
        Logger.log(message, 'FATAL')
