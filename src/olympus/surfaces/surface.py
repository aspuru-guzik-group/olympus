#!/usr/bin/env python

from olympus import Logger

from . import get_surfaces_list
from . import import_surface
from . import AbstractSurface


def Surface(kind='Dejong', param_dim=2):
    """Convenience function to access surfaces via a slightly higher level interface. It returns a certain surface
    with defaults arguments by keyword.

    Args:
        kind (str or AbstractPlanner): Keyword identifying one of the algorithms available in Olympus. Alternatively,
            you can pass a custom algorithm that is a subclass of AbstractPlanner.
        param_dim (int):

    Returns:
        Surface: An instance of the chosen surface.
    """
    _validate_surface_kind(kind)
    # if a string is passed, then load the corresponding wrapper
    if type(kind) == str:
        surface = import_surface(kind)
        if kind in ['Branin', 'Denali', 'Everest', 'K2', 'Kilimanjaro', 'Matterhorn', 'MontBlanc']:
            surface = surface()
            if param_dim != 2:
                message = f'Surface {kind} is only defined in 2 dimensions: setting `param_dim`=2'
                Logger.log(message, 'WARNING')
        else:
            surface = surface(param_dim=param_dim)
    # if an instance of a planner is passed, simply return the same instance
    elif isinstance(kind, AbstractSurface):
        surface = kind
    # if a custom class is passed, then that is the 'wrapper'
    elif issubclass(kind, AbstractSurface):
        surface = kind()

    return surface


def _validate_surface_kind(kind):
    # if we received a string
    if type(kind) == str:
        avail_surfaces = get_surfaces_list()
        if kind not in avail_surfaces:
            message = ('Surface "{0}" not available in Olympus. Please choose '
                       'from one of the available surfaces: {1}'.format(kind, ', '.join(avail_surfaces)))
            Logger.log(message, 'FATAL')

    # if we get an instance of a planner class
    elif isinstance(kind, AbstractSurface):
        # make sure it has the necessary methods
        for method in ['_run']:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f'The object {kind} does not implement the necessary method "{method}"'
                Logger.log(message, 'FATAL')

    # if we received a custom surface class
    elif issubclass(kind, AbstractSurface):
        # make sure it has the necessary methods
        for method in ['_run']:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f'The object {kind} does not implement the necessary method "{method}"'
                Logger.log(message, 'FATAL')

    # if we do not know what was passed raise an error
    else:
        message = 'Could not initialize Surface: the argument "kind" is neither a string or AbstractSurface subclass'
        Logger.log(message, 'FATAL')
