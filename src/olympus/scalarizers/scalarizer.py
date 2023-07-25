#!/usr/bin/env python


from olympus import Logger
from olympus.campaigns.param_space import ParameterSpace

from . import AbstractScalarizer, get_scalarizers_list, import_scalarizer


def Scalarizer(kind="Chimera", value_space=None, **kwargs):

    """Convenience function to access the scalarizers via a slightly higher level interface.
    Returns a certain surface with defaults arguments by keyword.

    Args:
            kind (str or AbstractASF): Keyword identifying one of the achievement
                    scalarizing functions available in Olympus. Alternatively,
                    you can pass a custom algorithm that is a subclass of AbstractASF.
            value_space (ParameterSpace): object representing the value space for the
                    optimization problem

    Returns:
            Scalarizer: An instance of the chosen scalatizer initialize with the
                    chosen configuration
    """
    _validate_asf_kind(kind)
    _validate_value_space(value_space)

    # _validate_asf_params(**kwargs)
    # if a string is passed, load the corresponding asf wrapper
    if type(kind) == str:
        scalarizer = import_scalarizer(kind)
        scalarizer.check_kwargs(kwargs)
        scalarizer = scalarizer(value_space, **kwargs)

    elif isinstance(kind, AbstractScalarizer):
        scalarizer = kind

    elif issubclass(kind, AbstractScalarizer):
        scalarizer = kind(**kwargs)

    return scalarizer


def _validate_value_space(value_space):

    if not isinstance(value_space, ParameterSpace):
        message = "You must pass a value space object which is an instance of an Olympus ParameterSpace"
        Logger.log(message, "FATAL")


def _validate_asf_kind(kind):

    if type(kind) == str:
        avail_scalarizers = get_scalarizers_list()
        # passed string
        if kind not in avail_scalarizers:
            message = (
                f'Surface "{kind}" not available in Olympus. Please choose '
                'from one of the available surfaces: {", ".join(avail_scalarizers)}'
            )
            Logger.log(message, "FATAL")

    # passed an instance of a planner class
    elif isinstance(kind, AbstractScalarizer):
        # make sure the class has the proper methods
        for method in ["scalarize", "validate_asf_params"]:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f"The object {kind} does not implement the nessecary method {method}"
                Logger.log(message, "FATAL")

    # passed a custom scalarizer
    elif issubclass(kind, AbstractScalarizer):
        # make sure the class has the proper methods
        for method in ["scalarize", "_validate_asf_params"]:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f"The object {kind} does not implement the nessecary method {method}"
                Logger.log(message, "FATAL")

    else:
        message = 'Could not initialize Scalarizer: the argument "kind" is neither a string or AbstractScalarizer instance or subclass'
        Logger.log(message, "FATAL")


# TODO: impement this!! Quick check that the provided params matches the
# expected parameters for each scalarizer
def _validate_asf_params(params):
    pass


# -----------
# DEBUGGING
# -----------

if __name__ == "__main__":
    pass
