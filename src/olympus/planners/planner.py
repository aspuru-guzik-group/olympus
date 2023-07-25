#!/usr/bin/env python

from olympus import Logger

from . import AbstractPlanner, get_planners_list, import_planner


# NOTE: This goes against the python convention of having function names being
# lowercase, but I thought Planner still works in this case as it effectively
# returns a class instance
# A function seems to me the easiest way to do this right now, but another
# option would be to have a class and fiddle with __new__ an return an instance
# of a different class. I would stick with the simplest option that achieve
# what we need though, unless we already foresee possible expansions that need a
# more complex object
def Planner(kind="ConjugateGradient", goal="minimize", param_space=None):
    """Convenience function to access planners via a slightly higher level interface It returns a certain planner
    with defaults arguments by keyword.

    Args:
        kind (str or AbstractPlanner): Keyword identifying one of the algorithms available in Olympus. Alternatively,
            you can pass a custom algorithm that is a subclass of AbstractPlanner.
        goal (bool): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
        param_space (ParamSpace): A ParameterSpace object defining the space over which to search.

    Returns:
        Planner: An instance of the chosen planning algorithm.
    """
    _validate_planner_kind(kind)
    # if a string is passed, then load the corresponding wrapper
    if type(kind) == str:
        from . import PlannerLoader

        kind = PlannerLoader.file_to_class(kind)
        planner = import_planner(kind)
        planner = planner(goal=goal)
    # if an instance of a planner is passed, simply return the same instance
    elif isinstance(kind, AbstractPlanner):
        planner = kind
    # if a custom class is passed, then that is the 'wrapper'
    elif issubclass(kind, AbstractPlanner):
        planner = kind()

    # load param_space already if provided, otherwise it will have to be set by self.set_param_space
    if param_space is not None:
        planner.set_param_space(param_space)

    return planner


def _validate_planner_kind(kind):
    # if we received a string
    if type(kind) == str:
        from . import PlannerLoader

        kind = PlannerLoader.file_to_class(kind)
        avail_planners = get_planners_list()
        if kind not in avail_planners:
            message = (
                'Planner "{0}" not available in Olympus. Please choose '
                "from one of the available planners: {1}".format(
                    kind, ", ".join(avail_planners)
                )
            )
            Logger.log(message, "FATAL")

    # if we get an instance of a planner class
    elif isinstance(kind, AbstractPlanner):
        # make sure it has the necessary methods
        for method in ["_set_param_space", "_tell", "_ask"]:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f'The object {kind} does not implement the necessary method "{method}"'
                Logger.log(message, "FATAL")

    # if we received a custom planner class
    elif issubclass(kind, AbstractPlanner):
        # make sure it has the necessary methods
        for method in ["_set_param_space", "_tell", "_ask"]:
            implementation = getattr(kind, method, None)
            if not callable(implementation):
                message = f'The object {kind} does not implement the necessary method "{method}"'
                Logger.log(message, "FATAL")

    # if we do not know what was passed raise an error
    else:
        message = 'Could not initialize Planner: the argument "kind" is neither a string or AbstractPlanner subclass'
        Logger.log(message, "FATAL")
