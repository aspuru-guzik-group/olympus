#!/usr/bin/env python

# ===============================================================================

import traceback

from olympus import Logger

from .abstract_plotter import AbstractPlotter

# ===============================================================================


# ===============================================================================
# check for matplotlib

Plotter = None
check_further = False

# ===============================================================================
# check for matplotlib

try:
    import matplotlib
except ModuleNotFoundError:
    error = traceback.format_exc()
    for line in error.split("\n"):
        if "ModuleNotFoundError" in line:
            module = line.strip().strip("'").split("'")[-1]
    message = """Plotter requires {module}, which could not be found.
    Install {module} for prettier plots""".format(
        module=module
    )
    Logger.log(message, "WARNING", only_once=True)
else:
    from .plotter_matplotlib import PlotterMatplotlib

    Plotter = PlotterMatplotlib
    check_further = True

    from .olympus_colors import get_olympus_colors

# ===============================================================================
# check for seaborn

if check_further:
    try:
        import seaborn
    except ModuleNotFoundError:
        error = traceback.format_exc()
        for line in error.split("\n"):
            if "ModuleNotFoundError" in line:
                module = line.strip().strip("'").split("'")[-1]
        message = """Plotter requires {module}, which could not be found.
        Please install {module} to use the plotter""".format(
            module=module
        )
        Logger.log(message, "WARNING", only_once=True)
    else:
        from .plotter_seaborn import PlotterSeaborn

        Plotter = PlotterSeaborn
