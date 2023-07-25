#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "Botorch"
module = "botorch"
link = "https://pypi.org/project/botorch/"
check_planner_module(planner, module, link)

param_types = ["continuous", "discrete", "categorical"]

# ===============================================================================

from .wrapper_botorch import Botorch
