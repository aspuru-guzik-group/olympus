#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "Hebo"
module = "hebo"
link = "https://pypi.org/project/HEBO/"
check_planner_module(planner, module, link)

param_types = ["continuous", "discrete", "categorical"]

# ===============================================================================

from .wrapper_hebo import Hebo
