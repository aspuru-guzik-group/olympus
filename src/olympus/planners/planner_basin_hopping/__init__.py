#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ----------------
# Import planner
# ----------------

planner = "BasinHopping"
module = "scipy"
link = "https://www.scipy.org/"
check_planner_module(planner, module, link)

param_types = ["continuous"]

# ------------------------
# Import Planner Wrapper
# ------------------------

from .wrapper_basin_hopping import BasinHopping
