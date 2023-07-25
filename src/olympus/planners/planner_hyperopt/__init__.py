#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ----------------
# Import planner
# ----------------

planner = "Hyperopt"
module = "hyperopt"
link = "http://hyperopt.github.io/hyperopt"
check_planner_module(planner, module, link)

param_types = ["continuous", "categorical"]

# ------------------------
# Import Planner Wrapper
# ------------------------

from .wrapper_hyperopt import Hyperopt
