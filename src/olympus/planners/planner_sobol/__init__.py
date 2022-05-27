#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "Sobol"
module = "sobol_seq"
link = "https://pypi.org/project/sobol_seq/"
check_planner_module(planner, module, link)

param_types = ["continuous"]

# ===============================================================================

from .wrapper_sobol import Sobol
