#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "Snobfit"
module = "SQSnobFit"
link = "https://pypi.org/project/SQSnobFit/"
check_planner_module(planner, module, link)

param_types = ["continuous"]

# ===============================================================================

from .wrapper_snobfit import Snobfit
