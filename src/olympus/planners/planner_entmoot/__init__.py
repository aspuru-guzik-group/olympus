#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "Entmoot"
module = "entmoot"
link = "https://pypi.org/project/lightgbm/"
check_planner_module(planner, module, link)

# param_types = ['continuous', 'discrete', 'categorical']

param_types = ["continuous"]

# ===============================================================================

from .wrapper_entmoot import Entmoot
