#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "LatinHypercube"
module = "pyDOE"
link = "https://pythonhosted.org/pyDOE/"
check_planner_module(planner, module, link)

param_types = ["continuous"]

# ===============================================================================

from .wrapper_latin_hypercube import LatinHypercube
