#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ----------------
# Import planner
# ----------------

planner = 'Genetic'
module  = 'deap'
link    = 'https://deap.readthedocs.io/en/master/'
check_planner_module(planner, module, link)

# ------------------------
# Import Planner Wrapper
# ------------------------
from .wrapper_deap import Genetic
