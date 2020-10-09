#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ----------------
# Import planner
# ----------------

planner = 'DifferentialEvoluation'
module  = 'scipy'
link    = 'https://www.scipy.org/'
check_planner_module(planner, module, link)

# ------------------------
# Import Planner Wrapper
# ------------------------

from .wrapper_differential_evolution import DifferentialEvolution
