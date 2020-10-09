#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ----------------
# Import planner
# ----------------

planner = 'Scipy'
module  = 'scipy'
link    = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html'
check_planner_module(planner, module, link)

# ------------------------
# Import Planner Wrapper
# ------------------------

from .wrapper_scipy import WrapperScipy
