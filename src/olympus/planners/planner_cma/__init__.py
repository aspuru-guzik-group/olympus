#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

#===============================================================================

planner = 'Cma'
module  = 'cma'
link    = 'https://pypi.org/project/cma/'
check_planner_module(planner, module, link)

#===============================================================================

from .wrapper_cma import Cma
