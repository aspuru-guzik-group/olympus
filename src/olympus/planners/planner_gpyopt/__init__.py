#!/usr/bin/env python

#======================================================================

from olympus.utils.misc import check_planner_module

#======================================================================

planner = 'Gpyopt'
module  = 'GPyOpt'
link    = 'https://github.com/SheffieldML/GPyOpt'
check_planner_module(planner, module, link)

#======================================================================

from .wrapper_gpyopt import Gpyopt
