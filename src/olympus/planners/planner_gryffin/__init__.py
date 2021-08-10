#!/usr/bin/env python

#======================================================================

from olympus.utils.misc import check_planner_module

#======================================================================

planner = 'Gryffin'
module  = 'gryffin'
link    = 'https://github.com/aspuru-guzik-group/gryffin/tree/feas'
check_planner_module(planner, module, link)

#======================================================================

from .wrapper_gryffin import Gryffin
