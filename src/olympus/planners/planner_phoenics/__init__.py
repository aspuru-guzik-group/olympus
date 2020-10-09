#!/usr/bin/env python

#======================================================================

from olympus.utils.misc import check_planner_module

#======================================================================

planner = 'Phoenics'
module  = 'phoenics'
link    = 'https://github.com/ChemOS-Inc/phoenics'
check_planner_module(planner, module, link)

#======================================================================

from .wrapper_phoenics import Phoenics
