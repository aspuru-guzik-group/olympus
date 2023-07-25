#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

# ===============================================================================

planner = "ParticleSwarms"
module = "pyswarms"
link = "https://pyswarms.readthedocs.io/en/latest/"
check_planner_module(planner, module, link)

param_types = ["continuous"]

# ===============================================================================

from .wrapper_particle_swarms import ParticleSwarms
