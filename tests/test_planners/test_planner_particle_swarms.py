#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import ParticleSwarms


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("max_iters, options, particles",
                         [(10**8, {'c1': 0.5, 'c2': 0.3, 'w': 0.9}, 10),
                          (10**8, {'c1': 0.5, 'c2': 0.3, 'w': 0.9}, 20)])

# NOTE: something about this test seems to mess up any subsequent tests;
#       unfortunately I understand too little about pytest to fix this; 
#       if you know what's going on, could you please give me a hint?

# Matteo: mmm I am not sure about this but it may be a pytest bug:
# https://github.com/pytest-dev/pytest/issues/5743
# The tests seem to run fine with the '-s' flag: pytest -s tests/test_planners
# The easiest solution/test would be to try switch off the pyswarm logging/messages, but I have not found a way to
# do this!
def _test_planner_ask_tell(two_param_space, max_iters, options, particles):
    planner = ParticleSwarms(max_iters=max_iters, options=options, particles=particles)
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector().from_dict({'objective': 0.})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)
