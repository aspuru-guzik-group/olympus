#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import Smac


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("goal, rng",
                         [('minimize', None,),
                          ('maximize', None,)])
def test_planner_ask_tell(two_param_space, goal, rng):
    planner = Smac(goal=goal, rng=rng)
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector().from_dict({'objective': 0.})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)