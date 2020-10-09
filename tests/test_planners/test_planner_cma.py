#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import Cma


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("stddev", [0.5, 0.4, 0.6])
def test_planner_ask_tell(two_param_space, stddev):
	planner = Cma(stddev=stddev)
	planner.set_param_space(param_space=two_param_space)
	param = planner.ask()
	value = ParameterVector().from_dict({'objective': 0.})
	obs = Observations()
	obs.add_observation(param, value)
	planner.tell(observations=obs)
