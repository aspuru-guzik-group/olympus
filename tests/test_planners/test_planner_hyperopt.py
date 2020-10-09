#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import Hyperopt


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("show_progressbar", [False, True])
def test_planner_ask_tell(two_param_space, show_progressbar):
	planner = Hyperopt(show_progressbar=show_progressbar)
	planner.set_param_space(param_space=two_param_space)
	param = planner.ask()
	value = ParameterVector().from_dict({'objective': 0.})
	obs = Observations()
	obs.add_observation(param, value)
	planner.tell(observations=obs)
