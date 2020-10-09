#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import Simplex


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("disp, maxiter, maxfev, initial_simplex, xatol, fatol, adaptive",
						 [(False, None, None, None, 0.0001, 0.0001, False),
						  (False, None, None, None, 0.0001, 0.0001, True),
						  (True, None, None, None, 0.001, 0.001, False)])
def test_planner_ask_tell(two_param_space, disp, maxiter, maxfev, initial_simplex, xatol, fatol, adaptive):
	planner = Simplex(disp=disp, maxiter=maxiter, maxfev=maxfev, initial_simplex=initial_simplex, xatol=xatol,
					  fatol=fatol, adaptive=adaptive)
	planner.set_param_space(param_space=two_param_space)
	param = planner.ask()
	value = ParameterVector().from_dict({'objective': 0.})
	obs = Observations()
	obs.add_observation(param, value)
	planner.tell(observations=obs)
