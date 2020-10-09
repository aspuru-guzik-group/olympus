#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import RandomSearch


# use parametrize to test multiple configurations of the planner
#@pytest.mark.parametrize("disp, eps, ftol, gtol, maxcor, maxfun, maxiter, maxls",
#						 [(None, 1e-8, 2.220446049250313e-9, 1e-5, 10, 15000, 15000, 20),
#						  (True, 1e-9, 2.220446049250313e-10, 1e-6, 15, 20000, 20000, 30)])
def test_planner_ask_tell(two_param_space):#, disp, eps, ftol, gtol, maxcor, maxfun, maxiter, maxls):
#	planner = Lbfgs(disp=disp, eps=eps, ftol=ftol, gtol=gtol, maxcor=maxcor, maxfun=maxfun, maxiter=maxiter, maxls=maxls)
	planner = RandomSearch()
	planner.set_param_space(param_space=two_param_space)
	param = planner.ask()
	value = ParameterVector().from_dict({'objective': 0.})
	obs = Observations()
	obs.add_observation(param, value)
	planner.tell(observations=obs)


if __name__ == '__main__':
    from olympus import Parameter, ParameterSpace
    param_space = ParameterSpace()
    param_space.add(Parameter(name='param_0'))
    param_space.add(Parameter(name='param_1'))
    test_planner_ask_tell(param_space)
