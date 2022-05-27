#!/usr/bin/env python

import pytest

from olympus import Observations, ParameterVector
from olympus.planners import Slsqp


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize(
    "disp, eps, ftol, maxiter",
    [
        (False, 1.4901161193847656e-8, 1e-6, 15000),
        (True, 1.4901161193847656e-8, 1e-6, 10000),
        (False, 1.4901161193847656e-8, 1e-5, 15000),
    ],
)
def test_planner_ask_tell(two_param_space, disp, eps, ftol, maxiter):
    planner = Slsqp(disp=disp, eps=eps, ftol=ftol, maxiter=maxiter)
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector().from_dict({"objective": 0.0})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)
