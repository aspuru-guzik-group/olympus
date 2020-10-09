#!/usr/bin/env python

import pytest
import numpy as np
from olympus import Observations, ParameterVector
from olympus.planners import ConjugateGradient


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("disp, maxiter, gtol, norm, eps",
                         [(False, None, 1e-05, np.inf, 1.4901161193847656e-8),
                          (True, None, 1e-04, np.inf, 2e-8)])
def test_planner_ask_tell(two_param_space, disp, maxiter, gtol, norm, eps):
    planner = ConjugateGradient(disp=disp, maxiter=maxiter, gtol=gtol, norm=norm, eps=eps)
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
