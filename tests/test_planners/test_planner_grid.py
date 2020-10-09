#!/usr/bin/env python

import pytest
from olympus import Observations, ParameterVector
from olympus.planners import Grid
import numpy as np


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize("levels, budget, shuffle, random_seed",
                         [(2, None, False, None),
                          (3, None, False, None),
                          (3, 30, False, None),
                          (2, None, True, None),
                          (2, None, True, 42),
                          (2, 30, True, 42)])
def test_planner_ask_tell(two_param_space, levels, budget, shuffle, random_seed):
    planner = Grid(goal='minimize', levels=levels, budget=budget, shuffle=shuffle, random_seed=random_seed)
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector(dict={'objective': 0.})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)


def test_resetting_planner(two_param_space):
    planner = Grid(levels=3)
    planner.set_param_space(param_space=two_param_space)

    # run once
    obs = Observations()
    for i in range(5):
        param = planner.recommend(observations=obs)
        obj = np.sum(param.to_array() ** 2)
        value = ParameterVector(dict={'objective': obj})
        obs.add_observation(param, value)

    # reset and run again
    planner.reset()
    obs = Observations()
    for i in range(5):
        param = planner.recommend(observations=obs)
        obj = np.sum(param.to_array() ** 2)
        value = ParameterVector(dict={'objective': obj})
        obs.add_observation(param, value)
