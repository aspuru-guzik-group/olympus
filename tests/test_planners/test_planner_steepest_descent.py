#!/usr/bin/env python

import pytest

from olympus import Observations, ParameterVector
from olympus.planners import SteepestDescent


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize(
    "learning_rate, dx, random_seed, init_guess",
    [
        (1e-3, 1e-5, None, None),
        (1e-4, 1e-6, 42, None),
        (1e-3, 1e-5, None, [0.5, 0.5]),
    ],
)
def test_planner_ask_tell(
    two_param_space, learning_rate, dx, random_seed, init_guess
):
    planner = SteepestDescent(
        learning_rate=1e-3, dx=1e-5, random_seed=None, init_guess=None
    )
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector().from_dict({"objective": 0.0})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)
