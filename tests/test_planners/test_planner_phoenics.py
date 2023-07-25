#!/usr/bin/env python

import pytest

from olympus import Observations, ParameterVector
from olympus.planners import Phoenics


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize(
    "batches, boosted, parallel, sampling_strategies",
    [
        (1, True, True, 1),
        (1, True, True, 2),
        (1, False, True, 2),
        (1, False, False, 2),
    ],
)
def test_planner_ask_tell(
    two_param_space, batches, boosted, parallel, sampling_strategies
):
    planner = Phoenics(
        batches=batches,
        boosted=boosted,
        parallel=parallel,
        sampling_strategies=sampling_strategies,
    )
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector().from_dict({"objective": 0.0})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)
