#!/usr/bin/env python

import numpy as np
import pytest
from deap import tools

from olympus import Observations, ParameterVector
from olympus.planners import Genetic


# use parametrize to test multiple configurations of the planner
@pytest.mark.parametrize(
    "pop_size, cx_prob, mut_prob, mate_args, mutate_args, select_args",
    [
        (
            10,
            0.5,
            0.3,
            {"function": tools.cxTwoPoint},
            {
                "function": tools.mutGaussian,
                "mu": 0,
                "sigma": 0.2,
                "indpb": 0.2,
            },
            {"function": tools.selTournament, "tournsize": 3},
        ),
        (
            15,
            0.2,
            0.8,
            {"function": tools.cxOnePoint},
            {"function": tools.mutShuffleIndexes, "indpb": 0.2},
            {"function": tools.selRoulette, "k": 5},
        ),
        (
            12,
            0.6,
            0.1,
            {"function": tools.cxUniform},
            {"function": tools.mutFlipBit, "indpb": 0.2},
            {"function": tools.selRandom, "k": 6},
        ),
        (
            16,
            0.4,
            0.2,
            {"function": tools.cxSimulatedBinary, "eta": 20},
            {
                "function": tools.mutGaussian,
                "mu": 0,
                "sigma": 0.2,
                "indpb": 0.5,
            },
            {"function": tools.selBest, "k": 4},
        ),
    ],
)
def test_planner_ask_tell(
    two_param_space,
    pop_size,
    cx_prob,
    mut_prob,
    mate_args,
    mutate_args,
    select_args,
):
    planner = Genetic(
        pop_size=pop_size,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        mate_args=mate_args,
        mutate_args=mutate_args,
        select_args=select_args,
    )
    planner.set_param_space(param_space=two_param_space)
    param = planner.ask()
    value = ParameterVector().from_dict({"objective": 0.0})
    obs = Observations()
    obs.add_observation(param, value)
    planner.tell(observations=obs)


def test_generating_new_offsprings(two_param_space):
    planner = Genetic(pop_size=4)
    planner.set_param_space(param_space=two_param_space)
    obs = Observations()
    for i in range(10):
        param = planner.recommend(observations=obs)
        obj = np.sum(param.to_array() ** 2)
        value = ParameterVector(dict={"objective": obj})
        obs.add_observation(param, value)


def test_resetting_planner(two_param_space):
    planner = Genetic(pop_size=3)
    planner.set_param_space(param_space=two_param_space)
    # run once
    obs = Observations()
    for i in range(5):
        param = planner.recommend(observations=obs)
        obj = np.sum(param.to_array() ** 2)
        value = ParameterVector(dict={"objective": obj})
        obs.add_observation(param, value)

    # run again from scratch
    planner.reset()
    obs = Observations()
    for i in range(5):
        param = planner.recommend(observations=obs)
        obj = np.sum(param.to_array() ** 2)
        value = ParameterVector(dict={"objective": obj})
        obs.add_observation(param, value)
