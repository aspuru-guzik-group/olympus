#!/usr/bin/env python

from olympus import Planner, Emulator
from olympus.datasets import datasets_list
from olympus.planners import planner_names
import pytest


test_tuples = []
for planner in planner_names:
    for dataset in datasets_list:
        test_tuples.append((planner, dataset))

@pytest.mark.parametrize("planner, dataset", test_tuples)
def test_bnn_emulators_optimization(planner, dataset):
    # this is because e.g. excitonics does not have a BNN emulator yet
    try:
        emulator = Emulator(dataset=dataset, model='BayesNeuralNet')
        planner = Planner(kind=planner, goal='minimize')
        campaign = planner.optimize(emulator=emulator, num_iter=3)
    except:
        pass


