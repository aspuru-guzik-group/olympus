#!/usr/bin/env python

from olympus import Planner, Emulator
from olympus.datasets import datasets_list
from olympus.planners import planner_names
import pytest


CONT_PLANNERS = [
    'Snobfit', 'Phoenics', 'Slsqp', 'Genetic',
    'ConjugateGradient', 'RandomSearch', 'DifferentialEvolution',
    'ParticleSwarms', 'SteepestDescent', 'Cma', 'Grid',
    'Hyperopt', 'BasinHopping', 'Gpyopt', 'Lbfgs',
    'LatinHypercube', 'Sobol', 'Gryffin', 'Simplex',
]

CAT_PLANNERS = []

EMULATED_DATASETS = [
    'snar', 'photo_wf3', 'benzylation',
    'fullerenes', 'colors_bob', 'photo_pce10',
    'alkox', 'hplc', 'colors_n9', 'suzuki',
]

FULL_CAT_DATASETS = ['perovskites']


emulated_tuples = []
for planner in CONT_PLANNERS:
    for dataset in EMULATED_DATASETS:
        emulated_tuples.append((planner, dataset))

full_cat_tuples = []
for planner in CAT_PLANNERS:
    for dataset in FULL_CAT_DATASETS:
        full_cat_tuples.append((planner, dataset))

@pytest.mark.parametrize("planner, dataset", test_tuples)
def test_bnn_emulators_optimization(planner, dataset):
    # this is because e.g. excitonics does not have a BNN emulator yet
    try:
        emulator = Emulator(dataset=dataset, model='BayesNeuralNet')
        planner = Planner(kind=planner, goal='minimize')
        campaign = planner.optimize(emulator=emulator, num_iter=3)
    except:
        pass


# TODO: implement optimization tests of fully categorical and mixed datasets
# TODO: also test optimization with descriptors vs optimization without
