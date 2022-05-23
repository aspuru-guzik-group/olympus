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
    'Botorch', 'Hebo',
] # Dragonfly, Entmoot, Smac

CAT_PLANNERS = [
    'RandomSearch', 'Botorch', 'Gryffin', 'Hebo', 'Gpyopt', 'Hyperopt', 'Grid',
] # Genetic, Dragonfly

EMULATED_DATASETS = [
    'snar', 'photo_wf3', 'benzylation',
    'fullerenes', 'colors_bob', 'photo_pce10',
    'alkox', 'hplc', 'colors_n9', 'suzuki',
]

FULL_CAT_DATASETS = ['perovskites', 'dye_lasers', 'redoxmers']

DESC_DATASETS = ['perovskites', 'redoxmers'] # datasets which have descriptors

MIXED_DATASETS = ['suzuki_i', 'suzuki_ii', 'suzuki_iii', 'suzuki_iv']

MOO_DATASETS = [
            'dye_lasers', 'redoxmers', 'suzuki_i', 
            'suzuki_ii', 'suzuki_iii', 'suzuki_iv',
    ]

SIMPLEX_CONSTRAINED_DATASETS = ['thin_film', 'photo_pce10', 'photo_wf3']



emulated_tuples = []
for planner in CONT_PLANNERS:
    for dataset in EMULATED_DATASETS:
        emulated_tuples.append((planner, dataset))

full_cat_tuples = []
for planner in CAT_PLANNERS:
    for dataset in FULL_CAT_DATASETS:
        full_cat_tuples.append((planner, dataset))

@pytest.mark.parametrize("planner, dataset", emulated_tuples)
def test_bnn_emulators_optimization(planner, dataset):
    # this is because e.g. excitonics does not have a BNN emulator yet
    #try:
    emulator = Emulator(dataset=dataset, model='BayesNeuralNet')
    planner = Planner(kind=planner, goal='minimize')
    campaign = planner.optimize(emulator=emulator, num_iter=3)
    #except:
    #    pass


# TODO: implement optimization tests of fully categorical and mixed datasets
# TODO: also test optimization with descriptors vs optimization without
#
# @pytest.mark.parametrize("planner, dataset", full_cat_tuples)
# def test_full_cat_optimization(planner, dataset):
#
#     dset = Dataset(kind=dataset)
#     planner = Planner(kind=planner, goal='minimize')
#     campaign = planner.optimize(emulator=emulator, num_iter=3)
