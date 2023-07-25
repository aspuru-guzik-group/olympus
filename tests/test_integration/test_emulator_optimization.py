#!/usr/bin/env python

import pytest

from olympus import Emulator, Planner
from olympus.datasets import datasets_list
from olympus.planners import planner_names

CONT_PLANNERS = [
    "Snobfit",
    "Slsqp",
    "ConjugateGradient",
    "RandomSearch",
    "DifferentialEvolution",
    "SteepestDescent",
    "Cma",
    "Grid",
    "Hyperopt",
    "BasinHopping",
    "Gpyopt",
    "Lbfgs",
    "LatinHypercube",
    "Sobol",
    "Gryffin",
    "Simplex",
    "Botorch",
    "Hebo",
]  # MISSING PLANNERS: Dragonfly, Entmoot, Smac,

# BROKEN PLANNERS: Phoenics, Genetic, ParticleSwarms,


CAT_PLANNERS = [
    "RandomSearch",
    "Botorch",
    "Gryffin",
    "Hebo",
    "Gpyopt",
    "Hyperopt",
    "Grid",
]  # Genetic, Dragonfly

MOO_PLANNERS = [
    "RandomSearch",
    "Botorch",
    "Gryffin",
    "Hebo",
    "Gpyopt",
    "Hyperopt",
    "Grid",
    "Sobol",
    "LatinHypercube",
    "Cma",
    "Snobfit",
]

EMULATED_DATASETS = [
    "snar",
    "photo_wf3",
    "benzylation",
    "fullerenes",
    "colors_bob",
    "photo_pce10",
    "alkox",
    "hplc",
    "colors_n9",
    "suzuki",
]

FULL_CAT_DATASETS = ["perovskites", "dye_lasers", "redoxmers"]

DESC_DATASETS = ["perovskites", "redoxmers"]  # datasets which have descriptors

MIXED_DATASETS = ["suzuki_i", "suzuki_ii", "suzuki_iii", "suzuki_iv"]

MOO_DATASETS = [
    "dye_lasers",
    "redoxmers",
    "suzuki_i",
    "suzuki_ii",
    "suzuki_iii",
    "suzuki_iv",
]

SIMPLEX_CONSTRAINED_DATASETS = [
    "thin_film",
    "photo_pce10",
    "photo_wf3",
    "oer_plate_3496",
    "oer_plate_3851",
    "oer_plate_3860",
    "oer_plate_4098",
]


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
    # try:
    emulator = Emulator(dataset=dataset, model="BayesNeuralNet")
    planner = Planner(kind=planner, goal="minimize")
    campaign = planner.optimize(emulator=emulator, num_iter=3)
    # except:
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


if __name__ == '__main__':

    pass