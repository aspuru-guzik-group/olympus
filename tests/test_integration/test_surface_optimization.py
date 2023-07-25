#!/usr/bin/env python

import pytest

from olympus import Planner, Surface
from olympus.planners import planner_names
from olympus.surfaces import surface_names

# TODO: now tests only that they run, but later on for a few simple cases like Dejong we should probably test that
#  the minimum/maximum is found (approximately)

CONT_SURFACES = [
    "Dejong",
    "Zakharov",
    "Matterhorn",
    "Rastrigin",
    "Kilimanjaro",
    "Rosenbrock",
    "NarrowFunnel",
    "DiscreteDoubleWell",
    "MontBlanc",
    "K2",
    "Denali",
    "Schwefel",
    "DiscreteMichalewicz",
    "StyblinskiTang",
    "Levy",
    "LinearFunnel",
    "GaussianMixture",
    "Branin",
    "AckleyPath",
    "HyperEllipsoid",
    "Everest",
    "Michalewicz",
    "DiscreteAckley",
]

CAT_SURFACES = [
    "CatDejong",
    "CatMichalewicz",
    "CatAckley",
    "CatCamel",
    "CatSlope",
]

MOO_SURFACES = [
    "MultFonseca",
    "MultViennet",
    "MultZdt1",
    "MultZdt2",
    "MultZdt3",
]  # all continuous, for now

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


cont_tuples = []
for planner in CONT_PLANNERS:
    for surface in CONT_SURFACES:
        cont_tuples.append((planner, surface))

cat_tuples = []
for planner in CAT_PLANNERS:
    for surface in CAT_SURFACES:
        cat_tuples.append((planner, surface))

moo_tuples = []
for planner in MOO_PLANNERS:
    for surface in MOO_SURFACES:
        moo_tuples.append((planner, surface))


@pytest.mark.parametrize("planner, surface", cont_tuples)
def test_cont_surface_optimization(planner, surface):
    surface = Surface(kind=surface, param_dim=2)
    planner = Planner(kind=planner, goal="minimize")
    campaign = planner.optimize(emulator=surface, num_iter=3)


@pytest.mark.parametrize("planner, surface", cat_tuples)
def test_cat_surface_optimization(planner, surface):
    surface = Surface(kind=surface, param_dim=2, num_opts=21)
    planner = Planner(kind=planner, goal="minimize")
    campaign = planner.optimize(emulator=surface, num_iter=3)


# @pytest.mark.parametrize("planner, surface", moo_tuples)
# def test_moo_surface_optimization(planner, surface):
#     surface = Surface(kind=surface, param_dim=2)
#     planner = Planner(kind=planner, goal='minimize')
#     campaign = planner.optimize(emulator=surface, num_iter)
