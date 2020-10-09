#!/usr/bin/env python

from olympus import Planner, Surface
from olympus.planners import planner_names
from olympus.surfaces import surface_names
import pytest

# TODO: now tests only that they run, but later on for a few simple cases like Dejong we should probably test that
#  the minimum/maximum is found (approximately)

test_tuples = []
for planner in planner_names:
    for surface in surface_names:
        test_tuples.append((planner, surface))

@pytest.mark.parametrize("planner, surface", test_tuples)
def test_surface_optimization(planner, surface):
    surface = Surface(kind=surface, param_dim=2)
    planner = Planner(kind=planner, goal='minimize')
    campaign = planner.optimize(emulator=surface, num_iter=3)

    #values = campaign.get_values()
    # now check that we are close to the minimum
    #assert np.min(values) < 0.5
