#!/usr/bin/env python

import numpy as np
import pytest
from olympus.surfaces import Surface
from olympus.surfaces import get_surfaces_list


def test_init():
    surface = Surface(kind='Dejong', param_dim=2)
    assert surface.kind == 'Dejong'
    assert surface.param_dim == 2


def test_run_dejong():
    surface = Surface(kind='Dejong', param_dim=2)
    params = np.zeros(2) + 0.5
    values = surface.run(params)[0][0]
    assert values < 1e-7


@pytest.mark.parametrize("kind", get_surfaces_list())
def test_surfaces(kind):
    surface = Surface(kind=kind, param_dim=2)
    min_dicts = surface.minima
    if kind not in ['Denali', 'Everest', 'K2', 'Kilimanjaro', 'Matterhorn', 'MontBlanc', 'GaussianMixture']:
        for min_dict in min_dicts:
            params, value = min_dict['params'], min_dict['value']
            calc_value = surface.run(params)[0][0]
            np.testing.assert_almost_equal(value, calc_value)
