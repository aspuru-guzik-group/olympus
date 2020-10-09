#!/usr/bin/env python

import numpy as np
from olympus import Observations
from olympus import ParameterVector


def test_declaration():
    np.random.seed(100691)
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3,1))
    params = []
    values = []
    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(ParameterVector().from_dict({'p0': param_vect[0], 'p1': param_vect[1]}))
        values.append(ParameterVector().from_dict({'obj0': value_vect[0]}))

    observations = Observations()
    observations.params = params
    observations.values = values

    test_params = [param.to_array() for param in params]
    test_values = [value.to_array() for value in values]

    assert np.linalg.norm(test_params - observations.get_params()) < 1e-7
    assert np.linalg.norm(test_values - observations.get_values()) < 1e-7


def test_add_obseration():
    np.random.seed(100691)
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3,1))
    params = []
    values = []
    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(ParameterVector().from_dict({'p0': param_vect[0], 'p1': param_vect[1]}))
        values.append(ParameterVector().from_dict({'obj0': value_vect[0]}))

    obs = Observations()
    obs.add_observation(params, values)
    assert obs.params == params 
    assert obs.values == values


def test_get_values():
    np.random.seed(100691)
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3,1))
    params = []
    values = []
    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(ParameterVector().from_dict({'p0': param_vect[0], 'p1': param_vect[1]}))
        values.append(ParameterVector().from_dict({'obj0': value_vect[0]}))

    obs = Observations()
    obs.add_observation(params, values)

    obs_vals = obs.get_values(as_array=True, opposite=False)
    assert [_.to_array() for _ in values] == list(obs_vals)

    obs_vals = obs.get_values(as_array=True, opposite=True)
    assert [-1 * _.to_array() for _ in values] == list(obs_vals)

    obs_vals = obs.get_values(as_array=False, opposite=False)
    assert [_.to_dict() for _ in values] == list(obs_vals)

    obs_vals  = obs.get_values(as_array=False, opposite=True)
    obs_dicts = [_.to_dict() for _ in values]
    for _, obs_dict in enumerate(obs_dicts):
        for key, val in obs_dict.items():
            assert val == -1 * obs_vals[_][key]

if __name__ == '__main__':
    test_get_values()