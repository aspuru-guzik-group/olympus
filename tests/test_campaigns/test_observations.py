#!/usr/bin/env python

import numpy as np
from olympus import Observations
from olympus import ParameterVector
from olympus import Campaign
from olympus.scalarizers import Scalarizer
from olympus import ParameterSpace, Parameter


def test_declaration():
    np.random.seed(100691)
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3, 1))
    params = []
    values = []
    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0]}))

    observations = Observations()
    observations.add_observation(params, values)

    test_params = [param.to_array() for param in params]
    test_values = [value.to_array() for value in values]

    assert np.linalg.norm(test_params - observations.get_params()) < 1e-7
    assert np.linalg.norm(test_values - observations.get_values()) < 1e-7


def test_add_observation():
    np.random.seed(100691)
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3, 1))
    params = []
    values = []
    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0]}))

    obs = Observations()
    obs.add_observation(params, values)

    assert (
        np.linalg.norm(
            np.array(obs.params) - np.array([param.to_array() for param in params])
        )
        < 1e-7
    )
    assert (
        np.linalg.norm(
            np.array(obs.values) - np.array([value.to_array() for value in values])
        )
        < 1e-7
    )


def test_get_values():
    np.random.seed(100691)
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3, 1))
    params = []
    values = []
    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0]}))

    obs = Observations()
    obs.add_observation(params, values)

    obs_vals = obs.get_values(as_array=True, opposite=False)
    assert [_.to_array() for _ in values] == list(obs_vals)

    obs_vals = obs.get_values(as_array=True, opposite=True)
    assert [-1 * _.to_array() for _ in values] == list(obs_vals)

    obs_vals = obs.get_values(as_array=False, opposite=False)
    assert [_.to_dict() for _ in values] == list(obs_vals)

    obs_vals = obs.get_values(as_array=False, opposite=True)
    obs_dicts = [_.to_dict() for _ in values]
    for _, obs_dict in enumerate(obs_dicts):
        for key, val in obs_dict.items():
            assert val == -1 * obs_vals[_][key]



def test_observations_to_simpl():
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3, 1))
    params = []
    values = []

    campaign = Campaign()

    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0]}))

    campaign.add_observation(params, values)

    campaign.observations_to_simpl()

    assert campaign.observations.get_params().shape == (3, 3)
    assert campaign.observations.get_values().shape == (3, 1)


def test_observations_to_cube():
    param_vects = np.array([[0.2, 0.7, 0.1], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3]])
    values_vects = np.random.uniform(low=0, high=1, size=(3, 1))
    params = []
    values = []

    campaign = Campaign()

    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1], "p2": param_vect[2]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0]}))

    campaign.add_observation(params, values)

    campaign.observations_to_cube()

    assert campaign.observations.get_params().shape == (3, 2)
    assert campaign.observations.get_values().shape == (3, 1)


def test_add_and_scalarize():
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    params = []
    values = []

    campaign = Campaign()
    value_space = ParameterSpace()
    value_space.add(Parameter(name='obj0'))
    value_space.add(Parameter(name='obj1'))
    campaign.set_value_space(value_space)

    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0], "obj1": value_vect[1]}))


    scalarizer = Scalarizer(
        kind='Chimera', 
        value_space=campaign.value_space,
        goals=['min', 'min'],
        tolerances=[0.5, 0.5],
        absolutes=[False, False]
    )

    campaign.add_observation(params, values)

    campaign.add_and_scalarize(
        ParameterVector().from_dict({"p0": 0.43, "p1": 0.06}),
        ParameterVector().from_dict({"obj0": 0.14, "obj1": 0.21}),
        scalarizer,
    )

    scalarized_values = campaign.scalarized_observations.get_values()
    values = campaign.observations.get_values()

    assert len(scalarized_values)==len(values)
    assert all(scalarized_values >= 0.) and all(scalarized_values <= 1.)


def test_reset_history():
    param_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    values_vects = np.random.uniform(low=0, high=1, size=(3, 2))
    params = []
    values = []

    campaign = Campaign()
    value_space = ParameterSpace()
    value_space.add(Parameter(name='obj0'))
    value_space.add(Parameter(name='obj1'))
    campaign.set_value_space(value_space)

    for param_vect, value_vect in zip(param_vects, values_vects):
        params.append(
            ParameterVector().from_dict({"p0": param_vect[0], "p1": param_vect[1]})
        )
        values.append(ParameterVector().from_dict({"obj0": value_vect[0], "obj1": value_vect[1]}))


    scalarizer = Scalarizer(
        kind='Chimera', 
        value_space=campaign.value_space,
        goals=['min', 'min'],
        tolerances=[0.5, 0.5],
        absolutes=[False, False]
    )

    campaign.add_observation(params, values)

    merits = np.array([0.1, 0.0, 1.0])

    campaign.reset_merit_history(merits)
    
    scalarized_values = campaign.scalarized_observations.get_values()
    values = campaign.observations.get_values()

    assert len(scalarized_values)==len(values)
    assert all(scalarized_values >= 0.) and all(scalarized_values <= 1.)






if __name__ == "__main__":
    test_declaration()
