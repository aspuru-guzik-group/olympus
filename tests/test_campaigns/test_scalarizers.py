#!/usr/bin/env python

import pytest

import numpy as np
from olympus.campaigns import Campaign
from olympus.scalarizers import Scalarizer
from olympus import ParameterVector
from olympus.surfaces import Surface

chimera_goals = ['min', 'max']
chimera_absolute_tolerances = []

num_absolutes = 6
num_relatives = 6

chimera_configs = []
for _ in range(num_absolutes):
	pass 
for _ in range(num_relatives):
	chimera_configs.append(
		(
			np.random.choice(chimera_goals, size=(2,), replace=True),
			np.random.uniform(size=(2,)),
			[False, False]
		)
	)


weighted_sum_goals = ['min', 'max']
num_tests = 10

weighted_sum_configs = []
for _ in range(num_tests):
	w0 = np.random.uniform()
	weighted_sum_configs.append(
		(
			np.random.choice(weighted_sum_goals, size=(2,), replace=True),
			(w0, 1-w0),
		)
	)


@pytest.mark.parametrize("goals, tolerances, absolutes", chimera_configs)
def test_chimera_scalarizer(goals, tolerances, absolutes):

	surface = Surface(kind='MultFonseca', param_dim=2)

	scalarizer = Scalarizer(
		kind='Chimera', 
		value_space=surface.value_space,
		goals=goals,
		tolerances=tolerances,
		absolutes=absolutes
	)

	params = np.array([[0.01, 0.99], [0.99, 0.01], [0.45, 0.55]])
	values = np.array(surface.run(params))

	scalarized_values = scalarizer.scalarize(values)

	assert len(scalarized_values.shape)==1
	assert all(scalarized_values>=0.) and all(scalarized_values<=1.)



@pytest.mark.parametrize("goals, weights", weighted_sum_configs)
def test_weighted_sum_scalarizer(goals, weights):

	surface = Surface(kind='MultFonseca', param_dim=2)

	scalarizer = Scalarizer(
		kind='WeightedSum', 
		value_space=surface.value_space,
		weights=weights,
		goals=goals,
	)

	params = np.array([[0.01, 0.99], [0.99, 0.01], [0.45, 0.55]])
	values = np.array(surface.run(params))

	scalarized_values = scalarizer.scalarize(values)

	assert len(scalarized_values.shape)==1
	assert all(scalarized_values>=0.) and all(scalarized_values<=1.)


# def test_constrained_asf_scalarizer(kind):

# 	surface = Surface(kind='MultFonseca', param_dim=2)