#!/usr/bin/env python
import pytest
from olympus.objects import ParameterVector
from olympus import ParameterSpace
from olympus import Parameter


def test_from_functions():

	param_0 = Parameter(name='p0', type='continuous', low=0.0, high=1.0)
	param_1 = Parameter(name='p1', type='continuous', low=0.0, high=1.0)

	param_space = ParameterSpace()
	param_space.add([param_0, param_1])

	# from array
	paramvector0 = ParameterVector().from_array([0.1, 0.2], param_space)
	# from dictionary
	paramvector1 = ParameterVector().from_dict({'p0': 0.1, 'p1': 0.2})

	for attr0, attr1 in zip(paramvector0.attrs, paramvector1.attrs):
		assert attr0 == attr1

	for prop0, prop1 in zip(paramvector0.props, paramvector1.props):
		assert prop0 == prop1


def test_to_string():
	param_vector = ParameterVector().from_dict({'p0': 0.1, 'p1': 0.2})
	assert str(param_vector) == 'ParamVector(p0 = 0.1, p1 = 0.2)'
