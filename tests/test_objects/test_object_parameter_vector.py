#!/usr/bin/env python
import pytest
from olympus.objects import ParameterVector
from olympus import ParameterSpace
from olympus import Parameter


def test_from_functions():

	param_0 = Parameter(name='p0', type='continuous', low=0.0, high=1.0)
	param_1 = Parameter(name='p1', type='continuous', low=0.0, high=1.0)
	param_2 = Parameter(
		name='p2', type='categorical',
		options=['x0', 'x1', 'x2'],
		descriptors=[[1., 2.], [3., 4.], [5., 6.]]
	)
	param_3 = Parameter(
		name='p3', type='discrete',
		low=0.0, high=1.0, stride=0.1,
	)
	param_space = ParameterSpace()
	param_space.add([param_0, param_1, param_2, param_3])

	# from array
	paramvector0 = ParameterVector().from_array([0.1, 0.2, 'x0', 0.7], param_space)
	# from dictionary
	paramvector1 = ParameterVector().from_dict(
		{
			'p0': 0.1,
			'p1': 0.2,
			'p2': 'x0',
			'p3': 0.7,
		}
	)

	for attr0, attr1 in zip(paramvector0.attrs, paramvector1.attrs):
		assert attr0 == attr1

	for prop0, prop1 in zip(paramvector0.props, paramvector1.props):
		assert prop0 == prop1


def test_to_string():
	param_vector = ParameterVector().from_dict({'p0': 0.1, 'p1': 0.2, 'p2': 'x0', 'p3': 0.7})
	assert str(param_vector) == 'ParamVector(p0 = 0.1, p1 = 0.2, p2 = x0, p3 = 0.7)'
