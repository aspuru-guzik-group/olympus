#!/usr/bin/env python

from olympus import Logger
from olympus import Parameter
from olympus import ParameterSpace


def test_single_parameter():
	param_space = ParameterSpace()
	param = Parameter()
	param_space.add(param)
	assert param_space.param_names == ['parameter']
	for attr in ['name', 'type', 'low', 'high']:
		assert getattr(param_space[0], attr) == getattr(param, attr)


def test_multiple_parameters():
	param_space = ParameterSpace()
	params      = [Parameter(name = f'param_{_}') for _ in range(4)]
	param_space.add(params)
	assert param_space.param_names == [f'param_{_}' for _ in range(4)]


def test_name_collisions():
	param_space = ParameterSpace()
	for _ in range(4):
		param = Parameter(name = f'param_{_}')
		param_space.add(param)
	param_space.add(Parameter(name = 'param_0'))
	assert len(Logger.ERRORS) == 1
	Logger.purge()


def test_parameter_ordering():
	param_space = ParameterSpace()
	for _ in range(4):
		param = Parameter(name='param_{}'.format(_))
		param_space.add(param)
	for _, param in enumerate(param_space):
		assert param.name == 'param_{}'.format(_)
