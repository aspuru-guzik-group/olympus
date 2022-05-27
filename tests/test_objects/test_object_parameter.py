#!/usr/bin/env python

import pytest

from olympus import Parameter, ParameterSpace
from olympus import (
	ParameterContinuous,
	ParameterDiscrete,
	ParameterCategorical
)
from olympus import Logger

#===============================================================================

def test_abstract_parameter_defaults():
	param = Parameter()
	for attr, value in {'name': 'parameter', 'type': 'continuous', 'low': 0.0, 'high': 1.0}.items():
		assert param[attr] == value


@pytest.mark.parametrize("name, paramtype, low, high", [
	('temperature', 'continuous', 0.0, 1.0),
	('pressure', 'continuous', -1.0, 1.0),
	('concentration', 'continuous', 0.0, 2.0)])
def test_abstract_parameter(name, paramtype, low, high):
	param = Parameter(name=name, type=paramtype, low=low, high=high)
	for attr, value in {'name': name, 'type': paramtype, 'low': low, 'high': high}.items():
		assert param[attr] == value


@pytest.mark.parametrize("name, param_type, low, high", [
	('temperature', 'continuous', 0.0, 1.0),
	('pressure', 'continuous', -1.0, 1.0),
	('concentration', 'continuous', 0.0, 2.0)])
def test_abstract_parameter_dict_conversion(name, param_type, low, high):
	param_orig = Parameter(name = name, type = param_type, low = low, high = high)
	param_dict = param_orig.to_dict()
	param_conv = Parameter().from_dict(param_dict)
	for attr in ['name', 'type', 'low', 'high']:
		assert param_conv.get(attr) == param_orig.get(attr)


def test_abstract_parameter_set_low_high_invalid():
	param = Parameter(low=1.0, high=0.0)
	assert len(Logger.WARNINGS) >= 1


@pytest.mark.parametrize("name, low, high", [
	('temperature', 0.0, 1.0),
	('pressure', -1.0, 1.0),
	('concentration', 0.0, 2.0)])
def test_continuous_parameter(name, low, high):
	param = ParameterContinuous(name = name, low = low, high = high)
	for attr, value in zip(['name', 'low', 'high'], [name, low, high]):
		assert param.get(attr) == value


@pytest.mark.parametrize("name, low, high, stride", [
	('temperature', 50., 100., 10.),
	('pressure', -1.0, 1.0, 0.2)])
def test_discrete_parameters(name, low, high, stride):
	param = ParameterDiscrete(name=name, low=low, high=high, stride=stride)
	for attr, value in zip(['name', 'low', 'high', 'stride'], [name, low, high, stride]):
		assert param.get(attr) == value


@pytest.mark.parametrize("name, options, descriptors", [
	('organic_cation', ['methylammonium', 'ethylammonium'], [ [2., 32.065], [3., 46.09] ] ),
	('anion', ['Br', 'Cl'], [None, None]),
	('inorganic_cation', ['Pb', 'Sn'], [None, None])])
def test_categorical_parameters(name, options, descriptors):
	param = ParameterCategorical(name=name, options=options, descriptors=descriptors)
	for attr, value in zip(['name', 'options', 'descriptors'], [name, options, descriptors]):
		assert param.get(attr) == value


def test_mixed_param_space():
	param_space = ParameterSpace()
	cont_args = {'name': 'param_0', 'low': 0., 'high': 1.}
	param_space.add(
		ParameterContinuous(
			name=cont_args['name'],
			low=cont_args['low'],
			high=cont_args['high'],
		)
	)
	disc_args = {'name': 'param_1', 'low': 0., 'high': 1., 'stride': 0.1}
	param_space.add(
		ParameterDiscrete(
			name=disc_args['name'],
			low=disc_args['low'],
			high=disc_args['high'],
			stride=disc_args['stride'],
		)
	)
	cat_args = {
		'name': 'param_2',
		'options': [f'x_{i}' for i in range(5)],
		'descriptors': [[i] for i in range(5)],
	}
	param_space.add(
		ParameterCategorical(
			name=cat_args['name'],
			options=cat_args['options'],
			descriptors=cat_args['descriptors'],
		)
	)
