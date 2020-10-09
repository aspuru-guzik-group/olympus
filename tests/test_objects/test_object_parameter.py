#!/usr/bin/env python

import pytest

from olympus import Parameter
from olympus import ParameterContinuous
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
