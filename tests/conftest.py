#!/usr/bin/env python
import pytest

from olympus import Parameter
from olympus import ParameterSpace
from olympus.planners import get_planners_list

# This is where we can put functions that are useful across multiple tests. It is a bit like
# the base_test_case.py file we had before. For instance, with the fixture two_param_space we
# instantiate a simple ParameterSpace object that can be passed to other functions, like
# test_generators/test_generators.py

@pytest.fixture(scope='module')
def two_param_space():
    param_space = ParameterSpace()
    param_space.add(Parameter(name='param_0'))
    param_space.add(Parameter(name='param_1'))
    return param_space


@pytest.fixture(scope='module')
def list_of_planners():
    list_of_planners = get_planners_list()
    return list_of_planners
