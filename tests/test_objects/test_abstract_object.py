#!/usr/bin/env python

import json
import pytest

from olympus import Parameter


def test_update():
    param = Parameter(name = 'temperature')
    param.update('name', 'pressure')
    assert param.name == 'pressure'

def test_contains():
    param = Parameter()
    assert 'name' in param

def test_dict_conversion():
    param_orig = Parameter(name = 'temperature')
    param_dict = param_orig.to_dict()
    param_dict['low']  = 0.
    param_dict['high'] = 273.
    param_conv = Parameter().from_dict(param_dict)
    assert param_conv.low  == param_dict['low']
    assert param_conv.high == param_dict['high']
    assert param_conv.name == param_orig.name

def test_json_conversion():
    param_orig = Parameter(name = 'temperature')
    param_json = param_orig.to_json()
    param_dict = json.loads(param_json)
    param_dict['low']  = 0.
    param_dict['high'] = 273.
    param_conv = Parameter().from_json(json.dumps(param_dict))
    assert param_conv.low  == param_dict['low']
    assert param_conv.high == param_dict['high']
    assert param_conv.name == param_orig.name


def test_set_get_methods_generation():
    param = Parameter(name = 'temperature')
    param.set_name('pressure')
    assert param.get_name() == 'pressure'


if __name__ == '__main__':
    test_json_conversion()
