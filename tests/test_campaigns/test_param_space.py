#!/usr/bin/env python

from olympus import Logger, Parameter, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)


def test_single_parameter():
    param_space = ParameterSpace()
    param = Parameter()
    param_space.add(param)
    assert param_space.param_names == ["parameter"]
    for attr in ["name", "type", "low", "high"]:
        assert getattr(param_space[0], attr) == getattr(param, attr)


def test_discrete_parameter():
    param_space = ParameterSpace()
    param = ParameterDiscrete()
    param_space.add(param)
    assert param_space.param_names == ["parameter"]
    for attr in ["name", "type", "low", "high", "stride"]:
        assert getattr(param_space[0], attr) == getattr(param, attr)


def test_categorical_parameter():
    param_space = ParameterSpace()
    param = ParameterCategorical()
    param_space.add(param)
    assert param_space.param_names == ["parameter"]
    for attr in ["name", "type", "options", "descriptors"]:
        assert getattr(param_space[0], attr) == getattr(param, attr)


def test_multiple_parameters():
    param_space = ParameterSpace()
    params = [Parameter(name=f"param_{_}") for _ in range(4)]
    param_space.add(params)
    assert param_space.param_names == [f"param_{_}" for _ in range(4)]


def test_mixed_parameters():
    param_space = ParameterSpace()
    param_0 = ParameterContinuous(
        name="param_0",
        low=0.0,
        high=1.0,
    )
    param_1 = ParameterDiscrete(
        name="param_1",
        low=0.0,
        high=1.0,
        stride=0.1,
    )
    param_2 = ParameterCategorical(
        name="param_2", options=["a", "b", "c"], descriptors=[None, None, None]
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    assert len(param_space) == 3
    assert [param.type for param in param_space] == [
        "continuous",
        "discrete",
        "categorical",
    ]


def test_name_collisions():
    param_space = ParameterSpace()
    for _ in range(4):
        param = Parameter(name=f"param_{_}")
        param_space.add(param)
    param_space.add(Parameter(name="param_0"))
    assert len(Logger.ERRORS) == 1
    Logger.purge()


def test_parameter_ordering():
    param_space = ParameterSpace()
    for _ in range(4):
        param = Parameter(name="param_{}".format(_))
        param_space.add(param)
    for _, param in enumerate(param_space):
        assert param.name == "param_{}".format(_)
