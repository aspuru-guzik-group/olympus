#!/usr/bin/env python

from abc import abstractmethod

import numpy as np

from olympus import Logger
from olympus.campaigns.param_space import ParameterContinuous, ParameterSpace
from olympus.objects import (
    ABCMeta,
    Object,
    ParameterVector,
    abstract_attribute,
)


class AbstractNoise(Object, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

    @abstract_attribute
    def scale(self):
        pass

    @abstractmethod
    def _add_noise(self, value):
        pass

    @property
    def kind(self):
        return type(self).__name__

    # TODO: possible extension is to take also as argument params, to allow for noise that depends on the
    #  location in param space, and not just output location
    def __call__(self, value):
        """Add noise to the provided values.

        Args:
            values (float): Measurement to which noise is to be added.

        Returns:
            noisy_values (array): Noisy measurement.
        """
        return self._add_noise(value)
