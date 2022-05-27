#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus.surfaces import AbstractSurface


class Rastrigin(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """Rastrigin function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        value_dim = 1
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # minimum at the centre
        params = [0.5] * self.param_dim
        value = self._run(params)
        return [{"params": params, "value": value}]

    @property
    def maxima(self):
        # maxima near the corners
        maxima = []
        params = product([0.05, 0.95], repeat=self.param_dim)
        for param in params:
            param = list(param)
            value = self._run(param)
            maxima.append({"params": param, "value": value})
        return maxima

    def _run(self, params):
        result = 10.0 * len(params)
        params = 10 * np.array(params) - 5  # rescale onto [-5, 5]
        for index, element in enumerate(params):
            result += element**2 - 10 * np.cos(2 * np.pi * element)

        if self.noise is None:
            return result
        else:
            return self.noise(result)
