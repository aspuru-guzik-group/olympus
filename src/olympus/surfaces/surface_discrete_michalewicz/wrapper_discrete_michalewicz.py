#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus import Logger
from olympus.surfaces import AbstractSurface


class DiscreteMichalewicz(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """Discrete Michalewicz function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        value_dim = 1
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        params = [0.4] * self.param_dim
        value = self._run(params)
        return [{"params": params, "value": value}]

    @property
    def maxima(self):
        message = "DiscreteMichalewicz has an infinite number of maxima"
        Logger.log(message, "INFO")
        # some maxima
        maxima = []
        params = product([0, 1], repeat=self.param_dim)
        for param in params:
            param = list(param)
            value = self._run(param)
            maxima.append({"params": param, "value": value})
        return maxima

    def _run(self, params):
        params = np.array(params)
        params = 100 * params - 40

        bounds = [1.0, 2.0, 3.0, 4.0, 5.0]
        bounds = np.array(bounds) ** 2
        result = 5
        for bound in bounds[::-1]:
            if np.amax(np.abs(params)) < bound:
                result -= 1
        bounds = [1.25, 2.0, 2.5, 3.0]
        bounds = np.array(bounds) ** 2
        new_res = 5
        for bound in bounds[::-1]:
            if np.amin(np.abs(params)) < bound:
                new_res -= 1
        result = np.amin([result, new_res, 4])

        if self.noise is None:
            return result
        else:
            return self.noise(result)
