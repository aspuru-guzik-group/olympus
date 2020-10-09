#!/usr/bin/env python

import numpy as np
from olympus.surfaces import AbstractSurface
from itertools import product
from olympus import Logger


class NarrowFunnel(AbstractSurface):

    def __init__(self, param_dim=2, noise=None):
        """Narrow Funnel function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        message = 'NarrowFunnel has an infinite number of minima at 0.49 < x_i < 0.51, for each x_i in x'
        Logger.log(message, 'INFO')
        # minimum at the centre
        params = [0.5] * self.param_dim
        value = self._run(params)
        return [{'params': params, 'value': value}]

    @property
    def maxima(self):
        message = 'NarrowFunnel has an infinite number of maxima'
        Logger.log(message, 'INFO')
        # some maxima
        maxima = []
        params = product([0, 1], repeat=self.param_dim)
        for param in params:
            param = list(param)
            value = self._run(param)
            maxima.append({'params': param, 'value': value})
        return maxima

    def _run(self, params):
        params = np.array(params)
        params = 100 * params - 50
        bounds = [1.0, 2.0, 3.0, 4.0, 5.0]
        bounds = np.array(bounds) ** 2
        result = 5
        for bound in bounds[::-1]:
            if np.amax(np.abs(params)) < bound:
                result -= 1
        result = np.amin([4, result])

        if self.noise is None:
            return result
        else:
            return self.noise(result)
