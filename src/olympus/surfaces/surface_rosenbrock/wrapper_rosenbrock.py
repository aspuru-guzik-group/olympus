#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface


class Rosenbrock(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """Rosenbrock function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        value_dim = 1
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # minimum at (1,1) for 2d
        # vector = np.array([1,1])
        # print((vector+2)/4)
        # minimum at the centre
        params = [0.75] * self.param_dim
        value = self._run(params)
        return [{"params": params, "value": value}]

    @property
    def maxima(self):
        return None

    def _run(self, params):
        params = np.array(params)
        params = 4 * params - 2  # rescale onto [-2, 2]
        result = 0
        for index, element in enumerate(params[:-1]):
            result += (
                100 * (params[index + 1] - element**2) ** 2
                + (1 - element) ** 2
            )

        if self.noise is None:
            return result
        else:
            return self.noise(result)
