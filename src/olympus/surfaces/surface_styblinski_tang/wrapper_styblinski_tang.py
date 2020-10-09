#!/usr/bin/env python

import numpy as np
from olympus.surfaces import AbstractSurface


class StyblinskiTang(AbstractSurface):

    def __init__(self, param_dim=2, noise=None):
        """Styblinski-Tang function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        x = (-2.903534 + 5) / 10  # rescale onto unit square
        params = [x] * self.param_dim
        value = self._run(params)
        return [{'params': params, 'value': value}]

    @property
    def maxima(self):
        return None

    def _run(self, params):
        params = np.array(params)
        params = 10 * params - 5  # rescale onto [-5, 5]
        result = 0.5 * np.sum(params**4 - 16*params**2 + 5*params)

        if self.noise is None:
            return result
        else:
            return self.noise(result)
