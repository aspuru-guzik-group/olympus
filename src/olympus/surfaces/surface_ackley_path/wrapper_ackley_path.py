#!/usr/bin/env python

import numpy as np
from olympus.surfaces import AbstractSurface


class AckleyPath(AbstractSurface):

    def __init__(self, param_dim=2, noise=None):
        """Ackley path function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # minimum at the centre
        params = [0.5] * self.param_dim
        value = self._run(params)
        return [{'params': params, 'value': value}]

    @property
    def maxima(self):
        return None

    def _run(self, params):
        params = np.array(params)
        params = 64 * np.array(params) - 32  # rescale onto [-32, 32]
        a = 20.
        b = 0.2
        c = 2 * np.pi
        n = float(len(params))
        params = np.array(params)
        result = - a * np.exp(- b * np.sqrt(np.sum(params ** 2) / n)) - np.exp(np.sum(np.cos(c * params)) / n) + a + np.exp(1.)

        if self.noise is None:
            return result
        else:
            return self.noise(result)
