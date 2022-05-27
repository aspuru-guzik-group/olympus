#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface


class Zakharov(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """Zakharov function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        value_dim = 1
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        params = [1.0 / 3.0] * self.param_dim
        value = self._run(params)
        return [{"params": params, "value": value}]

    @property
    def maxima(self):
        return None

    def _run(self, params):
        params = np.array(params)
        params = 15 * params - 5  # rescale onto [-5, 10]

        term1 = np.sum(params**2)
        term2 = 0.0
        for i, x in enumerate(params):
            term2 += 0.5 * (i + 1) * x
        term3 = term2**4
        term2 = term2**2

        result = term1 + term2 + term3

        if self.noise is None:
            return result
        else:
            return self.noise(result)
