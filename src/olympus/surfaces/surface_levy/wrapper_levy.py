#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface


class Levy(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        value_dim = 1
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        params = [0.55] * self.param_dim
        value = self._run(params)
        return [{"params": params, "value": value}]

    @property
    def maxima(self):
        return None

    def _run(self, params):
        params = np.array(params)
        params = 20 * params - 10  # rescale onto [-10, 10]
        w = 1.0 + ((params - 1.0) / 4.0)
        result = np.sin(np.pi * w[0]) ** 2 + ((w[-1] - 1) ** 2) * (
            1 + np.sin(2 * np.pi * w[-1]) ** 2
        )
        for i, x in enumerate(params):
            if i + 1 == len(params):
                break
            result += ((w[i] - 1) ** 2) * (
                1 + 10 * np.sin(np.pi * w[i] + 1) ** 2
            )

        if self.noise is None:
            return result
        else:
            return self.noise(result)
