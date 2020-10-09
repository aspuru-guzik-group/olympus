#!/usr/bin/env python

import numpy as np
from olympus.surfaces import AbstractSurface
from itertools import product


class Schwefel(AbstractSurface):

    def __init__(self, param_dim=2, noise=None):
        """Schwefel function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # minimum at 420.9687
        min_loc = (420.9687 + 500) / 1000  # rescale onto unit hypercube
        params = [min_loc] * self.param_dim
        value = self._run(params)
        return [{'params': params, 'value': value}]

    @property
    def maxima(self):
        # is there an analytical maximum?
        return None

    def _run(self, params):
        params = 1000 * np.array(params) - 500  # rescale onto [-500, 500]
        result = 0
        for index, element in enumerate(params):
            result += - element * np.sin(np.sqrt(np.abs(element)))

        if self.noise is None:
            return result
        else:
            return self.noise(result)
