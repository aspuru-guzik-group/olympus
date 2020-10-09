#!/usr/bin/env python

import numpy as np
from olympus.surfaces import AbstractSurface


class Michalewicz(AbstractSurface):

    def __init__(self, param_dim=2, m=10, noise=None):
        """Michalewicz function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            m (float): Parameter that defines the steepness of the valleys and ridges. A larger m leads to a more
            difficult search. Default is 10.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # "Certified Global Minima for a Benchmark of Difficult Optimization Problems":
        # https://pdfs.semanticscholar.org/6e82/d0cd7acbd55774c131af8941db6c2f84b7ec.pdf
        if self.param_dim == 2:
            params = [2.202906, 1.570796]
        elif self.param_dim == 3:
            params = [2.202906, 1.570796, 1.284992]
        elif self.param_dim == 4:
            params = [2.202906, 1.570796, 1.284992, 1.923058]
        elif self.param_dim == 5:
            params = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470]
        elif self.param_dim == 6:
            params = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796]
        elif self.param_dim == 7:
            params = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414]
        elif self.param_dim == 8:
            params = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087]
        elif self.param_dim == 9:
            params = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717]
        elif self.param_dim == 10:
            params = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.5]
        else:
            return None

        # scale onto (0,1)
        params = np.array(params) / np.pi
        value = self._run(params)
        return [{'params': params, 'value': value}]

    @property
    def maxima(self):
        # maxima at the corners
        return None

    def _run(self, params):
        params = np.array(params)
        params = np.pi * params  # rescale onto [0, pi]
        result = 0.
        for i, x in enumerate(params):
            result -= np.sin(x) * np.sin((i + 1) * (x ** 2) / np.pi) ** (2 * self.m)

        if self.noise is None:
            return result
        else:
            return self.noise(result)
