#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus.surfaces import AbstractSurface


class Branin(AbstractSurface):
    def __init__(self, noise=None):
        """Branin function.

        Args:
            param_dim (int): Number of input dimensions. Default is 2.
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        param_dim = 2
        value_dim = 1
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # 3 global minima
        x0 = (
            np.array([-np.pi, np.pi, 9.42478]) + 5
        ) / 15  # rescale onto unit square
        x1 = np.array([12.275, 2.275, 2.475]) / 15  # rescale onto unit square
        params0 = np.array([x0[0], x1[0]])
        params1 = np.array([x0[1], x1[1]])
        params2 = np.array([x0[2], x1[2]])
        value0 = self._run(params0)
        value1 = self._run(params1)
        value2 = self._run(params2)

        # check the minima are the same
        np.testing.assert_almost_equal(value0, value1)
        np.testing.assert_almost_equal(value0, value2)

        return [
            {"params": params0, "value": value0},
            {"params": params1, "value": value1},
            {"params": params2, "value": value2},
        ]

    @property
    def maxima(self):
        params = [0.0, 0.0]
        value = self._run(params)
        return {"params": params, "value": value}

    def _run(self, params):
        params = np.array(params)
        # we always only receive 2-dimensional arrays
        x0 = 15 * params[0] - 5  # rescale onto [-5, 10]
        x1 = 15 * params[1]  # rescale onto [0, 15]

        # Branin params
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        result = (
            a * (x1 - b * (x0**2) + c * x0 - r) ** 2
            + s * (1 - t) * np.cos(x0)
            + s
        )

        if self.noise is None:
            return result
        else:
            return self.noise(result)
