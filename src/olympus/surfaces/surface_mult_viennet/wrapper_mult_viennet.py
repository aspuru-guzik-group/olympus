#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface


class MultViennet(AbstractSurface):
    def __init__(self, noise=None):
        """Viennet function

        Args:
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        param_dim = 2
        value_dim = 3
        AbstractSurface.__init__(**locals())

    @property
    def minima(self):
        # TODO: implement me
        return None
        # return [{'params': params, 'value': value}]

    @property
    def maxima(self):
        # TODO: implement me
        return None

    def _run(self, params):
        params = np.array(params)
        obj_0 = 0.5 * (params[0] ** 2 + params[1] ** 2) + np.sin(
            params[0] ** 2 + params[1] ** 2
        )
        obj_1 = (
            (((3 * params[0] - 2 * params[1] + 4) ** 2) / 8)
            + (((params[0] - params[1] + 1) ** 2) / 27)
            + 15
        )
        obj_2 = (1 / (params[0] ** 2 + params[1] ** 2 + 1)) - (
            1.1 * np.exp(-(params[0] ** 2 + params[1] ** 2))
        )

        raw_results = [obj_0, obj_1, obj_2]

        if self.noise is None:
            return raw_results
        else:
            results = []
            for result in raw_results:
                results.append(self.noise(result))
            return results
