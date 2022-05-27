#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface


class MultFonseca(AbstractSurface):
    def __init__(self, noise=None):
        """Fonseca–Fleming function in 2 dimensions
        Fonseca, C. M.; Fleming, P. J. (1995). "An Overview of Evolutionary
        Algorithms in Multiobjective Optimization". Evol Comput. 3 (1): 1–16.

        Args:
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        param_dim = 2
        value_dim = 2
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
        obj_0 = 1 - np.exp(-np.sum((params - 1.0 / np.sqrt(len(params))) ** 2))
        obj_1 = 1 - np.exp(-np.sum((params + 1.0 / np.sqrt(len(params))) ** 2))

        raw_results = [obj_0, obj_1]

        if self.noise is None:
            return raw_results
        else:
            results = []
            for result in raw_results:
                results.append(self.noise(result))
            return results
