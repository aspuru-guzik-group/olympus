#!/usr/bin/env python

import numpy as np

from olympus.surfaces import AbstractSurface


class MultZdt2(AbstractSurface):
    def __init__(self, param_dim=2, noise=None):
        """Zitzler–Deb–Thiele's function N. 2
        Deb, Kalyan; Thiele, L.; Laumanns, Marco; Zitzler, Eckart (2002).
        "Scalable multi-obj@ective optimization test problems". Proceedings
        of the 2002 IEEE Congress on Evolutionary Computation. Vol. 1. pp.
        825–830

        Args:
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
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
        obj_0 = params[0]
        g = 1 + ((9 / 29) * np.sum(params))
        h = 1 - (obj_0 / g) ** 2
        obj_1 = g * h

        raw_results = [obj_0, obj_1]

        if self.noise is None:
            return raw_results
        else:
            results = []
            for result in raw_results:
                results.append(self.noise(result))
            return results
