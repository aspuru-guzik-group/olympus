#!/usr/bin/env python

from olympus.surfaces import GaussianMixture
from olympus import Logger


class Matterhorn(GaussianMixture):

    def __init__(self, noise=None):
        """Gaussian Mixture surface derived from the GaussianMixture generator using ``random_seed=4478``.

        Args:
            noise (Noise): Noise object that injects noise into the evaluations of the surface. Default is None.
        """
        peak = 4478
        GaussianMixture.__init__(self, param_dim=2, num_gauss=7, cov_scale=0.1, diagonal_cov=False,
                                 noise=noise, random_seed=peak)

    @property
    def minima(self):
        # TODO: find minimum numerically and hardcode
        message = 'Unknown minima: these need to be found numerically'
        Logger.log(message, 'WARNING')
        return None

    @property
    def maxima(self):
        # TODO: find minimum numerically and hardcode
        message = 'Unknown maxima: these need to be found numerically'
        Logger.log(message, 'WARNING')
        return None


