#!/usr/bin/env python

import numpy as np
from scipy.stats import gamma
from olympus.noises import AbstractNoise


class GammaNoise(AbstractNoise):

    def __init__(self, scale=1, lower_bound=0):
        """Gamma-distributed noise parametrised by its standard deviation (``scale``) and its mode.

        Args:
            scale (float): Standard deviation of the Gaussian. Default is 1.
            lower_bound (float): Lower bound of the Gamma distribution. The distribution will not have support below this
                lower bound. Default is 0.
        """
        AbstractNoise.__init__(**locals())

    def _add_noise(self, value):
        # value = mode
        value = value - self.lower_bound
        var = self.scale ** 2.
        s = np.sqrt(var + (value ** 2.) / 4.) - value / 2.
        k = value / s + 1.
        noisy_value = gamma.rvs(a=k, loc=self.lower_bound, scale=s)
        return noisy_value
