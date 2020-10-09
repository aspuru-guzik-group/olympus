#!/usr/bin/env python

import numpy as np
from olympus.noises import AbstractNoise


class GaussianNoise(AbstractNoise):

    def __init__(self, scale=1):
        """Gaussian noise.

        Args:
            scale (float): Standard deviation of the Gaussian. Default is 1.
        """
        AbstractNoise.__init__(**locals())

    def _add_noise(self, value):
        noisy_value = np.random.normal(loc=value, scale=self.scale)
        return noisy_value
