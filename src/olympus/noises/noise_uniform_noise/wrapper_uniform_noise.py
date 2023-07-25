#!/usr/bin/env python

import numpy as np

from olympus.noises import AbstractNoise


class UniformNoise(AbstractNoise):
    def __init__(self, scale=1):
        """Gaussian noise.

        Args:
            scale (float): Half-range :math:`r_{1/2}` of the uniform distribution. A random sample will be drawn from
                the half-open interval :math:`[\mu - r_{1/2}, \mu + r_{1/2})`, where :math:`\mu` the argument ``value``. Default is 1.
        """
        AbstractNoise.__init__(**locals())

    def _add_noise(self, value):
        low = value - self.scale * 0.5
        high = value + self.scale * 0.5
        noisy_value = np.random.uniform(low=low, high=high)
        return noisy_value
