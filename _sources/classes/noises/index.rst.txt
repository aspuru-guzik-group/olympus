.. _noises:

Noises
======

**Olympus** provides a way to inject noise explicitly into the benchmarks. At the moment, only adding noise to the
output of the :ref:`surfaces` is supported.

Noise objects can be accessed by importing the specific noise class
from the ``olympus.noises`` module, or via the ``Noise`` function. For instance, to load an object that injects Gaussian
noise, you can use the ``Noise`` function::

    from olympus import Noise
    noise = Noise(kind='GaussianNoise', scale=1.2)

The above is equivalent to importing the class ``GaussianNoise`` directly::

    from olympus.noises import GaussianNoise
    noise = GaussianNoise(scale=1.2)

This latter approach, however, allows for more control over the details of the noise object. For instance, the
Gamma-distributed noise also allows to set a lower bound for its support that is different from zero::

    from olympus.noises import GammaNoise
    noise = GammaNoise(scale=1, lower_bound=-5)

Once a noise instance is defined as above, it can be passed to one of the :ref:`surfaces` to enable obtaining noisy evaluations::

    from olympus.surfaces import HyperEllipsoid
    from olympus.noises import GaussianNoise

    noise = GaussianNoise(scale=2)
    noisy_surface = HyperEllipsoid(param_dim=1, noise=noise)

    # evaluate the same location three times
    for i in range(3):
        print(noisy_surface.run(0.5))

    >>> [[0.7303638140484728]]
    >>> [[0.6398992941066822]]
    >>> [[0.29602882702845845]]


If you wanted to define your own, customized noise class, you can do it as follows by inheriting from the ``AbstractNoise``
class::

    from olympus.noises import AbstractNoise

    class CustomNoise(AbstractNoise):

        def __init__(self, scale=1):
            AbstractNoise.__init__(**locals())

        def _add_noise(self, value):
            # ----------------------------------------------
            # Here you define the noise as you desire, e.g.:
            # ----------------------------------------------
            if value > 2:
                noisy_value = value + self.scale
            else:
                noisy_value = value - self.scale

            return noisy_value

Noise Classes
-------------

These are the types of noise currently available in **Olympus**.

.. toctree::
   :maxdepth: 1

   gaussian_noise
   uniform_noise
   gamma_noise

Noise Function
--------------

.. currentmodule:: olympus.noises

.. autofunction:: Noise
   :noindex:
