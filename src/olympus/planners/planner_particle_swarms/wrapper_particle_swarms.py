#!/usr/bin/env python

import time
from olympus.objects  import ParameterVector
from olympus.planners import AbstractPlanner
from olympus.utils    import daemon
import numpy as np


class ParticleSwarms(AbstractPlanner):

    def __init__(self, goal='minimize', max_iters=10**8, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, particles=10):
        """
        Particle swarm optimizer.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            max_iters (int): The maximum number of iterations for the swarm to search.
            options (dict): ???
            particles (int): The number of particles in the swarm.
        """
        AbstractPlanner.__init__(**locals())
        self.has_optimizer = False
        self.is_converged  = False

    def _set_param_space(self, param_space):
        self.param_space = param_space

    def _tell(self, observations):
        self._params = observations.get_params(as_array = False)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)
        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])

    def _priv_evaluator(self, params_array):
        for params in params_array:
            params = self._project_into_domain(params)
            self.SUBMITTED_PARAMS.append(params)
        while len(self.RECEIVED_VALUES) < self.particles:
            time.sleep(0.1)
        measurements = np.array(self.RECEIVED_VALUES)
        self.RECEIVED_VALUES = []
        return measurements

    @daemon
    def create_optimizer(self):
        from pyswarms.single import GlobalBestPSO
        self.optimizer = GlobalBestPSO(
                n_particles=self.particles,
                options=self.options,
                dimensions=len(self.param_space))
        cost, pos = self.optimizer.optimize(self._priv_evaluator, iters=self.max_iters)
        self.is_converged = True

    def _ask(self):
        if self.has_optimizer is False:
            self.create_optimizer()
            self.has_optimizer = True

        while len(self.SUBMITTED_PARAMS) == 0:
            time.sleep(0.1)
            if self.is_converged:
                return ParameterVector().from_dict(self._params[-1])
        params = self.SUBMITTED_PARAMS.pop(0)
        return ParameterVector().from_array(params, self.param_space)
