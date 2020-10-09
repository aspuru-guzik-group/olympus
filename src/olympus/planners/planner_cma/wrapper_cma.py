#!/usr/bin/env python

import time
import numpy as np
from olympus.objects  import ParameterVector
from olympus.planners import AbstractPlanner
from olympus.planners.utils_planner import get_init_guess, get_bounds
from olympus.utils    import daemon


class Cma(AbstractPlanner):

    def __init__(self, goal='minimize', stddev=0.5, init_guess=None,
                 init_guess_method='random', init_guess_seed=None):
        """
        Covariance Matrix Adaptation Evolution Strategy (CMA-ES) planner.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            stddev (float): Initial standard deviation.
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        """
        AbstractPlanner.__init__(**locals())
        self.has_optimizer = False
        self.is_converged  = False

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.bounds      = np.transpose(get_bounds(param_space)).tolist()
        if self.init_guess is None:
            self.init_guess = get_init_guess(param_space, method=self.init_guess_method, random_seed=self.init_guess_seed)

    def _tell(self, observations):
        self._params = observations.get_params(as_array = False)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)
        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])

    def _priv_evaluator(self, params):
        params = self._project_into_domain(params)
        self.SUBMITTED_PARAMS.append(params)
        while len(self.RECEIVED_VALUES) == 0:
            time.sleep(0.1)
        measurements = self.RECEIVED_VALUES.pop(0)
        return measurements

    @daemon
    def create_optimizer(self):
        from cma import CMAEvolutionStrategy
        opts = {'bounds': self.bounds}
        stddevs = (self.param_space.param_uppers - self.param_space.param_lowers) * self.stddev + self.param_space.param_lowers
        optimizer = CMAEvolutionStrategy(self.init_guess, np.amin(stddevs), inopts=opts)
        _ = optimizer.optimize(self._priv_evaluator)
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
