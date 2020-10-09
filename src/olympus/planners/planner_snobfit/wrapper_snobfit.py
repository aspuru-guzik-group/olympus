#!/usr/bin/env python

import time
import numpy as np

from olympus.objects  import ParameterVector
from olympus.planners import AbstractPlanner
from olympus.planners.utils_planner import get_init_guess, get_bounds
from olympus.utils    import daemon

#===============================================================================

class Snobfit(AbstractPlanner):

    SUBMITTED_PARAMS = []
    RECEIVED_VALUES  = []

    def __init__(self, goal='minimize',
                 init_guess=None, init_guess_method='random', init_guess_seed=None):
        '''
        Args:
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        '''
        AbstractPlanner.__init__(**locals())
        self.has_optimizer = False
        self.is_converged  = False
        self.budget        = 10**8

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.bounds      = get_bounds(param_space)
        if self.init_guess is None:
            self.init_guess = get_init_guess(param_space, method=self.init_guess_method, random_seed=self.init_guess_seed)

    def _tell(self, observations):
        self._params = observations.get_params(as_array = False)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)
        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])

    def _priv_evaluator(self, param):
        self.SUBMITTED_PARAMS.append(param)
        while len(self.RECEIVED_VALUES) == 0:
            time.sleep(0.1)
        measurement = self.RECEIVED_VALUES.pop(0)
        return np.squeeze(measurement)

    @daemon
    def create_optimizer(self):
        from SQSnobFit import minimize, optset
        options = optset(maxmp=2+6)
        _, __ = minimize(self._priv_evaluator, x0=self.init_guess, bounds=self.bounds, budget=self.budget)#, options=options)
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
