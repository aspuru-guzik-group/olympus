#!/usr/bin/env python

#===============================================================================

import time

from olympus.objects                import ParameterVector
from olympus.planners               import AbstractPlanner
from olympus.planners.utils_planner import get_init_guess, get_bounds
from olympus.utils                  import daemon

#===============================================================================

class WrapperScipy(AbstractPlanner):

    KNOWS_BOUNDS     = False

    def __init__(self, *args, **kwargs):
        AbstractPlanner.__init__(self, *args, **kwargs)
        self.has_minimizer = False
        self.is_converged  = False

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.bounds      = get_bounds(param_space)
        if self.init_guess is None:
            self.init_guess = get_init_guess(param_space, method=self.init_guess_method, random_seed=self.init_guess_seed)


    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)
        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])

    def _priv_evaluator(self, params):
        if self.KNOWS_BOUNDS is False:
            params = self._project_into_domain(params)
        self.SUBMITTED_PARAMS.append(params)
        while len(self.RECEIVED_VALUES) == 0:
            time.sleep(0.1)
        value = self.RECEIVED_VALUES.pop(0)
        return value

    @daemon
    def create_minimizer(self):
        from scipy.optimize import minimize
        config = self.config.to_dict().copy()
        for attr in ['init_guess', 'init_guess_seed', 'init_guess_method']:
            if attr in config: del config[attr]
        if self.KNOWS_BOUNDS:
            _ = minimize(self._priv_evaluator, x0=self.init_guess, bounds=self.bounds, method=self.METHOD, options=config)
        else:
            _ = minimize(self._priv_evaluator, x0=self.init_guess, method=self.METHOD, options=config)
        self.is_converged = True

    def _ask(self):
        if self.has_minimizer is False:
            self.create_minimizer()
            self.has_minimizer = True

        while len(self.SUBMITTED_PARAMS) == 0:
            time.sleep(0.1)
            if self.is_converged:
                return ParameterVector(dict=self._params[-1])
        params = self.SUBMITTED_PARAMS.pop(0)
        return ParameterVector(array=params, param_space=self.param_space)
