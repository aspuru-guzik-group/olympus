#!/usr/bin/env python

import time
import numpy as np

from olympus.objects                   import ParameterVector
from olympus.planners.abstract_planner import AbstractPlanner
from olympus.planners.utils_planner import get_init_guess, get_bounds
from olympus.utils    import daemon

#===============================================================================

class SteepestDescent(AbstractPlanner):

    def __init__(self, goal='minimize', learning_rate=1e-3, dx=1e-5, random_seed=None,
                 init_guess=None, init_guess_method='random', init_guess_seed=None):
        """

        Args:
            goal:
            learning_rate:
            dx:
            random_seed:
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        """

        AbstractPlanner.__init__(**locals())
        self.has_optimizer = False

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
        params = self._project_into_domain(params)
        self.SUBMITTED_PARAMS.append(params)
        while len(self.RECEIVED_VALUES) == 0:
            time.sleep(0.1)
        value = self.RECEIVED_VALUES.pop(0)
        return value

    @daemon
    def start_optimizer(self):
        guess   = self.init_guess.copy()
        while True:
            func    = self._priv_evaluator(guess)
            dy      = np.zeros(len(guess))
            perturb = guess.copy()
            for index in range(len(guess)):
                perturb[index] += self.dx
                probed = self._priv_evaluator(perturb)
                dy[index] = (probed - func) / self.dx
                perturb[index] -= self.dx
            guess = guess - self.eta * dy
            guess = self._project_into_domain(domain)

    def _ask(self):
        if self.has_optimizer is False:
            self.start_optimizer()
            self.has_optimizer = True

        while len(self.SUBMITTED_PARAMS) == 0:
            print('SUBMITTED_PARAMS', len(self.SUBMITTED_PARAMS))
            time.sleep(0.1)
        params = self.SUBMITTED_PARAMS.pop(0)
        return ParameterVector().from_array(params, self.param_space)



#===============================================================================

if __name__ == '__main__':

    from olympus import Parameter, ParameterSpace
    param_space = ParameterSpace()
    param_space.add(Parameter(name='param_0'))
    param_space.add(Parameter(name='param_1'))

    planner = SteepestDescent(learning_rate=1e-3, dx=1e-5, random_seed=None, init_guess=None)
    planner.set_param_space(param_space=param_space)
    param = planner.ask()
    print('PARAM', param)
