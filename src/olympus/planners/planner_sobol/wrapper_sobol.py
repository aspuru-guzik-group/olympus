#!/usr/bin/env python

import time

import numpy as np
from sobol_seq import i4_sobol

from olympus.objects import ParameterVector
from olympus.planners.abstract_planner import AbstractPlanner

# ===============================================================================


class Sobol(AbstractPlanner):

    PARAM_TYPES = ["continuous"]

    def __init__(self, goal="minimize", budget=None, seed=None):
        AbstractPlanner.__init__(**locals())
        if seed is None:
            self.seed = np.random.randint(low=0, high=1000)

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.dim = len(self.param_space)

    def _tell(self, observations):
        # random search does not need observations
        pass

    def _ask(self):
        vector, self.seed = i4_sobol(self.dim, self.seed)
        for index, param in enumerate(self.param_space):
            vector[index] = (param.high - param.low) * vector[
                index
            ] + param.low
        return ParameterVector().from_array(vector, self.param_space)
