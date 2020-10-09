#!/usr/bin/env python

import time
import numpy as np 

from olympus.objects                   import ParameterVector
from olympus.planners.abstract_planner import AbstractPlanner


#===============================================================================

class RandomSearch(AbstractPlanner):    

    def __init__(self, goal='minimize', seed=None):
        AbstractPlanner.__init__(**locals())
        if self.seed is not None:
            np.random.seed(seed)

    def _set_param_space(self, param_space):
        self.param_space = param_space

    def _tell(self, observations):
        # random search does not need observations
        pass

    def _ask(self):
        vals = []
        for param in self.param_space:
            if param.type == 'continuous': 
                val = np.random.uniform(low=param.low, high=param.high)
            else:
                raise NotImplementedError
            vals.append(val)
        vals = np.array(vals)
        return ParameterVector().from_array(vals, self.param_space)



