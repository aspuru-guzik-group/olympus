#!/usr/bin/env python

import numpy as np

#from dragonfly import load_config_file
from dragonfly.exd.cp_domain_utils import load_config
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly.opt import random_optimiser, gp_bandit

from olympus.planners import AbstractPlanner
from olympus.objects import ParameterVector



class Dragonfly(AbstractPlanner):

    PARAM_TYPES = ['continuous', 'discrete', 'categorical']

    def __init__(
        self,
        goal='minimize',
        opt_method='bo',
        random_seed=None,


    ):
        """
        Scalable Bayesian optimization as implemented in the Dragonfly package:
        https://github.com/dragonfly/dragonfly

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
        """
        AbstractPlanner.__init__(**locals())
        # check for and set the random seed
        if not self.random_seed:
            self.random_seed = np.random.randint(1, int(1e7))

        self._has_dragonfly_domain = False


    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {
                    "name": param.name,
                    "type": "float",
                    "min": param.low,
                    "max": param.high,
                }
            elif param.type == 'discrete':
                # discrete numeric
                param_dict = {
                    "name": param.name,
                    "type": "discrete_numeric",
                    "items": f"{param.low}:{param.stride}:{param.high}",
                }
            elif param.type == 'categorical':
                # discrete
                param_dict = {
                    "name": param.name,
                    "type": "discrete",
#                    "items": "-".join([opt for opt in param.options]),
                    "items": param.options,
                }
            else:
                raise NotImplementedError(f'Parameter type {param.type} for {param.name} not implemnted')

            self._param_space.append(param_dict)


    def _build_dragonfly():

        config_params = {"domain": self._param_space}
        config = load_config(config_params)





    def _tell(self, observations):
        self._params = observations.get_params()
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements,
        )



        return None

    def _ask(self):

        return None
