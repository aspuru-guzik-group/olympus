#!/usr/bin/env python

import numpy as np

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner


class Entmoot(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(
        self,
        goal="minimize",
        random_seed=None,
    ):

        """
        ENTMOOT (ENsemble Tree MOdel Optimization Tool) is a novel framework
        to handle tree-based models in Bayesian optimization applications.
        Gradient-boosted tree models from lightgbm are combined with a
        distance-based uncertainty measure in a deterministic global
        optimization framework to optimize black-box functions.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'.
                Default is 'minimize'.
        """
        AbstractPlanner.__init__(**locals())
        # check for and set the random seed
        if not self.random_seed:
            self.random_seed = np.random.randint(1, int(1e7))

    def _set_param_space(self, param_space):

        return None

    def _tell(self, observations):

        return None

    def _ask(self):

        return None
