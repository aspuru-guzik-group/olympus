#!/usr/bin/env python

import time

import numpy as np

from olympus import Logger
from olympus.objects import ParameterVector
from olympus.planners.abstract_planner import AbstractPlanner

# ===============================================================================


class LatinHypercube(AbstractPlanner):

    PARAM_TYPES = ["continuous"]

    def __init__(self, goal="minimize", budget=None):
        AbstractPlanner.__init__(**locals())
        self.has_optimizer = False
        self.budget = budget

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.dim = len(self.param_space)

    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )

    def _create_optimizer(self):
        from pyDOE import lhs

        if self.budget is None:
            message = (
                f"Please provide a number of samples for this planner. Given no number of samples provided, "
                f"falling back to setting `budget` to {len(self.param_space)}"
            )
            Logger.log(message, "WARNING")
            self.budget = len(self.param_space)
        self.samples = lhs(len(self.param_space), samples=self.budget)
        for index, param in enumerate(self.param_space):
            self.samples[:, index] = (param.high - param.low) * self.samples[
                :, index
            ] + param.low
        self.samples = list(self.samples)
        self.has_optimizer = True

    def _ask(self):
        if not self.has_optimizer:
            self._create_optimizer()
        param = self.samples.pop(0)
        return ParameterVector().from_array(param, self.param_space)
