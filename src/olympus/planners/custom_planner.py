#!/usr/bin/env python

from . import AbstractPlanner


class CustomPlanner(AbstractPlanner):
    def __init__(self, goal='minimize'):
        """Parent class to be used to create custom Planner classes.

        Args:
            goal: The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
        """
        AbstractPlanner.__init__(**locals())

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
                self._param_space.append(param_dict)

    def _tell(self, observations):
        self._params = observations.get_params(as_array=True)
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)