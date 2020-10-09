#!/usr/bin/env python

import numpy as np

from olympus.planners              import AbstractPlanner
from olympus.objects               import ParameterVector
from olympus.objects.object_config import Config


class Phoenics(AbstractPlanner):

    def __init__(self, goal='minimize', batches=1, boosted=True, parallel=True, sampling_strategies=2):
        """
        A Bayesian optimization algorithm based on Bayesian Kernel Density estimation.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            batches (int): number of parameter batches to return at each 'ask'
                iteration.
            boosted (bool): whether to use a lower fidelity approximation in
                regions of low density during kernel density estimation. Setting
                this to True reduces the run time of the planner.
            parallel (bool): whether to run the code in parallel.
            sampling_strategies (int): number of sampling strategies to use.
                Each sampling strategy uses a different balance between
                exploration and exploitation.
        """
        AbstractPlanner.__init__(**locals())

        self._observations = []
        self._counter = 0

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'size': 1, 'low': param.low, 'high': param.high}
            else:
                raise NotImplementedError
            self._param_space.append(param_dict)

    def _tell(self, observations):
        self._params       = observations.get_params()
        self._values       = observations.get_values(as_array=True, opposite=self.flip_measurements)
        self._observations = []
        for _, param in enumerate(self._params):
            obs = ParameterVector().from_array(param, self.param_space).to_dict()
            for key, value in obs.items():
                obs[key] = np.array([value])
            obs['obj'] = self._values[_]
            # WARNING: THIS IS JUST A HACK
            # NOTE: we should make sure we are not adding unnecessary dimensions to this array
            obs['obj'] = np.squeeze(obs['obj'])
            self._observations.append(obs)

    def _get_phoenics_instance(self):
        from phoenics import Phoenics as ActualPhoenics
        params = self.param_space.item()
        for param in params:
            param['size'] = 1
        config_dict = {
            'general': self.config.to_dict(),
            'parameters': params,
            'objectives': [{'name': 'obj', 'goal': 'minimize'}]}
        self.phoenics = ActualPhoenics(config_dict=config_dict)

    def _ask(self):
        self._get_phoenics_instance()
        params = self.phoenics.recommend(observations=self._observations)
        # check which param to return
        param = params[self._counter % len(params)]
        for key, value in param.items():
            param[key] = np.squeeze(value)
        self._counter += 1
        param = ParameterVector().from_dict(param)
        return param
