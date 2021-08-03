#!/usr/bin/env python

import numpy as np

from olympus.planners              import AbstractPlanner
from olympus.objects               import ParameterVector
from olympus.objects.object_config import Config


class Gryffin(AbstractPlanner):

    def __init__(
        self,
        goal='minimize',
        num_cpus=4,
        auto_desc_gen=False,
        batches=1,
        sampling_strategies=[-1, 1],
        boosted=False,
        caching=True,
        random_seed=None,
        acquisition_optimizer='adam',#'genetic',
        verbosity=4,
    ):
        """
        Categorical Bayesian optimization based on Bayesian deep learning and Bayesian KDE
        Args:
        """
        AbstractPlanner.__init__(**locals())
        # check for and set random seed
        if self.random_seed is None:
            self.random_seed = np.random.randint(1, int(1e7))

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "size": 1,
                    "low": param.low,
                    "high": param.high,
                }
            else:
                raise NotImplementedError(f'Parameter type {param.type} for {param.name} not implemnted')
            self._param_space.append(param_dict)


    def _tell(self, observations):

        self._params = observations.get_params()
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)

        self._observations = []
        for obs_ix, param in enumerate(self._params):
            # format the observations
            obs = {}
            for param_val, space_true in zip(param, self.param_space.parameters):

                obs[space_true.name] = np.float(param_val)
            obs["obj"] = self._values[obs_ix]
            obs["obj"] = np.squeeze(obs["obj"])

            self._observations.append(obs)


    def _get_gryffin_instance(self):
        from gryffin import Gryffin as ActualGryffin

        params = self.param_space.item()
        config = {
            "general": {
                "num_cpus": self.num_cpus,
                "auto_desc_gen": self.auto_desc_gen,
                "batches": self.batches,
                "sampling_strategies": self.batches,
                "boosted":  self.boosted,
                "caching": self.caching,
                "random_seed": self.random_seed,
                "acquisition_optimizer": self.acquisition_optimizer,
                "verbosity": self.verbosity
                },
            "parameters": params,
            "objectives": [{'name': 'obj', 'goal': self.goal[:3]}],
        }

        self.gryffin = ActualGryffin(config_dict=config)


    def _ask(self):

        self._get_gryffin_instance()

        # check which params to return - select alternating sampling strategy
        select_ix = len(self._values) % len(self.sampling_strategies)
        sampling_strategy = self.sampling_strategies[select_ix]

        # query for new parameters
        params = self.gryffin.recommend(
            observations=self._observations,
            sampling_strategies=[sampling_strategy],
        )
        param = params[0]

        return ParameterVector().from_dict(param, self.param_space)
