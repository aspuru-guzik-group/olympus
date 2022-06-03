#!/usr/bin/env python

import numpy as np

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner


class Gryffin(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(
        self,
        goal="minimize",
        num_cpus=4,
        use_descriptors=True,
        auto_desc_gen=False,
        batches=1,
        sampling_strategies=[-1, 1],
        boosted=False,
        random_seed=None,
        acquisition_optimizer="adam",  #'genetic',
        verbosity=4,
    ):
        """
        A Bayesian optimization algorithm based on Bayesian Kernel Density estimation which supports
        categorical variables

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            num_cpus (int): Number of parallel cpus to use in multiprocessing
            auto_desc_gen (bool): switch on automatic descriptor refinement (for categorical params only)
            batches (int): number of parameter batches to return at each 'ask'
                iteration.
            sampling_strategies (list): list of sampling strategies to use. Each sampling
                strategy uses a different balance between exploration and exploitation.
            boosted (bool): whether to use a lower fidelity approximation in
                regions of low density during kernel density estimation. Setting
                this to True reduces the run time of the planner.
            random_seed (int): random seed
            acquisition_optimizer (str): algorithm to optimize the acquisition function - currently
                supported are 'adam' and 'genetic'
        """
        AbstractPlanner.__init__(**locals())
        # check for and set random seed
        if self.random_seed is None:
            self.random_seed = np.random.randint(1, int(1e7))

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == "continuous":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "size": 1,
                    "low": param.low,
                    "high": param.high,
                }
            elif param.type == "discrete":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "size": 1,
                    "low": param.low,
                    "high": param.high,
                    "stride": param.stride,
                }
            elif param.type == "categorical":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "size": 1,
                    "options": param.options,
                    "descriptors": param.descriptors,
                }
            else:
                raise NotImplementedError(
                    f"Parameter type {param.type} for {param.name} not implemnted"
                )
            self._param_space.append(param_dict)

    def _tell(self, observations):

        self._params = observations.get_params()
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )

        self._observations = []
        for obs_ix, param in enumerate(self._params):
            # format the observations
            obs = {}
            for param_val, space_true in zip(
                param, self.param_space.parameters
            ):
                if space_true.type == "categorical":
                    obs[space_true.name] = param_val
                else:
                    obs[space_true.name] = np.float(param_val)
            obs["obj"] = self._values[obs_ix]
            obs["obj"] = np.squeeze(obs["obj"])

            self._observations.append(obs)

    def _get_gryffin_instance(self):

        from gryffin import Gryffin as ActualGryffin

        params = self.param_space.item()
        # gryffin expects category detail dict for categorical parameters
        # with options as keys and descriptors as values, e.g.
        # {'A':[1, 2], 'B':[5,6]}
        # if there are no descriptors, then use None or empty lists as the values
        # {'A':None, 'B':None}`` or ``{'A':[], 'B':[]}
        for param in params:
            if param["type"] == "categorical":
                if self.use_descriptors:
                    # user elects to use descriptors
                    category_details = {
                        str(param["options"][ix]): param["descriptors"][ix]
                        for ix in range(len(param["options"]))
                    }
                else:
                    # user overides the use of descriptors
                    category_details = {
                        str(param["options"][ix]): None for ix in range(len(param["options"]))
                    }
                param["category_details"] = category_details
                del param["options"]
                del param["descriptors"]

        config = {
            "general": {
                "num_cpus": self.num_cpus,
                "auto_desc_gen": self.auto_desc_gen,
                "batches": self.batches,
                "sampling_strategies": self.batches,
                "boosted": self.boosted,
                "caching": True,
                "random_seed": self.random_seed,
                "acquisition_optimizer": self.acquisition_optimizer,
                "verbosity": self.verbosity,
            },
            "parameters": params,
            "objectives": [{"name": "obj", "goal": self.goal[:3]}],
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


# DEBUG:
if __name__ == "__main__":

    from olympus import Campaign
    from olympus.datasets import Dataset

    d = Dataset(kind="perovskites")

    planner = Gryffin(goal="minimize")
    planner.set_param_space(d.param_space)

    campaign = Campaign()
    campaign.set_param_space(d.param_space)

    BUDGET = 200
    for i in range(BUDGET):
        print(f"ITERATION : ", i)

        sample = planner.recommend(campaign.observations)
        print("SAMPLE : ", sample)

        measurement = d.run([sample], return_paramvector=False)[0]
        print("MEASUREMENT : ", measurement)

        campaign.add_observation(sample, measurement)
