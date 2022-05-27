#!/usr/bin/env python

import itertools
import random
import time

import numpy as np

from olympus import Logger
from olympus.objects import ParameterVector
from olympus.planners.abstract_planner import AbstractPlanner

# ===============================================================================


class RandomSearch(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(self, goal="minimize", seed=None):
        AbstractPlanner.__init__(**locals())
        if self.seed is None:
            self.seed = np.random.randint(1, int(1e7))
        np.random.seed(self.seed)

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self._treat_param_types()

    def _tell(self, observations):
        # register iteration number
        # TODO: this will only work for a batch size of 1...
        self.iteration = len(observations.get_values(as_array=True))
        # random search is not informed by previous observations

    def _treat_param_types(self):
        opts_ = []
        self.has_continuous = False
        self.has_cat_discr = False

        for param in self.param_space:
            if param.type == "continuous":
                self.has_continuous = True
            elif param.type == "categorical":
                opts_.append(param.options)
                self.has_cat_discr = True
            elif param.type == "discrete":
                opts_.append(np.arange(param.low, param.high, param.stride))
                self.has_cat_discr = True
            else:
                raise NotImplementedError(
                    f"Parameter type {param.type} for {param.name} not implemented"
                )
        if self.has_cat_discr:
            # generate set of possible categorcial and/or discrete options
            self.opts = np.array(list(itertools.product(*opts_)))
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.opts)
            # total number of options
            self.num_opts = len(self.opts)

    def _ask(self):
        vals = []
        if self.has_cat_discr:
            # sample categorical and discrete parameters together
            if self.has_continuous:
                # do not need to remove parameters, sample randomly with replacement
                indices = np.arange(len(self.opts))
                np.random.shuffle(indices)
                cat_discr_vals = self.opts[
                    indices[0]
                ]  # np.random.choice(self.opts)
            else:
                # fully categorical space, remove selected options to avoid
                # iterate through by iteration nunber
                # TODO: treat the case where the number of requested observations
                # out number the number of options in the dataset
                cat_discr_vals = self.opts[self.iteration]

        cat_discr_ind = 0
        for param in self.param_space:
            if param.type == "continuous":
                val = np.random.uniform(low=param.low, high=param.high)
            elif param.type in ["categorical", "discrete"]:
                val = cat_discr_vals[cat_discr_ind]
                cat_discr_ind += 1
            else:
                raise NotImplementedError
            vals.append(val)

        return ParameterVector().from_list(vals, self.param_space)


# DEBUG
if __name__ == "__main__":

    from olympus import Campaign
    from olympus.datasets import Dataset

    d = Dataset(kind="cross_barrel")

    planner = RandomSearch(goal="maximize")
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
