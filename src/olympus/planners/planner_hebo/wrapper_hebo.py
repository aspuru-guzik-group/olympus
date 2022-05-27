#!/usr/bin/env python

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner

from .hebo_utils import propose_randomly


class Hebo(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(
        self,
        goal="minimize",
        batch_size=1,
        model_name="gp",  # ['gp', 'gpy', 'gpy_mlp', 'rf']
        num_init_design=2,  # this must be >= 2
        init_design_strategy="random",
        random_seed=None,
        model_config=None,
    ):

        """
        Bayesian optimsation library developped by Huawei Noahs Ark
        Decision Making and Reasoning (DMnR) lab.

        HBEO minimizes the objective function by default

        Args:
                goal (str): The optimization goal, either 'minimize' or 'maximize'.
                        Default is 'minimize'.
        """
        AbstractPlanner.__init__(**locals())
        # check for and set the random seed
        if not self.random_seed:
            self.random_seed = np.random.randint(1, int(1e7))

    def _set_param_space(self, param_space):
        self._param_space = []
        self.discrete_param_encodings = {}
        for param in param_space:
            if param.type == "continuous":
                param_dict = {
                    "name": param.name,
                    "type": "num",
                    "lb": param.low,
                    "ub": param.high,
                }
            elif param.type == "discrete":
                # TODO: map the discrete parameters onto integers
                self.olympus_encoding = np.arange(
                    param.low, param.high, param.stride
                )  # actual parameter values
                self.int_encoding = np.arange(
                    len(self.olympus_encoding)
                )  # integers from 0, ..., num_opts
                self.discrete_param_encodings[param.name] = {
                    i: o
                    for i, o in zip(self.int_encoding, self.olympus_encoding)
                }
                param_dict = {
                    "name": param.name,
                    "type": "int",
                    "lb": param.low,
                    "ub": param.high,
                }

            elif param.type == "categorical":
                param_dict = {
                    "name": param.name,
                    "type": "cat",
                    "categories": param.options,
                    # NOTE: does not support descritpors, at least as far as I know...
                }
            else:
                raise NotImplementedError(
                    f"Parameter type {param.type} for {param.name} not implemnted"
                )

            self._param_space.append(param_dict)

    def _build_hebo(self):

        if self.model_config == None:
            if self.model_name == "gp":
                self.model_config = {
                    "lr": 0.01,
                    "num_epochs": 100,
                    "verbose": False,
                    "noise_lb": 8e-4,
                    "pred_likeli": False,
                }
            elif self.model_name == "gpy":
                self.model_config = {
                    "verbose": False,
                    "warp": True,
                    "space": self._param_space,
                }
            elif self.model_name == "gpy_mlp":
                self.model_config = {"verbose": False}
            elif self.model_name == "rf":
                self.model_config = {"n_estimators": 20}
            else:
                self.model_config = {}

        self.opt = HEBO(
            DesignSpace().parse(self._param_space),
            model_name=self.model_name,
            rand_sample=None,
            model_config=self.model_config,
        )

    def _tell(self, observations):
        """hebo takes a dataframe of parameters (# measurements, # parameters) for parameters
        and np.ndarray (# measurements, # objectives) for the objectives

        self.opt.observe(parameters, objectives)
        """
        # make a new version of HEBO at every iterations (i.e. one with
        # no observations. This will allow us to re-write the entire optimization
        # history in the case of MOO with a-priori scalarizing functions)

        self._build_hebo()

        self._params = observations.get_params()
        self._values = observations.get_values(
            as_array=True,
            opposite=self.flip_measurements,
        )

        if len(self._params) >= 1:
            # change the params to a dataframe
            params_dict = {}
            for idx, param in enumerate(self.param_space):
                params_dict[param.name] = self._params[:, idx]

            self.hebo_params = pd.DataFrame(params_dict)

            # reshape the values
            self.hebo_values = np.array(self._values).reshape(-1, 1)

            self.opt.observe(self.hebo_params, self.hebo_values)

    def _ask(self):

        if len(self._params) < self.num_init_design:
            # init design strategy (select points randomly)
            # samples = self.opt.quasi_sample(1, fix_input=None)
            sample, raw_sample = propose_randomly(1, self.param_space)

            return_params = ParameterVector().from_list(
                raw_sample[0], self.param_space
            )

        else:
            # will be a dataframe
            samples = self.opt.suggest(n_suggestions=self.batch_size)

            # generate a dictionary and convert to olympus
            samples = samples.to_dict("r")[0]
            return_params = ParameterVector().from_dict(
                samples, self.param_space
            )

        return return_params


# -----------
# DEBUGGING
# -----------
if __name__ == "__main__":
    PARAM_TYPE = "mixed"

    NUM_RUNS = 40

    from olympus.campaigns import Campaign, ParameterSpace
    from olympus.objects import (
        ParameterCategorical,
        ParameterContinuous,
        ParameterDiscrete,
    )
    from olympus.surfaces import Surface

    def surface(x):
        return np.sin(8 * x)

    if PARAM_TYPE == "continuous":
        param_space = ParameterSpace()
        param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
        param_space.add(param_0)

        planner = Hebo(goal="minimize")
        planner.set_param_space(param_space)

        campaign = Campaign()
        campaign.set_param_space(param_space)

        BUDGET = 24

        for num_iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            print(f"ITER : {num_iter}\tSAMPLES : {samples}")
            # for sample in samples:
            sample_arr = samples.to_array()
            measurement = surface(sample_arr.reshape((1, sample_arr.shape[0])))
            campaign.add_observation(sample_arr, measurement[0])

    elif PARAM_TYPE == "categorical":

        surface_kind = "CatDejong"
        surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

        campaign = Campaign()
        campaign.set_param_space(surface.param_space)

        planner = Hebo(goal="minimize")
        planner.set_param_space(surface.param_space)

        OPT = ["x10", "x10"]

        BUDGET = 442

        for iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            print(f"ITER : {iter}\tSAMPLES : {samples}")
            # sample = samples[0]
            sample_arr = samples.to_array()
            measurement = np.array(surface.run(sample_arr))
            campaign.add_observation(sample_arr, measurement[0])

            if [sample_arr[0], sample_arr[1]] == OPT:
                print(f"FOUND OPTIMUM AFTER {iter+1} ITERATIONS!")
                break

    elif PARAM_TYPE == "mixed":

        def surface(params):
            return np.random.uniform()

        param_space = ParameterSpace()
        # continuous parameter 0
        param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
        param_space.add(param_0)

        # continuous parameter 1
        param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
        param_space.add(param_1)

        # categorical parameter 2
        param_2 = ParameterCategorical(name="param_2", options=["a", "b", "c"])
        param_space.add(param_2)

        # categorcial parameter 3
        param_3 = ParameterCategorical(name="param_3", options=["x", "y", "z"])
        param_space.add(param_3)

        campaign = Campaign()
        campaign.set_param_space(param_space)

        planner = Hebo(goal="minimize")
        planner.set_param_space(param_space)

        BUDGET = 20

        for iter in range(BUDGET):

            samples = planner.recommend(campaign.observations)
            # sample = samples[0]
            sample_arr = samples.to_array()
            measurement = surface(sample_arr)
            print(
                f"ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}"
            )
            campaign.add_observation(sample_arr, measurement)
