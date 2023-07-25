#!/usr/bin/env python

import ConfigSpace as CS
import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.optimizer.acquisition import EI, LCB, PI
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner
from olympus.utils import daemon


class Smac(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    KNOWS_BOUNDS = True

    def __init__(
        self,
        goal="minimize",
        model_type="rf",
        acquisition_function="ei",
        batch_size=1,
        random_seed=None,
    ):
        """
        Bayesian optimization with SMAC3
        """
        AbstractPlanner.__init__(**locals())
        # check for and set the random seed
        if not self.random_seed:
            self.random_seed = np.random.randint(1, int(1e7))

        self._is_smac_built = False
        self.has_minimizer = False
        self.is_converged = False

    def _set_param_space(self, param_space):
        self._param_space = []
        self.discrete_param_encodings = {}

        for param in param_space:
            if param.type == "continuous":
                self._param_space.append(
                    UniformFloatHyperparameter(
                        param.name,  # name
                        param.low,  # lower bound
                        param.high,  # upper bound
                        default_value=param.high - param.high / 2,
                    )
                )
            elif param_type == "discrete":
                # map the discrete parameters onto integers
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
                self._param_space.append(
                    UniformIntegerHyperparameter(
                        param.name,  # name
                        0,  # lower bound
                        len(self.int_encoding) - 1,  # higher bound
                        default_value=0,
                    )
                )

            elif param.type == "categorical":
                self._param_space.append(
                    CategoricaHyperparameter(
                        param.name,  # name
                        param.options,  # options
                        default_value=param.options[0],
                    )
                )

        # makee configuration space
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters(self._param_space)

    @daemon
    def create_minimizer(self):
        _ = self.smac.optimize()
        self.is_converged = True

    def _priv_evaluator(self, params):
        if self.KNOWS_BOUNDS is False:
            params = self._project_into_domain(params)
        self.SUBMITTED_PARAMS.append(params)
        while len(self.RECEIVED_VALUES) == 0:
            time.sleep(0.1)
        value = self.RECEIVED_VALUES.pop(0)
        return value

    def _build_smac(self):
        af_map = {"ei": EI, "lcb": LCB, "pi": PI}

        # generate the scenario
        self.scenario = Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": 1e6,
                "cs": self.cs,
                "deterministic": False,
            }
        )

        self.smac = SMAC4AC(
            scenario=self.scenario,
            model_type=self.model_type,
            rng=np.random.RandomState(self.random_seed),
            acquisition_function=af_map[self.acquisition_function],
            tae_runner=self._priv_evaluator,
        )

    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )

        if not self._is_smac_built:
            self._build_smac()

        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])

    def _ask(self):
        if not self.has_minimizer:
            self.create_minimizer()
            self.has_minimizer = True

        while len(self.SUBMITTED_PARAMS) == 0:
            time.sleep(0.1)
            if self.is_converged:
                return ParameterVector().from_dict(self._params[-1])
        params = self.SUBMITTED_PARAMS.pop(0)
        return ParameterVector(array=params, param_space=self.param_space)


# -----------
# DEBUGGING
# -----------
if __name__ == "__main__":
    PARAM_TYPE = "continuous"

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

        planner = Smac(goal="minimize")
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

        planner = Smac(goal="minimize")
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

        planner = Smac(goal="minimize")
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
