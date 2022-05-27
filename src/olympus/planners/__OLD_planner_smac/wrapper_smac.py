#!/usr/bin/env python

import time

import numpy as np

from olympus import __scratch__
from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner
from olympus.utils import daemon

# ===============================================================================


class Smac(AbstractPlanner):
    def __init__(self, goal="minimize", rng=None):
        AbstractPlanner.__init__(**locals())
        self.seed = 100691
        self.rng = np.random.RandomState()
        self.has_optimizer = False
        self.is_converged = False
        self.budget = 10**8
        self.SUBMITTED_PARAMS = []
        self.RECEIVED_VALUES = []
        self._params = np.array([[]])
        self._values = np.array([[]])

    def _set_param_space(self, param_space):
        self.param_space = param_space

        from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        from smac.configspace import ConfigurationSpace

        # from smac.optimizer.objective    import average_cost
        from smac.runhistory.runhistory import RunHistory
        from smac.scenario.scenario import Scenario
        from smac.stats.stats import Stats
        from smac.utils.io.traj_logging import TrajLogger

        self.cs = ConfigurationSpace()
        for param in param_space:
            if param.type == "continuous":
                var = UniformFloatHyperparameter(
                    param.name, param.low, param.high
                )
                self.cs.add_hyperparameter(var)
        self.runhistory = RunHistory(
            overwrite_existing_runs=True
        )  # (aggregate_func=average_cost)
        self.scenario = Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": self.budget,
                "cs": self.cs,
            }
        )
        self.stats = Stats(self.scenario)
        self.traj_logger = TrajLogger(output_dir=__scratch__, stats=self.stats)

    # def _set_observations(self, observations):
    def _tell(self, observations):
        from smac.tae.execute_ta_run import StatusType

        self._params = observations.get_params(as_array=True)
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )
        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])
            self.runhistory.add(
                self.smac_param,
                self._values[-1],
                time=1.0,
                status=StatusType.SUCCESS,
            )

    def _priv_evaluator(self, params):
        return None

    def create_optimizer(self):
        from smac.epm.rf_with_instances import RandomForestWithInstances
        from smac.initial_design.default_configuration_design import (
            DefaultConfiguration,
        )
        from smac.intensification.intensification import Intensifier
        from smac.optimizer.acquisition import EI
        from smac.optimizer.ei_optimization import (
            InterleavedLocalAndRandomSearch,
        )
        from smac.optimizer.objective import average_cost
        from smac.optimizer.smbo import SMBO
        from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
        from smac.tae.execute_ta_run import StatusType
        from smac.utils.constants import MAXINT
        from smac.utils.util_funcs import get_types

        TAE_RUNNER = self._priv_evaluator

        runhistory2epm = RunHistory2EPM4Cost(
            scenario=self.scenario,
            num_params=len(self.param_space),
            success_states=[StatusType.SUCCESS, StatusType.CRASHED],
            impute_censored_data=False,
            impute_state=None,
        )

        intensifier = Intensifier(
            tae_runner=TAE_RUNNER,
            stats=self.stats,
            traj_logger=self.traj_logger,
            rng=self.rng,
            instances=self.scenario.train_insts,
            cutoff=self.scenario.cutoff,
            deterministic=self.scenario.deterministic,
            run_obj_time=self.scenario.run_obj == "runtime",
            always_race_against=self.scenario.cs.get_default_configuration()
            if self.scenario.always_race_default
            else None,
            instance_specifics=self.scenario.instance_specific,
            minR=self.scenario.minR,
            maxR=self.scenario.maxR,
        )

        types, bounds = get_types(
            self.scenario.cs, self.scenario.feature_array
        )
        model = RandomForestWithInstances(
            types=types,
            bounds=bounds,
            seed=self.rng.randint(MAXINT),
            instance_features=self.scenario.feature_array,
            pca_components=self.scenario.PCA_DIM,
        )
        acq_func = EI(model=model)

        smbo_args = {
            "scenario": self.scenario,
            "stats": self.stats,
            "initial_design": DefaultConfiguration(
                tae_runner=TAE_RUNNER,
                scenario=self.scenario,
                stats=self.stats,
                traj_logger=self.traj_logger,
                rng=self.rng,
            ),
            "runhistory": self.runhistory,
            "runhistory2epm": runhistory2epm,
            "intensifier": intensifier,
            #'aggregate_func': average_cost,
            "num_run": self.seed,
            "model": model,
            "acq_optimizer": InterleavedLocalAndRandomSearch(
                acq_func,
                self.scenario.cs,
                np.random.RandomState(seed=self.rng.randint(MAXINT)),
            ),
            "acquisition_func": acq_func,
            "rng": self.rng,
            "restore_incumbent": None,
        }

        self.smbo = SMBO(**smbo_args)

    # def _generate(self):
    def _ask(self):
        if self.has_optimizer is False:
            self.create_optimizer()
            self.has_optimizer = True

        smac_param_list = self.smbo.choose_next(self._params, self._values)
        for smac_param in smac_param_list:
            self.smac_param = smac_param
            break

        param_dict = {key: self.smac_param[key] for key in self.smac_param}
        return ParameterVector().from_dict(param_dict)


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
