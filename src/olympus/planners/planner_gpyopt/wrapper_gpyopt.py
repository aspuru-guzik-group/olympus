#!/usr/bin/env python

import numpy as np

from olympus.objects import ParameterVector
from olympus.planners import AbstractPlanner


class Gpyopt(AbstractPlanner):

    PARAM_TYPES = ["continuous", "discrete", "categorical"]

    def __init__(
        self,
        goal="minimize",
        batch_size=1,
        exact_eval=True,
        model_type="GP",  #'GP_MCMC',
        acquisition_type="EI",  #'EI_MCMC',
        initial_design_num_data=1,
    ):
        """
        Gaussian Process optimization as implemented in GPyOpt.
        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            batch_size (int): size of the batch in which the objective is evaluated.
            exact_eval (bool): whether the outputs are exact.
            model_type (str): type of model to use as surrogate. 'GP': standard Gaussian process. 'GP_MCMC': Gaussian
                process with prior in the hyper-parameters. 'sparseGP': sparse Gaussian process. 'warperdGP': warped
                Gaussian process. 'InputWarpedGP': input warped Gaussian process. 'RF': random forest (scikit-learn).
            acquisition_type (str): type of acquisition function to use. 'EI': expected improvement. 'EI_MCMC': integrated
                expected improvement (requires GP_MCMC model). 'MPI': maximum probability of improvement. 'MPI_MCMC':
                maximum probability of improvement (requires GP_MCMC model). 'LCB': GP-Lower confidence bound. 'LCB_MCMC':
                integrated GP-Lower confidence bound (requires GP_MCMC model).
        """
        AbstractPlanner.__init__(**locals())
        self.has_categorical = False

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == "continuous":
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "domain": (param.low, param.high),
                    "dimensionality": 1,
                }
            elif param.type == "categorical":
                # generate an array of integers corresponding to each categorical variable option
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "domain": np.arange(len(param.options)),
                    "dimensionality": 1,
                    "original": param.options,
                }
                self.has_categorical = True
            elif param.type == "discrete":
                # make a map between discrete options and an array of integers that can be referenced later
                # TODO: this is a bit of a hack, since it will only work if upper and lower bounds are actually
                # included in the options
                num_options = (param.high - param.low / param.stride) + 1
                options = np.linspace(param.low, param.high, num_options)
                print(num_options, options)
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "domain": options,
                    "dimensionality": 1,
                }
            else:
                raise NotImplementedError(
                    f"Parameter type {param.type} for {param.name} is not implemnted"
                )
            self._param_space.append(param_dict)

    def _tell(self, observations):
        self.observations = observations
        self._params = self.observations.get_params(as_array=True)
        self._values = np.array(
            self.observations.get_values(
                as_array=True, opposite=self.flip_measurements
            )
        )

        # need to inflate the shape, I guess (at least 2d)
        if len(self._values.shape) == 1:
            self._values = self._values.reshape(-1, 1)
        # apply transformation for categorical variables "olympus --> Gpyopt"
        # if self.has_categorical:
        self._gpyopt_params = []
        for obs in self._params:
            gpyopt_obs = []
            for param_ix, param in enumerate(self._param_space):
                if param["type"] == "categorical":
                    ix = param["original"].index(obs[param_ix])
                    gpyopt_obs.append(int(ix))
                else:
                    gpyopt_obs.append(obs[param_ix])
            self._gpyopt_params.append(gpyopt_obs)
        self._gpyopt_params = np.array(self._gpyopt_params)

    def _get_bo_instance(self):
        from GPyOpt.methods import BayesianOptimization

        bo = BayesianOptimization(
            f=None,
            domain=self._param_space,
            batch_size=self.batch_size,
            exact_eval=self.exact_eval,
            model_type=self.model_type,
            acquisition_type=self.acquisition_type,
            X=self._gpyopt_params,
            Y=self._values,
            de_duplication=True,
        )
        return bo

    def _ask(self):
        params_return = []
        if 0 <= len(self._params) < self.initial_design_num_data:
            # do the initial design randomly
            from olympus.planners import Planner

            init_design_planner = Planner(kind="RandomSearch", goal=self.goal)
            init_design_planner.set_param_space(self.param_space)

            for _ in range(self.batch_size):
                init_design_planner.tell(self.observations)
                param = init_design_planner.ask(return_as=None)
                params_return.append(param)

        else:
            bo = self._get_bo_instance()
            array = bo.suggest_next_locations()
            for batch_ix in range(self.batch_size):
                # transform back to Olympus categotical parameters
                array_olymp = []
                for param_ix, suggestion in enumerate(array[batch_ix]):
                    if self._param_space[param_ix]["type"] == "categorical":
                        array_olymp.append(
                            self._param_space[param_ix]["original"][
                                int(suggestion)
                            ]
                        )
                    else:
                        array_olymp.append(suggestion)
                params_return.append(
                    ParameterVector().from_array(array_olymp, self.param_space)
                )

        # TODO: this is a hack returning the first list element
        return params_return[0]


# DEBUG:
if __name__ == "__main__":

    from olympus import Campaign
    from olympus.datasets import Dataset

    d = Dataset(kind="perovskites")

    planner = Gpyopt(goal="minimize")
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
