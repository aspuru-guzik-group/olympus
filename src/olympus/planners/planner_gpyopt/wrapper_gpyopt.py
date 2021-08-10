#!/usr/bin/env python

import numpy as np

from olympus.planners import AbstractPlanner
from olympus.objects import ParameterVector

from olympus import Planner


class Gpyopt(AbstractPlanner):

    def __init__(
        self,
        goal='minimize',
        batch_size=1,
        exact_eval=True,
        model_type='GP',
        acquisition_type='EI',
        num_init_design=2,
        init_design_type='RandomSearch',
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
            num_init_design (int): the number of points to be measurement using the initial design strategy
            init_design_type (str): Olympus planner name for the intial design strategy - supports only RandomSearch
                for now
        """
        AbstractPlanner.__init__(**locals())

        if self.init_design_type == 'RandomSearch':
            self.init_design_planner = Planner(
                kind=self.init_design_type, goal=self.goal,
            )
        else:
            raise NotImplementedError


    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
            self._param_space.append(param_dict)

        self.init_design_planner.set_param_space(self.param_space)

    def _tell(self, observations):
        self._params = observations.get_params(as_array=True)
        self._values = np.array(observations.get_values(as_array=True, opposite=self.flip_measurements))
        # need to inflate the shape, I guess (at least 2d)
        if len(self._values.shape) == 1:
            self._values = self._values.reshape(-1, 1)

    def _get_bo_instance(self):
        from GPyOpt.methods import BayesianOptimization
        bo = BayesianOptimization(
                f                = None,
                domain           = self._param_space,
                batch_size       = self.batch_size,
                exact_eval       = self.exact_eval,
                model_type       = self.model_type,
                acquisition_type = self.acquisition_type,
                X = self._params,
                Y = self._values,
                de_duplication = True,
            )
        return bo

    def _ask(self):
        if self._params is None or len(self._params) < self.num_init_design:
            # initial design suggestions
            param = self.init_design_planner.ask(return_as=None)
            return_param = param
        else:
            # bo suggestions
            bo    = self._get_bo_instance()
            array = bo.suggest_next_locations()[0]
            return_param = ParameterVector().from_array(array, self.param_space)

        return return_param
