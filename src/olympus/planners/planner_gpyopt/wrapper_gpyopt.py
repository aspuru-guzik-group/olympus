#!/usr/bin/env python

from olympus.planners import AbstractPlanner
from olympus.objects import ParameterVector


class Gpyopt(AbstractPlanner):

    def __init__(self, goal='minimize', batch_size=1, exact_eval=True, model_type='GP_MCMC', acquisition_type='EI_MCMC'):
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

    def _set_param_space(self, param_space):
        self._param_space = []
        for param in param_space:
            if param.type == 'continuous':
                param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
            self._param_space.append(param_dict)

    def _tell(self, observations):
        self._params = observations.get_params(as_array = 'True')
        self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)

    def _get_bo_instance(self):
        from GPyOpt.methods import BayesianOptimization
        bo = BayesianOptimization(
                f                = lambda x: 0, ### TODO: NEED TO CHECK IF THIS CAUSES ISSUES
                domain           = self._param_space,
                batch_size       = self.batch_size,
                exact_eval       = self.exact_eval,
                model_type       = self.model_type,
                acquisition_type = self.acquisition_type,
                X = self._params, Y = self._values,
            )
        return bo

    def _ask(self):
        if self._params is None or len(self._params) < 2:
            self._params = None
            self._values = None
            bo = self._get_bo_instance()
            generated = bo.get_evaluations()[0]
            array = generated[self.num_generated % len(generated)]
        else:
            bo    = self._get_bo_instance()
            array = bo.suggest_next_locations()[0]
        return ParameterVector().from_array(array, self.param_space)
