#!/usr/bin/env python


import numpy as np

from olympus import Logger
from olympus.scalarizers import AbstractScalarizer


class Parego(AbstractScalarizer):
    """ParEGO acheivement scalarizing function"""

    def __init__(self, value_space, goals, rho=0.05):
        AbstractScalarizer.__init__(**locals())

        self.validate_asf_params()

    def scalarize(self, objectives):

        theta = np.random.random_sample(len(self.value_space))
        sum_theta = np.sum(theta)
        theta = theta / sum_theta

        signs = [1 if self.goals[idx]=='min' else -1 for idx in range(len(self.value_space))]
        objectives = objectives*signs
        norm_objectives = self.normalize(objectives)

        theta_f = theta * norm_objectives
        max_k = np.amax(theta_f, axis=1)
        rho_sum_theta_f = self.rho * np.sum(theta_f, axis=1)
        merit = max_k + rho_sum_theta_f

        if merit.shape[0] > 1:
            # normalize the merit (best value is 0., worst is 1.)
            merit = self.normalize(merit.reshape(-1, 1))
            return np.squeeze(merit, axis=1)
        else:
            return merit

    @staticmethod
    def normalize(vector):
        min_ = np.amin(vector)
        max_ = np.amax(vector)
        ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
        if not ixs.size == 0:
            max_[ixs] = np.ones_like(ixs)
            min_[ixs] = np.zeros_like(ixs)
        return (vector - min_) / (max_ - min_)

    def validate_asf_params(self):
        pass

    @staticmethod
    def check_kwargs(kwargs):
        """quick and dirty check to see if the proper arguments are provided
        for the scalarizer
        """
        req_args = ["goals"]
        provided_args = list(kwargs.keys())
        missing_args = list(set(req_args).difference(provided_args))
        if not missing_args == []:
            message = (
                f'Missing required ParEGO arguments {", ".join(missing_args)}'
            )
            Logger.log(message, "FATAL")


if __name__ == "__main__":

    from olympus import Surface

    surf = Surface(kind="MultFonseca")

    scalarizer = ParEGO(surf.value_space, goals=["min", "min"])

    objectives = np.array([[0.1, 0.4], [0.7, 0.9], [0.04, 0.08]])

    scalarizer.scalarize(objectives)
