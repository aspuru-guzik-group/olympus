#!/usr/bin/env python

import numpy as np

from olympus import Logger
from olympus.scalarizers import AbstractScalarizer


class WeightedSum(AbstractScalarizer):
    """simple weighted sum acheivement scalarizing function
    weights is a 1d numpy array of
    """

    def __init__(self, value_space, weights, goals):
        AbstractScalarizer.__init__(**locals())

        self.validate_asf_params()
        # normalize the weight values such that their magnitudes
        # sum to 1
        self.norm_weights = self.softmax(self.weights)

    def scalarize(self, objectives):

        signs = [1 if self.goals[idx]=='min' else -1 for idx in range(len(self.value_space))]
        objectives = objectives*signs

        norm_objectives = self.normalize(objectives)
        merit = np.sum(norm_objectives * self.norm_weights, axis=1)
        # final normalization
        # smaller merit values are best
        if merit.shape[0] > 1:
            merit = self.normalize(merit)
        return merit

    @staticmethod
    def softmax(vector):
        vector = vector / np.amax(vector)
        return np.exp(vector) / np.sum(np.exp(vector))

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
        if not np.all(np.array(self.weights) >= 0.0):
            message = (
                "Weighted sum ASF weights must be non-negative real numbers"
            )
            Logger.log(message, "FATAL")
        if not len(self.weights) == len(self.value_space):
            message = (
                "Number of weights does not match the number of objectives"
            )
            Logger.log(message, "FATAL")

    @staticmethod
    def check_kwargs(kwargs):
        """quick and dirty check to see if the proper arguments are provided
        for the scalarizer
        """
        req_args = ["weights", "goals"]
        provided_args = list(kwargs.keys())
        missing_args = list(set(req_args).difference(provided_args))
        if not missing_args == []:
            message = f'Missing required WeightedSum arguments {", ".join(missing_args)}'
            Logger.log(message, "FATAL")
