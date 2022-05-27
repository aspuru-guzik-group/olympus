#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus.surfaces import AbstractSurface


class CatCamel(AbstractSurface):

    """
    The Camel surface is generalized from the Camel function on continuous domains and features
    a degenerate and pseudo-disconnected global minimum
    Camel is to be evaluated on the hypercube
    x_i in [-3, 3] for i = 1, ..., d
    """

    def __init__(self, param_dim, num_opts, descriptors=None):
        """descriptors must be an iterable with the length num_opts
        For these surfaces, the same descriptors are used for each dimension
        """
        value_dim = 1
        AbstractSurface.__init__(param_type="categorical", **locals())

    @property
    def minima(self):
        return self.get_best(goal="minimize")

    @property
    def maxima(self):
        return self.get_best(goal="maximize")

    def camel(self, vector):
        result = 0.0
        # global minima
        loc_0 = np.array([-1.0, 0.0])
        loc_1 = np.array([1.0, 0.0])
        weight_0 = np.array([4.0, 1.0])
        weight_1 = np.array([4.0, 1.0])

        # local minima
        loc_2 = np.array([-1.0, 1.5])
        loc_3 = np.array([1.0, -1.5])
        loc_5 = np.array([-0.5, -1.0])
        loc_6 = np.array([0.5, 1.0])
        loss_0 = (
            np.sum(weight_0 * (vector - loc_0) ** 2)
            + 0.01
            + np.prod(vector - loc_0)
        )
        loss_1 = (
            np.sum(weight_1 * (vector - loc_1) ** 2)
            + 0.01
            + np.prod(vector - loc_1)
        )
        loss_2 = np.sum((vector - loc_2) ** 2) + 0.075
        loss_3 = np.sum((vector - loc_3) ** 2) + 0.075
        loss_5 = 3000.0 * np.exp(-np.sum((vector - loc_5) ** 2) / 0.25)
        loss_6 = 3000.0 * np.exp(-np.sum((vector - loc_6) ** 2) / 0.25)
        result = loss_0 * loss_1 * loss_2 * loss_3 + loss_5 + loss_6
        return result

    def _run(self, params):
        # map the sample onto the unit hypercube
        vector = np.zeros(self.param_dim)
        for index, element in enumerate(params):
            # make a messy check to see if the user passes integerts or
            # strings representing the categories
            # we expect either a int here, e.g. 12 or str of form e.g. 'x12'
            if isinstance(element, str):
                element = int(element[1:])
            elif isinstance(element, int):
                pass
            # TODO: add else statement here and return error
            vector[index] = 6 * (element / float(self.num_opts)) - 3
        return self.camel(vector)

    def get_best(self, goal="minimize"):
        """get the location and value of the optimum (minimum) point on the
        surfaces
        """
        domain = [f"x{i}" for i in range(self.num_opts)]
        params = []
        values = []
        for x in domain:
            for y in domain:
                values.append(self._run([x, y]))
                params.append([x, y])
        values = np.array(values)
        params = np.array(params)
        # get the indices that sort values in ascending order
        ind = np.argsort(values)
        if goal == "maximize":
            ind = ind[::-1]  # default is from smallest to largest
        sort_values = values[ind]
        sort_params = params[ind]

        best_values = [
            sort_values[i]
            for i in range(len(sort_values))
            if sort_values[i] == sort_values[0]
        ]
        best_params = [list(sort_params[i]) for i in range(len(best_values))]

        result = []
        for param, value in zip(best_params, best_values):
            result.append({"params": param, "value": value})

        return result
