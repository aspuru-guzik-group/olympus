#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus.surfaces import AbstractSurface


class CatAckley(AbstractSurface):

    """The Ackley surface is inspired by the Ackley path function for
    continuous spaces. It features a narrow funnel around the global minimum,
    which is degenerate if the number of options along one (or more)dimensions
    is even and well-defined if the number of options for all dimensions is odd
    Ackley is to be evaluated on the hypercube
    x_i in [-32.768, 32.768] for i = 1, ..., d
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

    def ackley(self, vector, a=20.0, b=0.2, c=2.0 * np.pi):
        result = (
            -a * np.exp(-b * np.sqrt(np.sum(vector**2) / self.param_dim))
            - np.exp(np.sum(np.cos(c * vector)))
            + a
            + np.exp(1)
        )
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
            vector[index] = (
                65.536 * (element / float(self.num_opts - 1)) - 32.768
            )
        return self.ackley(vector)

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


if __name__ == "__main__":

    from olympus.surfaces import Surface

    surf = Surface(kind="CatAckley", param_dim=2, num_opts=21)

    best_results = surf.get_best(goal="minimize")

    print("best results :", best_results)

    print("-" * 50)

    best_results = surf.get_best(goal="maximize")

    print("best results :", best_results)
