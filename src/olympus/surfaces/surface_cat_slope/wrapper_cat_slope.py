#!/usr/bin/env python

from itertools import product

import numpy as np

from olympus.surfaces import AbstractSurface


class CatSlope(AbstractSurface):

    """The Slope surface is constructed such that the response linearly increases with the index of the
    option along each dimension in the reference ordering. As such, the Slope surface presents a generalization of a plane
    to categorical domains
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

    def slope(self, vector):
        seed = 0
        vector = np.array(vector)
        vector_ = []
        for index, element in enumerate(vector):
            # check to see if strings are passed
            if isinstance(element, str):
                element = int(element[1:])
            elif isinstance(element, int):
                pass
            seed += self.num_opts**index * element
            vector_.append(element)

        # print(vector)
        result = np.sum(np.array(vector_) / self.num_opts)
        return result

    def _run(self, params):
        return self.slope(params)

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

    surf = Surface(kind="CatSlope", param_dim=2, num_opts=21)

    best_results = surf.get_best(goal="minimize")

    print("best results :", best_results)

    print("-" * 50)

    best_results = surf.get_best(goal="maximize")

    print("best results :", best_results)
