#!/usr/bin/env python

from copy import deepcopy
from functools import reduce

import numpy as np

from olympus import Logger
from olympus.datasets.dataset import Dataset
from olympus.objects.abstract_object import Object


class DataTransformer(Object):
    def __init__(self, transformations="identity"):
        """Applies transformations to the columns of a 2d array.

        Args:
            transformations (str, list): Desired transformation. If a list is provided, the transformations will
                be performed one after the other one as provided in the list. Note that the `periodic` transformation
                is currently supported only as the first transformation, and only as a forward transform.
                Transformation options are:
                    `identity`: identity transform.
                    `normalize`: features are scaled to be between 0 and 1.
                    `standardize`: features are centered at 0 with standard deviation of 1.
                    `mean`: features are divided by their mean.
                    `sqrt_mean`: the square root of the features is taken after having been divided by their mean.
                    `periodic`: takes periodic variables in a Dataset and splits them into a sine and cosine representation.
        """
        Object.__init__(self)

        # check the input arguments
        if isinstance(transformations, str):
            transformations = [transformations]
        self._validate_args(transformations)

        # select transformations
        # note that we select the back transformations in reversed order
        self.transformations = transformations
        self._transforms = [
            getattr(self, f"_forward_{trans}")
            for trans in self.transformations
        ]
        self._back_transforms = [
            getattr(self, f"_backward_{trans}")
            for trans in reversed(self.transformations)
        ]

        # the stats we will need
        self._mean = None
        self._stddev = None
        self._min = None
        self._max = None
        self._periodic_info = (
            {}
        )  # each key is a column index, each value is low/high

        # other info
        self.trained = False
        self._dims = None

    # ====================
    # Methods for the user
    # ====================
    def train(self, data):
        """Computes the statistics (e.g. mean and standard deviation) needed for the chosen transformation from the
        provided dataset. With the exception of the 'identity' transform, the DataTransformer always needs to be
        trained before the `transform` and `back_transform` methods can be used.

        Args:
            data (array, Dataset): the data used to compute the statistics needed for the transformation. This can
                be a 2d numpy array, or a Dataset object.
        """

        self._dims = None  # reset _dims if we retrain the DataTransformer

        # for splitting periodic variables we need a dataset, so that we can check which variables are periodic
        # and that are their lower/upper bounds
        if "periodic" in self.transformations:
            if isinstance(data, Dataset) is False:
                message = "in order to transform periodic variables you need to provide a Dataset object as the data argument"
                Logger.log(message, "ERROR")

            # remember the input dimensions
            self._dims = np.shape(data.data)[1]

            # extract the info about periodic variables
            self._parse_dataset_for_periodic(data)

            # Now swap dataset for data after periodic transform. This is done just in case the periodic transform is
            # composed with other transformations that will then require operating on a higher dimensional array
            # the means, stddev etc. statistics will need to have matching dimensions
            data = self._forward_periodic(data.data.to_numpy())
        else:
            # allow passing a dataset
            if isinstance(data, Dataset) is True:
                data = data.data.to_numpy()
            self._validate_data(data)
            # remember the input dimensions
            self._dims = np.shape(data)[1]

        # ------------------------
        # Get stats about the data
        # ------------------------
        data = np.array(data)

        self._mean = np.mean(data, axis=0)
        self._stddev = np.std(data, axis=0)
        self._min = np.amin(data, axis=0)
        self._max = np.amax(data, axis=0)

        # replace instances with 0. stddev with 1.
        self._stable_stddev = np.where(self._stddev == 0.0, 1.0, self._stddev)

        # replace instances where the denominator of minmax scaling is 0.
        self._stable_min = deepcopy(self._min)
        self._stable_max = deepcopy(self._max)
        ixs = np.where(np.abs(self._max - self._min) < 1e-10)[0]
        if not ixs.size == 0:
            self._stable_max[ixs] = np.ones_like(ixs)
            self._stable_min[ixs] = np.zeros_like(ixs)

        self.trained = True

    def transform(self, data):
        """Performs the transform on the provided data.

        Args:
            data (ndarray): a 2-dimensional numpy array where each row is a sample and each column a feature.
                `numpy.shape(data)` should return (num_samples, num_features)'

        Returns
            data_transformed (ndarray): the transformed data.

        """
        if isinstance(data, Dataset) is True:
            data = data.data.to_numpy()
        else:
            data = np.array(data)
        self._validate_data(data)

        if self.trained is False and self.transformation != "identity":
            raise ValueError(
                f"DataTransformer needs to be trained before the transformation {self.transformation} "
                f"can be applied."
            )

        pipeline = self._compose_transformations(self._transforms)
        return pipeline(data)

    def back_transform(self, data):
        """Performs the inverse transform on the provided data.

        Args:
           data (ndarray): a 2-dimensional numpy array where each row is a sample and each column a feature.
                `numpy.shape(data)` should return (num_samples, num_features)'

        Returns
            data_transformed (ndarray): the transformed data.

        """
        if isinstance(data, Dataset) is True:
            data = data.data.to_numpy()
        else:
            data = np.array(data)
        self._validate_data(data)

        if self.trained is False and self.transformation != "identity":
            raise ValueError(
                f"DataTransformer needs to be trained before the transformation {self.transformation} "
                f"can be applied."
            )

        pipeline = self._compose_transformations(self._back_transforms)
        return pipeline(data)

    # ===============
    # Private Methods
    # ===============
    def _forward_standardize(self, data):
        return (data - self._mean) / self._stable_stddev

    def _backward_standardize(self, data):
        return data * self._stable_stddev + self._mean

    def _forward_normalize(self, data):
        return (data - self._stable_min) / (
            self._stable_max - self._stable_min
        )

    def _backward_normalize(self, data):
        return (self._stable_max - self._stable_min) * data + self._stable_min

    def _forward_identity(self, data):
        return data

    def _backward_identity(self, data):
        return data

    def _forward_mean(self, data):
        return data / self._mean

    def _backward_mean(self, data):
        return data * self._mean

    def _forward_log_mean(self, data):
        return np.log(data / self._mean + 1e-4)

    def _backward_log_mean(self, data):
        return (np.exp(data) - 1e-4) * self._mean

    def _forward_sqrt_mean(self, data):
        return np.sqrt(data / self._mean)

    def _backward_sqrt_mean(self, data):
        return np.square(data) * self._mean

    def _forward_periodic(self, data):
        projections = []
        # iterate over columns in data
        for i, col in enumerate(data.T):
            if i in self._periodic_info:
                low = self._periodic_info[i][0]
                high = self._periodic_info[i][1]
                cosine = np.cos(2 * np.pi * (col - low) / (high - low))
                sine = np.sin(2 * np.pi * (col - low) / (high - low))
                projections.append(cosine)
                projections.append(sine)
            else:
                projections.append(col)

        return np.array(projections).T

        # TODO: if we want to allow a backward transform, we will need to store the position of sin/cos for all
        #  variables/columns transformed

    def _backward_periodic(self, data):
        raise NotImplementedError

    def _parse_dataset_for_periodic(self, dataset):
        # TODO/FIXME: this assumes the target is always the last column in a Dataset!!
        for i, param in enumerate(dataset.param_space):
            if param.is_periodic is True:
                # each key is a column index, each value is low/high
                self._periodic_info[i] = [param.low, param.high]

    def _validate_data(self, data):
        # check a 2-dimensional object is provided
        if data.ndim != 2:
            raise ValueError(
                "Incorrect data format provided. Please provide a 2-dimensional array where each row is "
                "a sample and each column a feature. `numpy.shape(data)` should return (num_samples, "
                "num_features)"
            )

        # check we have the same number of features as before
        if self._dims is not None:
            if np.shape(data)[1] != self._dims:
                if "periodic" in self.transformations:
                    raise AssertionError(
                        "periodic back transformations are not yet supported"
                    )
                else:
                    raise AssertionError("dimensionality mismatch")

    @staticmethod
    def _validate_args(transformations):
        # check validity of transformation argument
        for transformation in transformations:
            if not (
                hasattr(DataTransformer, f"_forward_{transformation}")
                and hasattr(DataTransformer, f"_backward_{transformation}")
            ):
                raise NotImplementedError(
                    f"transformation {transformation} not implemented. Please select one of the "
                    f"available transformation."
                )

        if (
            "periodic" in transformations
            and transformations.index("periodic") != 0
        ):
            message = "periodic transform is allowed only as the first transformation"
            Logger.log(message, "ERROR")

    @staticmethod
    def _compose_transformations(functions_list):
        # compose transformations as provided in the list
        return lambda x: reduce(lambda a, f: f(a), functions_list, x)


# --------------------------------
# CUBE-SIMLPEX TRANSFORMATION
# --------------------------------/


def cube_to_simpl(cubes):
    """
    converts and n-cube (used for optimization) to an n+1 simplex (used
    as features for emulator)
    """
    features = []
    for cube in cubes:
        # cube = (1 - 2 * 1e-6) * np.squeeze(np.array([c for c in cube])) + 1e-6
        cube = np.array(cube)

        simpl = np.zeros(len(cube) + 1)
        sums = np.sum(cube / (1 - cube))

        alpha = 4.0
        simpl[-1] = alpha / (alpha + sums)
        for i in range(len(simpl) - 1):
            simpl[i] = (cube[i] / (1 - cube[i])) / (alpha + sums)
        features.append(np.array(simpl))
    return np.array(features)


def simpl_to_cube(simpls):
    """
    converts from an n+1 simplex (used as features for the emulator)
    to an n-cube (used for optimization)
    """
    features = []

    for simpl in simpls:
        alpha = 4.0
        sums = (alpha * (simpl[-1] - 1)) / (simpl[-1])

        cube = np.zeros(len(simpl) - 1)

        for i in range(len(cube)):
            cube[i] = (simpl[i] * (sums - alpha)) / (
                simpl[i] * (sums - alpha) - 1
            )

        features.append(cube)

    return np.array(features)


# ------------------------------------
# CATEGORICAL PARAMS TO OHE FEATURES
# ------------------------------------


def cat_param_to_feat(param, val):
    """convert the option selection of a categorical variable to
    a machine readable feature vector
    Args:
        param (object): the categorical olympus parameter
        val (): the value of the chosen categorical option
    """
    # print(param, val)
    # get the index of the selected value amongst the options
    arg_val = param.options.index(val)
    if np.all([d == None for d in param.descriptors]):
        # no provided descriptors, resort to one-hot encoding
        # feat = np.array([arg_val])
        feat = np.zeros(len(param.options))
        feat[arg_val] += 1.0
    else:
        # we have descriptors, use them as the features
        feat = param.descriptors[arg_val]
    return feat


# -----------
# DEBUGGING
# -----------

if __name__ == "__main__":

    cubes = [[0.1, 0.5, 0.8], [0.5, 0.9, 0.04]]

    print("CUBES : ", cubes)

    simpls = cube_to_simpl(cubes)

    print("SIMPLS : ", simpls)

    cubes = simpl_to_cube(simpls)

    print("CUBES : ", cubes)
