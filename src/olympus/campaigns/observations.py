#!/usr/bin/env python

import numpy as np

from olympus import Object
from olympus.objects import ParameterVector


# ==========
# Main Class
# ==========
class Observations:
    def __init__(self):
        self.params = None  # expected to be a list of numpy arrays
        self.values = None  # expected to be a list of numpy arrays

        self.param_space = None
        self.value_space = None

        self._params_as_vectors = None
        self._values_as_vectors = None

    def _params_append(self, param):
        if self.params is None:
            self.params = np.array([param])
        else:
            self.params = np.append(self.params, [param], axis=0)

    def _values_append(self, value):
        if self.values is None:
            self.values = np.array([value])
        else:
            self.values = np.append(self.values, [value], axis=0)

    def set_param_space(self, param_space):
        self.param_space = param_space

    def set_value_space(self, value_space):
        self.value_space = value_space

    def _guess_param_space(self, param):
        self.param_space = param.param_space

    def _guess_value_space(self, value):
        self.value_space = value.param_space

    def add_observation(self, param, value):
        if isinstance(param, list):
            for param_ in param:
                if isinstance(param_, ParameterVector):
                    self._params_append(param_.to_array())
                    if self.param_space is None:
                        self._guess_param_space(param_)
                else:
                    self._params_append(param)
        else:
            if isinstance(param, ParameterVector):
                self._params_append(param.to_array())
                if self.param_space is None:
                    self._guess_param_space(param)
            else:
                self._params_append(param)

        if isinstance(value, list):
            for value_ in value:
                if isinstance(value_, ParameterVector):
                    self._values_append(value_.to_array())
                    if self.value_space is None:
                        self._guess_value_space(value_)
                else:
                    self._values_append(value)
        else:
            if isinstance(value, ParameterVector):
                self._values_append(value.to_array())
                if self.value_space is None:
                    self._guess_value_space(value)
            else:
                self._values_append(value)

        # reset caches
        self._params_as_vectors = None
        self._values_as_vectors = None

    def _construct_param_vectors(self):
        if self.params is None:
            self._params_as_vectors = []
        else:
            self._params_as_vectors = np.array(
                [
                    ParameterVector(param, param_space=self.param_space)
                    for param in self.params
                ]
            )

    def _construct_value_vectors(self):
        if self.values is None:
            self._values_as_vectors = []
        else:
            self._values_as_vectors = np.array(
                [
                    ParameterVector(value, param_space=self.value_space)
                    for value in self.values
                ]
            )

    def get_params(self, as_array=True):
        if self.params is None:
            return []

        if as_array is True:
            return np.array(self.params)
        elif as_array is False:
            if self._params_as_vectors is None:
                self._construct_param_vectors()
            return np.array([param.to_dict() for param in self._params_as_vectors])
        else:
            NotImplementedError

    def get_values(self, as_array=True, opposite=False):
        if self.values is None:
            return []

        if as_array is True:
            if opposite is False:
                return np.array(self.values)
            elif opposite is True:
                return -np.array(self.values)
        elif as_array is False:
            if self._values_as_vectors is None:
                self._construct_value_vectors()
            if opposite is False:
                return [value.to_dict() for value in self._values_as_vectors]
            elif opposite is True:
                values_dict = [value.to_dict() for value in self._values_as_vectors]
                for value_dict in values_dict:
                    for key, value in value_dict.items():
                        value_dict[key] = -1 * value
                return np.array(values_dict)

    def __OLD__init__(self):
        """ Class that contains all parameters and measurements collected from planners and emulators, i.e. `observed`
        parameters and measurements.
        """
        self.params = []  # expected to be a list of ParamVector objects
        self.values = []  # expected to be a list of ParamVector objects

    def __OLD_add_observation(self, param, value):
        """

        Args:
            param:
            value:

        Returns:

        """
        if isinstance(param, list):
            self.params.extend(param)
        elif isinstance(param, ParameterVector):
            self.params.append(param)

        if isinstance(value, list):
            self.values.extend(value)
        elif isinstance(param, ParameterVector):
            self.values.append(value)
        else:
            raise NotImplementedError

    def __OLD_get_params(self, as_array=True):
        """

        Args:
            as_array:

        Returns:

        """
        if as_array is True:
            return np.array([param.to_array() for param in self.params])
        elif as_array is False:
            return np.array([param.to_dict() for param in self.params])

    def __OLD_get_values(self, as_array=True, opposite=False):
        """

        Args:
            as_array (bool): Whether to return the values as an array.
            opposite (bool): Whether to return the opposite of the values. Default is False.

        Returns:

        """

        # NOTE: we use flipped values for the measurements to allow maximization in the planners!

        # return as array
        if as_array is True:
            # return normal values
            if opposite is False:
                return np.array([value.to_array() for value in self.values])
            # return flipped values
            elif opposite is True:
                return np.array([-1 * value.to_array() for value in self.values])
        # return as dict
        elif as_array is False:
            # return normal values
            if opposite is False:
                return np.array([value.to_dict() for value in self.values])
            # return flipped values
            elif opposite is True:
                values_dict = [value.to_dict() for value in self.values]
                for value_dict in values_dict:
                    for key, value in value_dict.items():
                        value_dict[key] = -1 * value
                return np.array(values_dict)
