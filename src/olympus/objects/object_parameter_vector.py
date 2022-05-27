#!/usr/bin/env python

import numpy as np

from olympus import Logger
from olympus.objects import Object, Parameter

from . import Object, Parameter

# ===============================================================================


class ObjectParameterVector(Object):
    def __init__(self, array=None, dict=None, param_space=None):
        """
        Args:
            array (array):
            dict (dict): dictionary with parameter names and values.
            param_space (ParameterSpace): ParameterSpace instance. This is typically defined as part of a Dataset and
                is also inherited by Emulator. If a `param_space` is defined, `info_dict` will be checked to ensure the
                provided keys match those in `param_space`, otherwise `info_dict` is accepted as is. Default is None.
        """
        from olympus.campaigns import ParameterSpace

        self.param_space = ParameterSpace()
        Object.__init__(self)

        if array is not None and param_space is not None:
            _ = self.from_array(array=array, param_space=param_space)
        elif dict is not None:
            _ = self.from_dict(info_dict=dict, param_space=param_space)

    def __repr__(self):
        string = "ParamVector("
        for param in self.param_space:
            string += "{} = {}, ".format(param.name, getattr(self, param.name))
        if len(string) > 12:
            return string[:-2] + ")"
        else:
            return string + ")"

    def __str__(self):
        return self.__repr__()

    # ***************************************************************************

    def __abs__(self):
        return np.abs(self.to_array())

    def __gt__(self, value):
        array = self.to_array()
        if array.shape[0] > 1:
            Logger.log(f"undefined operation: {self} > {value}", "ERROR")
            raise SystemExit
        return array[0] > value

    def __lt__(self, value):
        array = self.to_array()
        if array.shape[0] > 1:
            Logger.log(f"undefined operation: {self} < {value}", "ERROR")
            raise SystemExit
        return array[0] < value

    def __sub__(self, vector):
        array = self.to_array()
        if array.shape[0] > 1:
            Logger.log(f"undefined operation: {self} < {value}", "ERROR")
            raise SystemExit
        return ObjectParameterVector().from_array(
            array - vector, self.param_space
        )

    # ***************************************************************************

    def from_array(self, array, param_space):
        self.param_space = param_space
        for value, param in zip(array, param_space):
            self.add(param.name, value)
        return self

    def to_array(self):
        array = np.array(
            [
                getattr(self, param_name)
                for param_name in self.param_space.param_names
            ]
        )
        return array

    def from_list(self, list_, param_space):
        self.param_space = param_space
        for value, param in zip(list_, param_space):
            self.add(param.name, value)
        return self

    def to_list(self):
        list_ = [
            getattr(self, param_name)
            for param_name in self.param_space.param_names
        ]
        return list_

    def from_dict(self, info_dict, param_space=None):
        """Creates a ParamVector representation of a given dictionary.

        Args:
            info_dict (dict): dictionary with parameter names and values.
            param_space (ParameterSpace): ParameterSpace instance. This is
                                typically defined as part of a Dataset and is also inherited by
                                Emulator. If a `param_space` is defined, `info_dict` will be
                                checked to ensure the provided keys match those in `param_space`,
                                otherwise `info_dict` is accepted as is. Default is None.
        """

        if param_space is None:
            for key, value in info_dict.items():
                # define parameter of parameter space
                self.param_space.add(Parameter(name=key))
                # add specific value for the parameter
                self.add(key, value)

        elif param_space is not None:
            if set(param_space.param_names) != set(list(info_dict.keys())):
                message = "The dictionary keys provided do not match those in the parameter space"
                Logger.log(message, "ERROR")
            self.param_space = param_space
            for param_name in param_space.param_names:
                # add specific value for the parameter
                self.add(param_name, info_dict[param_name])
        return self
