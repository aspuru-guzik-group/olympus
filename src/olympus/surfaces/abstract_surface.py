#!/usr/bin/env python

import numpy as np
from olympus import Logger
from abc import abstractmethod
from olympus.objects import Object, abstract_attribute, ABCMeta, ParameterVector
from olympus.campaigns.param_space import ParameterSpace, ParameterContinuous


class AbstractSurface(Object, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._create_param_space(self.param_dim)
        self._create_value_space()

    @abstract_attribute
    def param_dim(self):
        pass

    @abstract_attribute
    def minima(self):
        pass

    @abstract_attribute
    def maxima(self):
        pass

    @abstractmethod
    def _run(self, features):
        pass

    @property
    def kind(self):
        return type(self).__name__

    def _create_param_space(self, param_dim):
        """
        Creates as continuous k-dimensional unit hypercube parameter space, where
        k = `param_dim`. In this space, the surface will live.
        """
        self.param_space = ParameterSpace()
        for i in range(param_dim):
            param = ParameterContinuous(name=f"param_{i}", low=0.0, high=1.0)
            self.param_space.add(param)

    def _create_value_space(self, value_dim=1):
        self.value_space = ParameterSpace()
        for _ in range(value_dim):
            value = ParameterContinuous(name=f"value_{_}", low=0.0, high=1.0)
            self.value_space.add(value)

    def run(self, params, return_paramvector=False):
        """Evaluate the surface at the chosen location.

        Args:
            params (array): Set of input parameters for which to return the function value.
            return_paramvector (bool): Whether to return a ``ParameterVector`` object instead of a list of lists.
                Default is False.

        Returns:
            values (ParameterVector): function values evaluated at the chosen locations.
        """
        if isinstance(params, float) or isinstance(params, int):
            params = np.array([params])
        elif type(params) == list:
            params = np.array(params)
        if len(params.shape) == 1:
            params = np.expand_dims(params, axis=0)

        # TODO: these validations could be moved to ParameterSpace class
        # validate params
        if params.shape[1] != len(self.param_space):
            message = (f'Dimensions of provided params ({params.shape[1]}) does not match expected '
                       f'dimension ({len(self.param_space)})')
            Logger.log(message, 'ERROR')

        # this raises warnings for out-of-bounds parameters
        for param_set in params:
            self.param_space.validate(param_set)

        # get values from the surface class
        y_preds = [[self._run(param_set)] for param_set in params]  # 2d array

        # if we are not asking for a ParamVector, we can just return y_preds
        if return_paramvector is False:
            return y_preds

        # return a ParameterVector
        # NOTE: while we do not allow batches or multiple objectives yet, this code is supposed to be able to support
        #  those
        y_pred_objects = []  # list of ParamVectors with all samples and objectives
        # iterate over all samples (if we returned a batch of predictions)
        for y_pred in y_preds:
            y_pred_object = ParameterVector()
            # iterate over all objectives/targets
            for target_name, y in zip(['target_0'], y_pred):
                y_pred_object.from_dict({target_name: y})
            # append object to list
            y_pred_objects.append(y_pred_object)

        return y_pred_objects
