#!/usr/bin/env python

from abc import abstractmethod

import numpy as np

from olympus import Logger
from olympus.campaigns.param_space import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterSpace,
)
from olympus.objects import (
    ABCMeta,
    Object,
    ParameterVector,
    abstract_attribute,
)


class AbstractSurface(Object, metaclass=ABCMeta):

    # NOTE: currently we only support surfaces with only continuous, discrete,
    # categorical parameters. Mixed param spaces could be supported in future
    # versions
    ACCEPTED_PARAM_TYPES = ["continuous", "discrete", "categorical"]
    DESC_TYPES = ["desc", None]

    def __init__(
        self, param_type="continuous", descriptors=None, *args, **kwargs
    ):
        Object.__init__(self, *args, **kwargs)
        self.param_type = param_type
        self.descriptors = descriptors
        if self._validate_param_type:
            pass
        else:
            message = f"Parameter type not understood, should be in {self.ACCEPTED_PARAM_TYPES}"
            Logger.log(message, "FATAL")

        self._create_param_space(self.param_dim)
        self._create_value_space(value_dim=self.value_dim)

        # meta information for evalautor
        self.parameter_constriants = None

    def _validate_param_type(self):
        return self.param_type in self.ACCEPTED_PARAM_TYPES

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
            if self.param_type == "continuous":
                param = ParameterContinuous(
                    name=f"param_{i}", low=0.0, high=1.0
                )
            elif self.param_type == "categorical":
                # make the options
                options_ = [f"x{i}" for i in range(self.num_opts)]
                if self.descriptors is None:
                    descriptors_ = [None for _ in range(self.num_opts)]
                elif self.descriptors == "desc":
                    descriptors_ = [[opt] for opt in range(self.num_opts)]
                param = ParameterCategorical(
                    name=f"param_{i}",
                    options=options_,
                    descriptors=descriptors_,
                )
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
        elif isinstance(params, ParameterVector):
            params = np.array([params.to_array()])
        if len(params.shape) == 1:
            params = np.expand_dims(params, axis=0)

        # TODO: these validations could be moved to ParameterSpace class
        # validate params
        if params.shape[1] != len(self.param_space):
            message = (
                f"Dimensions of provided params ({params.shape[1]}) does not match expected "
                f"dimension ({len(self.param_space)})"
            )
            Logger.log(message, "ERROR")

        # this raises warnings for out-of-bounds parameters
        for param_set in params:
            self.param_space.validate(param_set)

        # get values from the surface class
        # messy check to make sure we are always returning a 2d array
        # with shape (num_obs, num_objs)
        if len(self.value_space) == 1:
            y_preds = [
                [self._run(param_set)] for param_set in params
            ]  # 2d array
        else:
            y_preds = [
                self._run(param_set) for param_set in params
            ]  # 2d array

        # if we are not asking for a ParamVector, we can just return y_preds
        if return_paramvector is False:
            return y_preds

        # return a ParameterVector
        # NOTE: while we do not allow batches or multiple objectives yet, this code is supposed to be able to support
        #  those
        y_pred_objects = (
            []
        )  # list of ParamVectors with all samples and objectives
        # iterate over all samples (if we returned a batch of predictions)
        for y_pred in y_preds:
            y_pred_object = ParameterVector()
            # iterate over all objectives/targets
            for value_name, y in zip(
                [v.name for v in self.value_space], y_pred
            ):
                y_pred_object.from_dict({value_name: y})
            # append object to list
            y_pred_objects.append(y_pred_object)

        return y_pred_objects
