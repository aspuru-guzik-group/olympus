#!/usr/bin/env python

# ===============================================================================

import numpy as np

from olympus import Logger
from olympus.objects import (
    ObjectParameter,
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)

# ===============================================================================


class ParameterSpace:
    """generic parameter space

    This class is intended to store all information about a parameter
    space subject to optimization, provide the interface to define a
    parameter space and perform basic operations on the parameter
    space (constraints, boundaries, etc.)
    """

    def __init__(self):
        self.parameters = []

    def __getitem__(self, index):
        return self.parameters[index]

    def __iter__(self):
        for param in self.parameters:
            yield param

    def __len__(self):
        return len(self.parameters)

    def __str__(self):
        return "\n".join([str(param) for param in self])

    @property
    def param_names(self):
        return [param.name for param in self]

    @property
    def param_types(self):
        return [param.name for param in self]

    @property
    def param_volumes(self):
        return [param.volume for param in self]

    @property
    def param_bounds(self):
        """parameter bounds only apply for discrete and continuous
        valued parameters
        """
        if not np.all(
            [param.type in ["discrete", "continuous"] for param in self]
        ):
            message = "Parameter space contains at least one categorical variable. Returning bounds for continuous and discrete parameters"
            Logger.log(message, "WARNING")
        elif np.all([param.type == "categorical" for param in self]):
            message = (
                "Fully categorical parameter space. No parameter bounds needed"
            )
            Logger.log(message, "WARNING")

        return [
            (param.low, param.high)
            for param in self
            if param.type in ["discrete", "continuous"]
        ]

    @property
    def param_lowers(self):
        """parameter bounds only apply for discrete and continuous
        valued parameters
        """
        if not np.all(
            [param.type in ["discrete", "continuous"] for param in self]
        ):
            message = "Parameter space contains at least one categorical variable. Returning bounds for continuous and discrete parameters"
            Logger.log(message, "WARNING")
        elif np.all([param.type == "categorical" for param in self]):
            message = (
                "Fully categorical parameter space. No parameter bounds needed"
            )
            Logger.log(message, "WARNING")

        return [
            param.low
            for param in self
            if param.type in ["discrete", "continuous"]
        ]

    @property
    def param_uppers(self):
        """parameter bounds only apply for discrete and continuous
        valued parameters
        """
        if not np.all(
            [param.type in ["discrete", "continuous"] for param in self]
        ):
            message = "Parameter space contains at least one categorical variable. Returning bounds for continuous and discrete parameters"
            Logger.log(message, "WARNING")
        elif np.all([param.type == "categorical" for param in self]):
            message = (
                "Fully categorical parameter space. No parameter bounds needed"
            )
            Logger.log(message, "WARNING")

        return np.array(
            [
                param.high
                for param in self
                if param.type in ["discrete", "continuous"]
            ]
        )

    @property
    def param_options(self):
        """parameter options only apply for categorical parameters"""
        if not np.all([param.type == "categorical" for param in self]):
            message = "Parameter space contains at least one discrete or continuous parameter. Returning options for continuous and discrete parameters"
            Logger.log(message, "WARNING")
        elif np.all(
            [param.type in ["discrete", "continuous"] for param in self]
        ):
            message = "Fully discrete and continuous parameter space. No parameter options needed"
            Logger.log(message, "WARNING")
        return np.array(
            [param.options for param in self if self.type == "categorcial"]
        )

    def _add_param(self, param):
        # check if we already have that param
        if param.name in self.param_names:
            message = f"""Parameter "{param.name}" is already defined"""
            Logger.log(message, "ERROR")
        else:
            self.parameters.append(param)

    def add(self, param):
        """

        Args:
            param:

        Returns:

        """
        if isinstance(param, ObjectParameter):
            self._add_param(param)
        elif isinstance(param, list):
            for _param in param:
                self._add_param(_param)
        else:
            Logger.log("Please provide a valid parameter", "ERROR")

    def get_param(self, name):
        """

        Args:
            name:

        Returns:

        """
        for param in self.parameters:
            if param["name"] == name:
                return param
        message = f"Could not find Parameter with name {name} in {str(self)}"
        Logger.log(message, "WARNING")
        return None

    def item(self):
        """

        Returns:

        """
        return [param.to_dict() for param in self]

    def validate(self, param_vector):
        """
        Args:
            param_vector: Olympus parameter vector
        Returns:
        """

        is_valid = True
        for param_ix, param_val in enumerate(param_vector):
            param = self[param_ix]
            if param.type in ["continuous", "discrete"]:
                is_valid = is_valid and param.low <= param_val
                is_valid = is_valid and param.high >= param_val
            elif param.type == "categorical":
                is_valid = param_val in param.options
        if is_valid is False:
            message = (
                f"Not all parameters of {param_vector} are within bounds!"
            )
            Logger.log(message, "WARNING")

        return is_valid


# DEBUGGING
if __name__ == "__main__":

    param_space = ParameterSpace()

    param_space.add(
        ParameterContinuous(
            name="param_0",
            low=0.0,
            high=1.0,
        )
    )

    param_space.add(
        ParameterDiscrete(
            name="param_1",
            low=0.0,
            high=1.0,
            stride=0.1,
        )
    )

    param_space.add(
        ParameterCategorical(
            name="param_2",
            options=[f"x_{i}" for i in range(5)],
            descriptors=[None for _ in range(5)],
        )
    )

    print(param_space.get_param("param_0"))
