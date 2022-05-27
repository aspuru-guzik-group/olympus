#!/usr/bin/env python


import numpy as np
from chimera import Chimera

from olympus import Logger
from olympus.scalarizers import AbstractScalarizer


class Chimera(AbstractScalarizer):

    """The Chimera achievement scalarizing function.
    Chem. Sci., 2018, 9, 7642
    """

    def __init__(self, value_space, tolerances, absolutes, goals):
        AbstractScalarizer.__init__(**locals())

        from chimera import Chimera

        self.validate_asf_params()
        self.chimera = Chimera(
            tolerances=self.tolerances,
            absolutes=self.absolutes,
            goals=self.goals,
        )

    def scalarize(self, objectives):
        """this expects a (# obs, # objs) numpy array, which is scalarized
        according to the given tolerances and goals. Returns a (# obs,)
        numpy array corresponding to the merits of each observation, 0 corresponding
        to the best value, and 1 corresponding to the worst value
        """
        assert len(objectives.shape) == 2
        return self.chimera.scalarize(objectives)

    def validate_asf_params(self):

        if not (
            len(self.tolerances) == len(self.absolutes) == len(self.goals)
        ):
            message = "Lengths of Chimera parameters do not match"
            Logger.log(message, "FATAL")
        if not len(self.tolerances) == len(self.value_space):
            message = "Number of Chimera parameters does not match the number of objectives"
            Logger.log(message, "FATAL")

    @staticmethod
    def check_kwargs(kwargs):
        """quick and dirty check to see if the proper arguments are provided
        for the scalarizer
        """
        req_args = ["tolerances", "absolutes", "goals"]
        provided_args = list(kwargs.keys())
        missing_args = list(set(req_args).difference(provided_args))

        if not missing_args == []:
            message = f'Missing required Chimera arguments: {", ".join(missing_args)}'
            Logger.log(message, "FATAL")
