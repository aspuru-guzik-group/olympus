#!/usr/bin/env python

import numpy as np

from olympus import Logger
from olympus.scalarizers import AbstractScalarizer


class ConstrainedAsf(AbstractScalarizer):
    def __init__(self, value_space, lowers, uppers, delta_fs):
        AbstractScalarizer.__init__(**locals())

        self.validate_asf_params()

    def scalarize(self, objectives):
        return None

    def validate_asf_params(self):
        if not (len(self.lowers) == len(self.uppers) == len(self.delta_fs)):
            message = "c-ASF parameters not the same length"
            Logger.log(message, "FATAL")
        if not len(self.lowers) == len(self.value_space):
            message = "Number of c-ASF parameters do not match the number of objectives"
            Logger.log(message, "FATAL")

    @staticmethod
    def check_kwargs(kwargs):
        """quick and dirty check to see if the proper arguments are provided
        for the scalarizer
        """
        req_args = ["lowers", "uppers", "delta_fs"]
        provided_args = kwargs.keys()
        missing_args = list(set(req_args).difference(provided_args))
        if not missing_args == []:
            message = (
                f'Missing required c-ASF arguments {", ".join(missing_args)}'
            )
            Logger.log(message, "FATAL")
