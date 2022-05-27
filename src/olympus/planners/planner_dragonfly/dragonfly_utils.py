#!/usr/bin/env python

import numpy as np


def infer_problem_type(param_space):
    """infer the parameter space from Olympus. The three possibilities are
    "fully_continuous", "mixed", or "fully_categorical"

    Args:
            param_space (obj): Olympus parameter space object
    """
    param_types = [p.type for p in param_space]
    if param_types.count("continuous") == len(param_types):
        problem_type = "fully_continuous"
    elif param_types.count("categorical") == len(param_types):
        problem_type = "fully_categorical"
    elif np.logical_and(
        "continuous" in param_types, "categorical" in param_types
    ):
        problem_type = "mixed"
    return problem_type
