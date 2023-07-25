#!/usr/bin/env python

import numpy as np


def _get_init_guess_random(param_space, random_seed=None):
    np.random.seed(random_seed)
    init_guess = []
    for param in param_space:
        if param.type == "continuous":
            init_guess.append(
                np.random.uniform(low=param.low, high=param.high)
            )
    return np.array(init_guess)


def get_init_guess(param_space, method="random", **kwargs):
    if method == "random":
        return _get_init_guess_random(param_space, **kwargs)
    else:
        raise NotImplementedError


def get_bounds(param_space):
    bounds = []
    for param in param_space:
        if param.type == "continuous":
            bounds.append((param.low, param.high))
    return bounds
