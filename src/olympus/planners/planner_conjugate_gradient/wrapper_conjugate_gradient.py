#!/usr/bin/env python

import numpy as np

from olympus.objects.object_config import Config
from olympus.planners.wrapper_scipy import WrapperScipy


class ConjugateGradient(WrapperScipy):

    PARAM_TYPES = ["continuous"]

    METHOD = "cg"

    # defaults are copied from scipy documentation
    # --> https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead
    def __init__(
        self,
        goal="minimize",
        disp=False,
        maxiter=None,
        gtol=1e-05,
        norm=np.inf,
        eps=1.4901161193847656e-8,
        init_guess=None,
        init_guess_method="random",
        init_guess_seed=None,
    ):
        """
        Conjugate Gradient optimizer based on the SciPy implementation.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            disp (bool): Set to True to print convergence messages.
            maxiter (int): Maximum number of iterations to perform.
            gtol (float): Gradient norm must be less than gtol before successful termination.
            norm (float): Order of norm (Inf is max, -Inf is min).
            eps (float or ndarray): If jac is approximated, use this value for the step size.
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        """

        WrapperScipy.__init__(**locals())


# ===============================================================================

if __name__ == "__main__":

    from olympus import Parameter, ParameterSpace

    param_space = ParameterSpace()
    param_space.add(Parameter(name="param_0"))
    param_space.add(Parameter(name="param_1"))

    planner = ConjugateGradient()
    planner.set_param_space(param_space=param_space)
    param = planner.ask()
    print("PARAM", param)
