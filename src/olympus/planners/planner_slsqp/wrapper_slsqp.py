#!/usr/bin/env python

from olympus.planners.wrapper_scipy import WrapperScipy
from olympus.objects.object_config import Config


class Slsqp(WrapperScipy):

    METHOD = 'slsqp'

    # defaults are copied from scipy documentation
    # --> https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
    def __init__(self, goal='minimize',  disp=False, eps=1.4901161193847656e-8, ftol=1e-6, maxiter=15000,
                 init_guess=None, init_guess_method='random', init_guess_seed=None):
        """
        Sequential Least SQuares Programming (SLSQP) optimizers. SciPy implementation.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            disp (bool): Set to True to print convergence messages. If False, verbosity is ignored and set to 0.
            eps (float): Step size used for numerical approximation of the Jacobian.
            ftol (float): Precision goal for the value of f in the stopping criterion.
            maxiter (int): Maximum number of iterations.
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        """

        WrapperScipy.__init__(**locals())
