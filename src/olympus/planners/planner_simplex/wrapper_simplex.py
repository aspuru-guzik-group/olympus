#!/usr/bin/env python

from olympus.planners.wrapper_scipy import WrapperScipy
from olympus.objects.object_config import Config


class Simplex(WrapperScipy):

    METHOD = 'Nelder-Mead'

    # defaults are copied from scipy documentation
    # --> https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead
    def __init__(self, goal='minimize', disp=False, maxiter=None, maxfev=None, initial_simplex=None, xatol=0.0001, fatol=0.0001,
                 adaptive=False, init_guess=None, init_guess_method='random', init_guess_seed=None):
        """
        Nelder-Mead simplex algorithm. Implementation from SciPy.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            disp (bool): Set to True to print convergence messages.
            maxiter (int): Maximum allowed number of iterations. Will default
                to N*200, where N is the number of variables, if neither maxiter
                or maxfev is set. If both maxiter and maxfev are set,
                minimization will stop at the first reached.
            maxfev (int): Maximum allowed number of function evaluations. Will
                default to N*200, where N is the number of variables, if neither
                maxiter or maxfev is set. If both maxiter and maxfev are set,
                minimization will stop at the first reached.
            initial_simplex (array_like of shape (N + 1, N)): Initial simplex.
                If given, overrides x0. initial_simplex[j,:] should contain the
                coordinates of the j-th vertex of the N+1 vertices in the
                simplex, where N is the dimension.
            xatol (float, optional): Absolute error in xopt between iterations
                that is acceptable for convergence.
            fatol (number, optional): Absolute error in func(xopt) between
                iterations that is acceptable for convergence.
            adaptive (bool, optional): Adapt algorithm parameters to
                dimensionality of problem. Useful for high-dimensional
                minimization.
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        """

        WrapperScipy.__init__(**locals())
