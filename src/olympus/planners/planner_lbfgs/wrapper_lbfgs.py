#!/usr/bin/env python

from olympus.objects.object_config import Config
from olympus.planners.wrapper_scipy import WrapperScipy


class Lbfgs(WrapperScipy):

    PARAM_TYPES = ["continuous"]

    METHOD = "L-BFGS-B"
    KNOWS_BOUNDS = True

    # defaults are copied from scipy documentation
    # --> https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    def __init__(
        self,
        goal="minimize",
        disp=None,
        eps=1e-8,
        ftol=2.220446049250313e-9,
        gtol=1e-5,
        maxcor=10,
        maxfun=15000,
        maxiter=15000,
        maxls=20,
        init_guess=None,
        init_guess_method="random",
        init_guess_seed=None,
    ):
        """
        L-BFGS-B optimizer based on the SciPy implementation.

        Args:
            goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
            disp (None or int): If disp is None (the default), then the supplied
                version of iprint is used. If disp is not None, then it
                overrides the supplied version of iprint with the behaviour you
                outlined.
            eps (float): Step size used for numerical approximation of the
                jacobian.
            ftol (float): The iteration stops when
                (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
            gtol (float): The iteration will stop when
                max{|proj g_i | i = 1, ..., n} <= gtol where pg_i is the i-th
                component of the projected gradient.
            maxcor (int): The maximum number of variable metric corrections used
                to define the limited memory matrix. (The limited memory BFGS
                method does not store the full hessian but uses this many terms
                in an approximation to it.)
            maxfun (int): Maximum number of function evaluations.
            maxiter (int): Maximum number of iterations.
            maxls (int, optional): Maximum number of line search steps (per
                iteration).
            init_guess (array, optional): initial guess for the optimization
            init_guess_method (str): method to construct initial guesses if init_guess is not provided.
                Choose from: random
            init_guess_seed (str): random seed for init_guess_method
        """

        WrapperScipy.__init__(**locals())
