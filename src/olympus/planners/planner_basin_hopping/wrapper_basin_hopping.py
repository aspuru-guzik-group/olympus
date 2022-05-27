#!/usr/bin/env python

import time

from olympus.objects import ParameterVector
from olympus.objects.object_config import Config
from olympus.planners import AbstractPlanner
from olympus.planners.utils_planner import get_bounds, get_init_guess
from olympus.utils import daemon

# ===============================================================================


class BasinHopping(AbstractPlanner):

    PARAM_TYPES = ["continuous"]

    # defaults are copied from scipy documentation
    # --> https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.basinhopping.html
    def __init__(
        self,
        goal="minimize",
        niter=100,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs=None,
        take_step=None,
        accept_test=None,
        callback=None,
        interval=50,
        disp=False,
        niter_success=None,
        seed=None,
        init_guess=None,
        init_guess_method="random",
        init_guess_seed=None,
    ):
        """
        Find the global minimum of a function using the basin-hopping algorithm
        Parameters
        ----------
        goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
        func : callable ``f(x, *args)``
            Function to be optimized.  ``args`` can be passed as an optional item
            in the dict ``minimizer_kwargs``
        x0 : ndarray
            Initial guess.
        niter : integer, optional
            The number of basin hopping iterations
        T : float, optional
            The "temperature" parameter for the accept or reject criterion.  Higher
            "temperatures" mean that larger jumps in function value will be
            accepted.  For best results ``T`` should be comparable to the
            separation
            (in function value) between local minima.
        stepsize : float, optional
            initial step size for use in the random displacement.
        minimizer_kwargs : dict, optional
            Extra keyword arguments to be passed to the minimizer
            ``scipy.optimize.minimize()`` Some important options could be:
                method : str
                    The minimization method (e.g. ``"L-BFGS-B"``)
                args : tuple
                    Extra arguments passed to the objective function (``func``) and
                    its derivatives (Jacobian, Hessian).
        take_step : callable ``take_step(x)``, optional
            Replace the default step taking routine with this routine.  The default
            step taking routine is a random displacement of the coordinates, but
            other step taking algorithms may be better for some systems.
            ``take_step`` can optionally have the attribute ``take_step.stepsize``.
            If this attribute exists, then ``basinhopping`` will adjust
            ``take_step.stepsize`` in order to try to optimize the global minimum
            search.
        accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
            Define a test which will be used to judge whether or not to accept the
            step.  This will be used in addition to the Metropolis test based on
            "temperature" ``T``.  The acceptable return values are True,
            False, or ``"force accept"``. If any of the tests return False
            then the step is rejected. If the latter, then this will override any
            other tests in order to accept the step. This can be used, for example,
            to forcefully escape from a local minimum that ``basinhopping`` is
            trapped in.
        callback : callable, ``callback(x, f, accept)``, optional
            A callback function which will be called for all minima found.  ``x``
            and ``f`` are the coordinates and function value of the trial minimum,
            and ``accept`` is whether or not that minimum was accepted.  This can be
            used, for example, to save the lowest N minima found.  Also,
            ``callback`` can be used to specify a user defined stop criterion by
            optionally returning True to stop the ``basinhopping`` routine.
        interval : integer, optional
            interval for how often to update the ``stepsize``
        disp : bool, optional
            Set to True to print status messages
        niter_success : integer, optional
            Stop the run if the global minimum candidate remains the same for this
            number of iterations.
        seed : int or `np.random.RandomState`, optional
            If `seed` is not specified the `np.RandomState` singleton is used.
            If `seed` is an int, a new `np.random.RandomState` instance is used,
            seeded with seed.
            If `seed` is already a `np.random.RandomState instance`, then that
            `np.random.RandomState` instance is used.
            Specify `seed` for repeatable minimizations. The random numbers
            generated with this seed only affect the default Metropolis
            `accept_test` and the default `take_step`. If you supply your own
            `take_step` and `accept_test`, and these functions use random
            number generation, then those functions are responsible for the state
            of their random number generator.
        init_guess (array, optional): initial guess for the optimization
        init_guess_method (str): method to construct initial guesses if init_guess is not provided.
            Choose from: random
        init_guess_seed (str): random seed for init_guess_method

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.  Important
            attributes are: ``x`` the solution array, ``fun`` the value of the
            function at the solution, and ``message`` which describes the cause of
            the termination. The ``OptimzeResult`` object returned by the selected
            minimizer at the lowest minimum is also contained within this object
            and can be accessed through the ``lowest_optimization_result`` attribute.
            See `OptimizeResult` for a description of other attributes.
        See Also
        --------
        minimize :
            The local minimization function called once for each basinhopping step.
            ``minimizer_kwargs`` is passed to this routine.
        Notes
        -----
        Basin-hopping is a stochastic algorithm which attempts to find the global
        minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
        [4]_.  The algorithm in its current form was described by David Wales and
        Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.
        The algorithm is iterative with each cycle composed of the following
        features
        1) random perturbation of the coordinates
        2) local minimization
        3) accept or reject the new coordinates based on the minimized function
           value
        The acceptance test used here is the Metropolis criterion of standard Monte
        Carlo algorithms, although there are many other possibilities [3]_.
        This global minimization method has been shown to be extremely efficient
        for a wide variety of problems in physics and chemistry.  It is
        particularly useful when the function has many minima separated by large
        barriers. See the Cambridge Cluster Database
        http://www-wales.ch.cam.ac.uk/CCD.html for databases of molecular systems
        that have been optimized primarily using basin-hopping.  This database
        includes minimization problems exceeding 300 degrees of freedom.
        See the free software program GMIN (http://www-wales.ch.cam.ac.uk/GMIN) for
        a Fortran implementation of basin-hopping.  This implementation has many
        different variations of the procedure described above, including more
        advanced step taking algorithms and alternate acceptance criterion.
        For stochastic global optimization there is no way to determine if the true
        global minimum has actually been found. Instead, as a consistency check,
        the algorithm can be run from a number of different random starting points
        to ensure the lowest minimum found in each example has converged to the
        global minimum.  For this reason ``basinhopping`` will by default simply
        run for the number of iterations ``niter`` and return the lowest minimum
        found.  It is left to the user to ensure that this is in fact the global
        minimum.
        Choosing ``stepsize``:  This is a crucial parameter in ``basinhopping`` and
        depends on the problem being solved.  Ideally it should be comparable to
        the typical separation between local minima of the function being
        optimized.  ``basinhopping`` will, by default, adjust ``stepsize`` to find
        an optimal value, but this may take many iterations.  You will get quicker
        results if you set a sensible value for ``stepsize``.
        Choosing ``T``: The parameter ``T`` is the temperature used in the
        metropolis criterion.  Basinhopping steps are accepted with probability
        ``1`` if ``func(xnew) < func(xold)``, or otherwise with probability::
            exp( -(func(xnew) - func(xold)) / T )
        So, for best results, ``T`` should to be comparable to the typical
        difference in function values between local minima.
        .. versionadded:: 0.12.0
        References
        ----------
        .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
            Cambridge, UK.
        .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
            the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
            110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.
        .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the
            multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,
            1987, 84, 6611.
        .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,
            crystals, and biomolecules, Science, 1999, 285, 1368.
        Examples
        --------
        The following example is a one-dimensional minimization problem,  with many
        local minima superimposed on a parabola.
        >>> from scipy.optimize import basinhopping
        >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
        >>> x0=[1.]
        Basinhopping, internally, uses a local minimization algorithm.  We will use
        the parameter ``minimizer_kwargs`` to tell basinhopping which algorithm to
        use and how to set up that minimizer.  This parameter will be passed to
        ``scipy.optimize.minimize()``.
        >>> minimizer_kwargs = {"method": "BFGS"}
        >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,
        ...                    niter=200)
        >>> print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
        global minimum: x = -0.1951, f(x0) = -1.0009
        Next consider a two-dimensional minimization problem. Also, this time we
        will use gradient information to significantly speed up the search.
        >>> def func2d(x):
        ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
        ...                                                            0.2) * x[0]
        ...     df = np.zeros(2)
        ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
        ...     df[1] = 2. * x[1] + 0.2
        ...     return f, df
        We'll also use a different local minimization algorithm.  Also we must tell
        the minimizer that our function returns both energy and gradient (jacobian)
        >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
        >>> x0 = [1.0, 1.0]
        >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
        ...                    niter=200)
        >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
        ...                                                           ret.x[1],
        ...                                                           ret.fun))
        global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109
        Here is an example using a custom step taking routine.  Imagine you want
        the first coordinate to take larger steps then the rest of the coordinates.
        This can be implemented like so:
        >>> class MyTakeStep(object):
        ...    def __init__(self, stepsize=0.5):
        ...        self.stepsize = stepsize
        ...    def __call__(self, x):
        ...        s = self.stepsize
        ...        x[0] += np.random.uniform(-2.*s, 2.*s)
        ...        x[1:] += np.random.uniform(-s, s, x[1:].shape)
        ...        return x
        Since ``MyTakeStep.stepsize`` exists basinhopping will adjust the magnitude
        of ``stepsize`` to optimize the search.  We'll use the same 2-D function as
        before
        >>> mytakestep = MyTakeStep()
        >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
        ...                    niter=200, take_step=mytakestep)
        >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
        ...                                                           ret.x[1],
        ...                                                           ret.fun))
        global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109
        Now let's do an example using a custom callback function which prints the
        value of every minimum found
        >>> def print_fun(x, f, accepted):
        ...         print("at minimum %.4f accepted %d" % (f, int(accepted)))
        We'll run it for only 10 basinhopping steps this time.
        >>> np.random.seed(1)
        >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
        ...                    niter=10, callback=print_fun)
        at minimum 0.4159 accepted 1
        at minimum -0.9073 accepted 1
        at minimum -0.1021 accepted 1
        at minimum -0.1021 accepted 1
        at minimum 0.9102 accepted 1
        at minimum 0.9102 accepted 1
        at minimum 2.2945 accepted 0
        at minimum -0.1021 accepted 1
        at minimum -1.0109 accepted 1
        at minimum -1.0109 accepted 1
        The minimum at -1.0109 is actually the global minimum, found already on the
        8th iteration.
        Now let's implement bounds on the problem using a custom ``accept_test``:
        >>> class MyBounds(object):
        ...     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
        ...         self.xmax = np.array(xmax)
        ...         self.xmin = np.array(xmin)
        ...     def __call__(self, **kwargs):
        ...         x = kwargs["x_new"]
        ...         tmax = bool(np.all(x <= self.xmax))
        ...         tmin = bool(np.all(x >= self.xmin))
        ...         return tmax and tmin
        >>> mybounds = MyBounds()
        >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
        ...                    niter=10, accept_test=mybounds)
        """

        kwargs = locals().copy()
        self.goal = kwargs["goal"]
        for attr in [
            "self",
            "goal",
            "init_guess",
            "init_guess_method",
            "init_guess_seed",
        ]:
            del kwargs[attr]

        self.kwargs = kwargs
        self.has_minimizer = False
        self.is_converged = False
        AbstractPlanner.__init__(**locals())

    def _set_param_space(self, param_space):
        self.param_space = param_space
        self.bounds = get_bounds(param_space)
        if self.init_guess is None:
            self.init_guess = get_init_guess(
                param_space,
                method=self.init_guess_method,
                random_seed=self.init_guess_seed,
            )

    def _tell(self, observations):
        self._params = observations.get_params(as_array=False)
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )
        if len(self._values) > 0:
            self.RECEIVED_VALUES.append(self._values[-1])

    def _priv_evaluator(self, params):
        params = self._project_into_domain(params)
        self.SUBMITTED_PARAMS.append(params)
        while len(self.RECEIVED_VALUES) == 0:
            time.sleep(0.1)
        value = self.RECEIVED_VALUES.pop(0)
        return value

    @daemon
    def create_optimizer(self):
        from scipy.optimize import basinhopping

        _ = basinhopping(
            self._priv_evaluator, x0=self.init_guess, **self.kwargs
        )
        self.is_converged = True

    def _ask(self):
        if self.has_minimizer is False:
            self.create_optimizer()
            self.has_minimizer = True
        while len(self.SUBMITTED_PARAMS) == 0:
            time.sleep(0.1)
            if self.is_converged:
                return ParameterVector().from_dict(self._params[-1])
        params = self.SUBMITTED_PARAMS.pop(0)
        return ParameterVector().from_array(params, self.param_space)
