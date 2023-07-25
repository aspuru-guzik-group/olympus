#!/usr/bin/env python

import time

from olympus.objects import ParameterVector
from olympus.objects.object_config import Config
from olympus.planners.abstract_planner import AbstractPlanner
from olympus.planners.utils_planner import get_bounds, get_init_guess
from olympus.utils import daemon

# ===============================================================================


class DifferentialEvolution(AbstractPlanner):

    PARAM_TYPES = ["continuous"]

    # defaults are copied from scipy documentation
    # --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    def __init__(
        self,
        goal="minimize",
        args=(),
        strategy="best1bin",
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        callback=None,
        disp=False,
        polish=True,
        init="latinhypercube",
        atol=0,
        updating="immediate",
        workers=1,
        init_guess=None,
        init_guess_method="random",
        init_guess_seed=None,
    ):
        """Finds the global minimum of a multivariate function.
        Differential Evolution is stochastic in nature (does not use gradient
        methods) to find the minimium, and can search large areas of candidate
        space, but often requires larger numbers of function evaluations than
        conventional gradient based techniques.
        The algorithm is due to Storn and Price [1]_.

        Parameters
        ----------
        goal (str): The optimization goal, either 'minimize' or 'maximize'. Default is 'minimize'.
        func : callable
            The objective function to be minimized.  Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a  tuple of any additional fixed parameters needed to
            completely specify the function.
        bounds : sequence
            Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
            defining the lower and upper bounds for the optimizing argument of
            `func`. It is required to have ``len(bounds) == len(x)``.
            ``len(bounds)`` is used to determine the number of parameters in ``x``.
        args : tuple, optional
            Any additional fixed parameters needed to
            completely specify the objective function.
        strategy : str, optional
            The differential evolution strategy to use. Should be one of:
                - 'best1bin'
                - 'best1exp'
                - 'rand1exp'
                - 'randtobest1exp'
                - 'currenttobest1exp'
                - 'best2exp'
                - 'rand2exp'
                - 'randtobest1bin'
                - 'currenttobest1bin'
                - 'best2bin'
                - 'rand2bin'
                - 'rand1bin'
            The default is 'best1bin'.
        maxiter : int, optional
            The maximum number of generations over which the entire population is
            evolved. The maximum number of function evaluations (with no polishing)
            is: ``(maxiter + 1) * popsize * len(x)``
        popsize : int, optional
            A multiplier for setting the total population size.  The population has
            ``popsize * len(x)`` individuals (unless the initial population is
            supplied via the `init` keyword).
        tol : float, optional
            Relative tolerance for convergence, the solving stops when
            ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
            where and `atol` and `tol` are the absolute and relative tolerance
            respectively.
        mutation : float or tuple(float, float), optional
            The mutation constant. In the literature this is also known as
            differential weight, being denoted by F.
            If specified as a float it should be in the range [0, 2].
            If specified as a tuple ``(min, max)`` dithering is employed. Dithering
            randomly changes the mutation constant on a generation by generation
            basis. The mutation constant for that generation is taken from
            ``U[min, max)``. Dithering can help speed convergence significantly.
            Increasing the mutation constant increases the search radius, but will
            slow down convergence.
        recombination : float, optional
            The recombination constant, should be in the range [0, 1]. In the
            literature this is also known as the crossover probability, being
            denoted by CR. Increasing this value allows a larger number of mutants
            to progress into the next generation, but at the risk of population
            stability.
        seed : int or `np.random.RandomState`, optional
            If `seed` is not specified the `np.RandomState` singleton is used.
            If `seed` is an int, a new `np.random.RandomState` instance is used,
            seeded with seed.
            If `seed` is already a `np.random.RandomState instance`, then that
            `np.random.RandomState` instance is used.
            Specify `seed` for repeatable minimizations.
        disp : bool, optional
            Display status messages
        callback : callable, `callback(xk, convergence=val)`, optional
            A function to follow the progress of the minimization. ``xk`` is
            the current value of ``x0``. ``val`` represents the fractional
            value of the population convergence.  When ``val`` is greater than one
            the function halts. If callback returns `True`, then the minimization
            is halted (any polishing is still carried out).
        polish : bool, optional
            If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
            method is used to polish the best population member at the end, which
            can improve the minimization slightly.
        init : str or array-like, optional
            Specify which type of population initialization is performed. Should be
            one of:
                - 'latinhypercube'
                - 'random'
                - array specifying the initial population. The array should have
                  shape ``(M, len(x))``, where len(x) is the number of parameters.
                  `init` is clipped to `bounds` before use.
            The default is 'latinhypercube'. Latin Hypercube sampling tries to
            maximize coverage of the available parameter space. 'random'
            initializes the population randomly - this has the drawback that
            clustering can occur, preventing the whole of parameter space being
            covered. Use of an array to specify a population subset could be used,
            for example, to create a tight bunch of initial guesses in an location
            where the solution is known to exist, thereby reducing time for
            convergence.
        atol : float, optional
            Absolute tolerance for convergence, the solving stops when
            ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
            where and `atol` and `tol` are the absolute and relative tolerance
            respectively.
        updating : {'immediate', 'deferred'}, optional
            If ``'immediate'``, the best solution vector is continuously updated
            within a single generation [4]_. This can lead to faster convergence as
            trial vectors can take advantage of continuous improvements in the best
            solution.
            With ``'deferred'``, the best solution vector is updated once per
            generation. Only ``'deferred'`` is compatible with parallelization, and
            the `workers` keyword can over-ride this option.
            .. versionadded:: 1.2.0
        workers : int or map-like callable, optional
            If `workers` is an int the population is subdivided into `workers`
            sections and evaluated in parallel (uses `multiprocessing.Pool`).
            Supply -1 to use all available CPU cores.
            Alternatively supply a map-like callable, such as
            `multiprocessing.Pool.map` for evaluating the population in parallel.
            This evaluation is carried out as ``workers(func, iterable)``.
            This option will override the `updating` keyword to
            ``updating='deferred'`` if ``workers != 1``.
            Requires that `func` be pickleable.
            .. versionadded:: 1.2.0
        init_guess (array, optional): initial guess for the optimization
        init_guess_method (str): method to construct initial guesses if init_guess is not provided.
            Choose from: random
        init_guess_seed (str): random seed for init_guess_method
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing, then
            OptimizeResult also contains the ``jac`` attribute.
        Notes
        -----
        Differential evolution is a stochastic population based method that is
        useful for global optimization problems. At each pass through the population
        the algorithm mutates each candidate solution by mixing with other candidate
        solutions to create a trial candidate. There are several strategies [2]_ for
        creating trial candidates, which suit some problems more than others. The
        'best1bin' strategy is a good starting point for many systems. In this
        strategy two members of the population are randomly chosen. Their difference
        is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,
        so far:
        .. math::
            b' = b_0 + mutation * (population[rand0] - population[rand1])
        A trial vector is then constructed. Starting with a randomly chosen 'i'th
        parameter the trial is sequentially filled (in modulo) with parameters from
        ``b'`` or the original candidate. The choice of whether to use ``b'`` or the
        original candidate is made with a binomial distribution (the 'bin' in
        'best1bin') - a random number in [0, 1) is generated.  If this number is
        less than the `recombination` constant then the parameter is loaded from
        ``b'``, otherwise it is loaded from the original candidate.  The final
        parameter is always loaded from ``b'``.  Once the trial candidate is built
        its fitness is assessed. If the trial is better than the original candidate
        then it takes its place. If it is also better than the best overall
        candidate it also replaces that.
        To improve your chances of finding a global minimum use higher `popsize`
        values, with higher `mutation` and (dithering), but lower `recombination`
        values. This has the effect of widening the search radius, but slowing
        convergence.
        By default the best solution vector is updated continuously within a single
        iteration (``updating='immediate'``). This is a modification [4]_ of the
        original differential evolution algorithm which can lead to faster
        convergence as trial vectors can immediately benefit from improved
        solutions. To use the original Storn and Price behaviour, updating the best
        solution once per iteration, set ``updating='deferred'``.
        .. versionadded:: 0.15.0
        Examples
        --------
        Let us consider the problem of minimizing the Rosenbrock function. This
        function is implemented in `rosen` in `scipy.optimize`.
        >>> from scipy.optimize import rosen, differential_evolution
        >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
        >>> result = differential_evolution(rosen, bounds)
        >>> result.x, result.fun
        (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
        Now repeat, but with parallelization.
        >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
        >>> result = differential_evolution(rosen, bounds, updating='deferred',
        ...                                 workers=2)
        >>> result.x, result.fun
        (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
        Next find the minimum of the Ackley function
        (https://en.wikipedia.org/wiki/Test_functions_for_optimization).
        >>> from scipy.optimize import differential_evolution
        >>> import numpy as np
        >>> def ackley(x):
        ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
        ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
        ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
        >>> bounds = [(-5, 5), (-5, 5)]
        >>> result = differential_evolution(ackley, bounds)
        >>> result.x, result.fun
        (array([ 0.,  0.]), 4.4408920985006262e-16)
        References
        ----------
        .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
               Efficient Heuristic for Global Optimization over Continuous Spaces,
               Journal of Global Optimization, 1997, 11, 341 - 359.
        .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
        .. [3] http://en.wikipedia.org/wiki/Differential_evolution
        .. [4] Wormington, M., Panaccione, C., Matney, K. M., Bowen, D. K., -
               Characterization of structures from X-ray scattering data using
               genetic algorithms, Phil. Trans. R. Soc. Lond. A, 1999, 357,
               2827-2848
        """
        kwargs = locals().copy()
        del kwargs["self"]

        self.goal = kwargs["goal"]
        for attr in [
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
        from scipy.optimize import differential_evolution

        _ = differential_evolution(
            self._priv_evaluator, bounds=self.bounds, **self.kwargs
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
