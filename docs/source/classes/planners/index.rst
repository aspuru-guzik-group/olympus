.. _planners:

Planners
========

**Olympus** provides wrappers for a number of algorithms. These can be accessed by importing the specific planner class
from the ``olympus.planners`` module, or via the ``Planner`` function. For instance, to load the **GPyOpt** algorithm,
you can use the ``Planner`` function::

    from olympus.planners import Planner
    planner = Planner(kind='Gpyopt')

The above is equivalent to importing the class ``Gpyopt`` directly::

    from olympus.planners import Gpyopt
    planner = Gpyopt()

This latter approach, however, allows for more control over the algorithm behaviour via the input arguments of the ``Gpyopt``
class.

Here below are all the planners available in **Olympus**.

Bayesian Algorithms
-------------------

These algorithms employ a sequential model-based approach for the global optimization of black-box functions without
requiring derivatives. More specifically, the function to be optimized is approximated by a surrogate model that is
refined as more data is collected. Based on this model, an acquisition function that evaluates the utility of candidate
points can be defined, leading to the balanced exploration and exploitation of the search space of interest.

.. toctree::
   :maxdepth: 1

   gpyopt
   hyperopt
   phoenics

Evolutionary Algorithms
-----------------------

These are population and heuristic based algorithms inspired by biological evolution. They mimic mechanisms such as
reproduction, mutation, recombination, and selection to iteratively improve the fitness of a population. Each individual
in the population represent a point in the search space, and their fitness corresponds to the objective evaluated at
that point. Similar to Bayesian optimization algorithms, no gradient information is required and they can be used for
the global optimization of black-box functions.

.. toctree::
   :maxdepth: 1

   deap
   cma
   particle_swarms
   differential_evolution

Gradient Methods
----------------

These algorithms use derivative information (gradient or Hessian) at the current point to determine the location of the
next point to be evaluated. These approaches are efficient on convex optimization problems, but are not guaranteed to
find the global optimum.

.. toctree::
   :maxdepth: 1

   steepest_descent
   conjugate_gradient
   lbfgs
   slsqp


Grid-Like Searches
------------------

These algorithms define a set of points in parameter space to be evaluated at the start of the optimization campaign.
Thus, the number of points to be evaluated needs to be chosen in advance. At every step of the campaign, the next point
to be evaluated is chosen deterministically. While being a global optimization strategies, their cost scales
exponentially with the number of parameters. Alternatives to standard full grid approaches involve the use of low
discrepancy sequences to sample more effectively high dimensional spaces.


.. toctree::
   :maxdepth: 1

   grid
   latin_hypercube
   sobol
   random_search


Others
------

These are algorithms the do not easily fit the categories above.

.. toctree::
   :maxdepth: 1

   simplex
   snobfit
   basin_hopping


Planner Function
----------------

.. currentmodule:: olympus.planners

.. autofunction:: Planner
   :noindex:
