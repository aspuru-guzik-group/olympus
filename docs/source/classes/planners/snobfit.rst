.. _planner_snobfit:

SNOBFIT
-------

This algorithm combines global and local search by branching and local fits, and can be used to solve the noisy
optimization of an expensive objective function.

This planner uses the ``SQSnobFit`` library, which needs to be installed if you want to use this algorithm. For more
information please visit the `SNOBFIT website <https://www.mat.univie.ac.at/~neum/software/snobfit/>`_.

.. currentmodule:: olympus.planners

.. autoclass:: Snobfit
   :noindex:
   :exclude-members: add, from_dict, generate, get, to_dict


   .. rubric:: Methods

   .. autosummary::
      tell
      ask
      recommend
      optimize
      set_param_space
