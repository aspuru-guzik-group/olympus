.. _planner_basin_hopping:

Basin Hopping
-------------

Basin-hopping is a two-phase method that combines a global stepping algorithm with local minimization at each step.
As the step-taking, step acceptance, and minimization methods are all customizable, this function can also be used to
implement other two-phase methods.

This planner is based on the ``scipy.optimize.basinhopping`` implementation and as such requires you to have the
``scipy`` library installed. For more information please visit the `SciPy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping>`_.


.. currentmodule:: olympus.planners

.. autoclass:: BasinHopping
   :noindex:
   :exclude-members: add, from_dict, generate, get, to_dict


   .. rubric:: Methods

   .. autosummary::
      tell
      ask
      recommend
      optimize
      set_param_space
