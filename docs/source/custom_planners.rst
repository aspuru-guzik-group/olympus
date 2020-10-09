.. _custom_planners:

Custom Planners
===============
You can integrate your own optimization algorithm within **Olympus**. A simple example where we create a ``Planner`` that
implements a random sampler can be found among the :ref:`examples`.

In brief, one can implement custom algorithms by inheriting from the `CustomPlanner` class::

   from olympus.planners import CustomPlanner

Here, we will discuss in more detail the methods implemented in this class to allow the user to customize it as needed.

``CustomPlanner`` class already comes with ``__init__``, ``_set_param_space``, ``_tell`` methods. This is important because all
``Planner`` instances in **Olympus** are expected to have these three methods, as well as an ``_ask`` method. Thus, at a minimum a custom
planner class needs to implement the ``_ask`` method, while expanding upon the default ``__init__``, ``_set_param_space``, ``_tell``
methods is optional. Here below we provide a quick summary of what these four methods should do.

``__init__``
------------
This method works as in all Python classes, and for any ``Planner`` in **Olympus**, the minimal ``__init__`` should look like this::

   def __init__(self, goal='minimize'):
       AbstractPlanner.__init__(**locals())

Where ``goal`` is a required "abstract attribute" and the parent ``AbstractPlanner`` class should be initialised with all
variables in the local namespace. This is the default implemented in ``CustomPlanner``. A customized ``__init__`` method
could contain additional arguments specific to your algorithms. For instance, it could take the ``random_seed`` argument
if the algorithm is stochastic.

``_set_param_space``
--------------------
This method defines the domain for the optimization. Only points within this search space should be queried by the algorithm.
This is what is implemented by default in ``CustomPlanner``::

   def _set_param_space(self, param_space):
       self._param_space = []
       for param in param_space:
           if param.type == 'continuous':
               param_dict = {'name': param.name, 'type': param.type, 'domain': (param.low, param.high)}
               self._param_space.append(param_dict)

Basically, it creates a ``_param_space`` attribute that is a list of dictionaries, with the name of the parameters, their
type (only 'continuous' is currently supported) and their bounds. You can then use this attribute to make sure your
algorithm samples points within the defined optimisation domain.

``_tell``
---------
This method provides the ``Planner`` instance with information about the history of the optimization, which may be used
to select the next query point. This is what is implemented by default in ``CustomPlanner``::

   def _tell(self, observations):
       self._params = observations.get_params(as_array=True)
       self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)

It defined ``_params`` and ``_values`` attributes that contain all previous queried parameter locations and their merit, respectively.
If the ``Planner`` was initialised with ``goal='minimize'``, then ``self.flip_measurements`` will be ``False``. If it
was initialised with ``goal='maximize'``, then ``self.flip_measurements`` will be ``True``, which means that the ``_values``
will contain the merits multiplied by -1. This is because ``Planner`` instances are expected to always aim at minimizing the
objective function.

``_ask``
--------
This is the only method that has no implementation in ``CustomPlanner`` and always needs to be implemented by the user.
What this method is expected to do is to decide which next set of parameters to query, and return this as a ``ParameterVector``
instance. This is an example in which we iterate over all dimensions of parameter space to sample the space at random::

   def _ask(self):
       new_params = []  # list with the parameter values for the next query location
       # go through all dimensions of parameter space
       for param in self._param_space:
           # sample uniformly at random within the bounds of the specific dimension
           new_param = np.random.uniform(low=param['domain'][0], high=param['domain'][1])
           # append to the list with the parameter values
           new_params.append(new_param)

       # return new_params as a ParameterVector object
       return ParameterVector(array=new_params, param_space=self.param_space)

In this case, the next sample does not depend on the history of the search, so we have not used ``_params`` and ``_values``,
but if you are implementing, e.g., a Bayesian optimization algorithm, you can used these attributes too.