.. _custom_scalarizers:


Custom Scalarizers
==================
You can integrate your own achievement scalarizing function within **Olympus** in a similar way as one would implement a custom ``Planner`` or ``Emulator``. Here, we will demonstate how to implement a custom ``Scalarizer``, which will be a simple weighted product approach. Given a set of :math:`n` objective functions :math:`\{ f_1, \ldots, f_n\}` all to be minimized, the weighted product scalarizer is defined according to a vector :math:`\mathbf{w} \in \mathbb{R}^n`, 


:math:`J \left( \mathbf{y};\mathbf{w}\right) = \prod_{i=1}^n y_i^{w_i}`.

Custom ``Scalarizers`` in **Olympus** must inherit from the ``AbstractScalarizer`` class and implement two abstract methods, ``validate_asf_params`` and ``scalarize``. 


``validate_asf_params``
-----------------------

This method should check wether or not the proper arguments were passed to the instance of the `Scalarizer`. For our weighted product implementation, we are only expecting a vector of `weights`, we can simply assert that the weights are postive values, and that we have one weight corresponding to each objective.

```
def validate_asf_params(self):
	if not np.all(np.array(self.weights) >= 0.0):
	    message = (
	        "Weighted sum ASF weights must be non-negative real numbers"
	    )
	    Logger.log(message, "FATAL")
	if not len(self.weights) == len(self.value_space):
	    message = (
	        "Number of weights does not match the number of objectives"
	    )
	    Logger.log(message, "FATAL")

```