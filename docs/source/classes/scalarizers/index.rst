.. _scalarizers:


Scalarizers
===========

A popular approach to multi-objective optimization is to construct one
single objective function from multiple. This mapping uses user-provided preferences about the optimization goal for each objective. For instance, one may formulate a single objective from measurements of efficacy and toxicity. Functions which perform this mapping are known as achievement scalarizing functions (ASFs), whose optimal solution ideally corresponds to the Pareto optimal
solution of the multi-objective optimization problem. 

**Olympus** provides wrappers for 4 different ASFs, namely ``WeightedSum``, ``ParEGO``, ``Chimera`` and ``Hypervolume``. ASFs can be accessed by importing the specific ASF class from the ``olympus.scalarizers`` module, or by interacting with the ``Scalarizer`` function. For example, to load the **Hypervolume** ASF, one can use the ``Scalarizer`` function as follows::

	from olympus.scalarizers import Scalarizer
	asf = Scalarizer(
		kind='Hypervolume', 
		value_space=emulator.value_space,
		goals=['min', 'min'],
	)

This is equivalent to importing directly the ``Hypervolume`` class::

	from olympus.scalarizers import Hypervolume
	asf = Hypervolume( 
		value_space=emulator.value_space,
		goals=['min', 'min'],
	)

Below we list specific details about each of the ASFs provided in **Olympus**.


Weighted Sum
------------

The ``WeightedSum`` ASF maps multiple objectives onto a cumualtive scalar objective using a vector of weights to produce a weighted sum. This ASF takes as input a list of weight values::

	from olympus.scalarizers import Scalarizer
	asf = Scalarizer(
		kind='WeightedSum', 
		value_space=emulator.value_space,
		goals=['min', 'min'],
		weights=[2., 1.],
	)


ParEGO
------

``ParEGO`` was introduced to extend the ``EGO`` algorithm to a multi-objective optimization setting. Similar to converts objective values to a single cumulative function via a parameterized scaling weight vector whose value is sampled uniformly at each optimization iteration. he scalar merit of an objective function is computed using the augmented Tchebycheff function. The ``ParEGO`` function takes as input a small positive number, which defaults to 0.05 in **Olympus**::


	from olympus.scalarizers import Scalarizer
	asf = Scalarizer(
		kind='ParEGO', 
		value_space=emulator.value_space,
		goals=['min', 'min'],
		rho=0.05,
	)



Chimera
-------

The ``Chimera`` ASF combines *a priori* scalarizing with lexicographic approaches. ``Chimera`` allows users to organize the multiple objectives in a hierarchy of importance, as well as allowing definiton of tolerance values on the objectives. To specify absolute tolerances, one can instantiate the ASF as::

	from olympus.scalarizers import Scalarizer
	asf = Scalarizer(
		kind='Chimera', 
		value_space=emulator.value_space,
		goals=['min', 'min'],
		tolerances=[0.5, 0.8],
		absolutes=[True, True],
	)

To specify *relative* tolerances, one should set the ``absolutes`` argument to False. More information on `Chimera` can be found in the original `publication <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c8sc02239a>`_.



Hypervolume Indicator
---------------------

The hypervolume indicator is an example of a set-quality indicator, which facilitate assessment of Pareto fronts by summarizing their characteristics (such as proximity to the Pareto front, diversity and spread) with a single scalar value. Owing to its ease of interpretation, hypervolume is one of the most widely employed set-quality indicators. The hypervolume indicator maps a set of objective values to a measure of the region dominated by that set and bounded above by some reference point. Intuitively, the indicator provides a notion of the size of the covered objective space or the size of the dominated space. Besides the individual optimization goals for each objective, ``Hypervolume`` takes no additonal arguments::

	from olympus.scalarizers import Scalarizer
	asf = Scalarizer(
		kind='Hypervolume', 
		value_space=emulator.value_space,
		goals=['min', 'min'],
	)




