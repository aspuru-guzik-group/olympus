Welcome to Olympus' documentation!
==================================

.. image:: https://travis-ci.com/FlorianHase/olympus.svg?token=rULvnKYmWdFF3JqQBVVW&branch=dev
    :target: https://travis-ci.com/FlorianHase/olympus
.. image:: https://codecov.io/gh/FlorianHase/olympus/branch/flo/graph/badge.svg?token=FyvePgBDQ5
    :target: https://codecov.io/gh/FlorianHase/olympus

**Olympus**  is a toolkit that provides a consistent and easy-to-use framework to access a number of optimization
algorithms (:ref:`planners`) as well as benchmark surfaces. In addition to analytic benchmark functions (:ref:`surfaces`),
a collection of experimental emulators (:ref:`emulators`) are provided. These are based on models (:ref:`models`) trained on experimental
datasets (:ref:`datasets`) from across the natural sciences, including examples from physics, chemistry, and materials science.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   overview
   install
   credits
   support


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/notebooks/use_emulators
   examples/notebooks/simple_benchmark
   examples/notebooks/larger_benchmarks
   examples/notebooks/planners_interface
   examples/notebooks/noisy_inputs
   examples/notebooks/custom_dataset
   examples/notebooks/custom_planner


.. toctree::
   :maxdepth: 1
   :caption: Core Classes

   classes/planners/index
   classes/datasets/index
   classes/models/index
   classes/emulators
   classes/scalarizers/index
   classes/surfaces/index
   classes/noises/index
   classes/plotter/index


.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage

   custom_emulators
   custom_planners
   custom_scalarizers


.. toctree::
   :maxdepth: 10
   :caption: Complete API Reference

   apidoc/olympus


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
