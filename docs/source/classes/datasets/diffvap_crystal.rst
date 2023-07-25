.. _dataset_diffvap_crystal:

Vapour Diffusion Crystallization
=================================

This dataset reports the results for high-throughput antisolvent vapor diffusion crystallization
of metal halide perovskitoids. This dataset contains 918 experimental measurements and experiment details can be
found in previous studies on perovskite formation. [#f1]_ [#f2]_

The dataset includes 918 samples with 10 parameters and 1 objective. The objective for this dataset is an ordinal variable called ``crystal_score``. The options for the objective value are (in order of increasing merit) ``clear_solution``, ``fine_powder``, ``small_crystallites``, ``large_crystallites``.

==================== =========== ==================== ========================================
Feature              Kind        Settings       	  Description
==================== =========== ==================== ========================================
organic              categorical 17    	      		  organic halide identity
orgaic molarity      continuous  [0.01434, 7.3935]    concentration organic halide (mol/L)
solvent              categorical 3    				  solvent identity
solvent molarity     continuous  [1.05569, 12.79558]  concentration solvent (mol/L)
inorganic molarity   continuous  [0.0, 2.26115]       concentration inorganic halide (mol/L)
acid molarity        continuous  [0.0, 22.42276]      concentration acid (mol/L)
alpha vial volume    continuous  [0.000149, 0.000744] volume of alpha vial (L)
beta vial volume     continuous  [0.001, 0.0008]      volume of beta vial (L)
reaction time        discrete    3                    vapour diffusion time (s)
reaction temperature discrete    3     				  vapour diffusion temperature (Celsius)
==================== =========== ==================== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
crystal score     ordinal    minimize
================= ========== ========

.. rubric:: Reference

.. [#f1] Z. Li, P. W. Nega, M. A. N. Nellikkal, C. Dun, M. Zeller, J. J. Urban, W. A. Saidi, J. Schrier, A. J. Norquist, E. M. Chan. Dimensional Control over Metal Halide Perovskite Crystallization Guided by Active Learning. Chemistry of Materials. 34, (2022) 756–767.
.. [#f2] N. T. P. Hartono, M. Ani Najeeb, Z. Li, P. W. Nega, C. A. Fleming, X. Sun, E. M. Chan, A. Abate, A. J. Norquist, J. Schrier, T. Buonassisi. Principled Exploration of Bipyridine and Terpyridine Additives to Promote Methylammonium Lead Iodide Perovskite Crystallization. Crystal Growth & Design. 22 (2022) 5424–5431.
