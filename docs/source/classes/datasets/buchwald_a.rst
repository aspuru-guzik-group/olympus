.. _dataset_buchwald_a:

Buchwald A
===========

The ``buchwald`` datasets comprise 5 datasets which each report the yield of Pd-catalyzed Buchwald-Hartwig amination reactions of aryl halides with 4-methylaniline in the presence of varying isoxazole additives, Pd catalyst ligands, and bases obtained via ultra-high-throughput experimentation. Each of the 5 datasets consists of 792 yield measurements. [#f1]_

=============== =========== ============== ========================================
Feature         Kind        Settings       Description
=============== =========== ============== ========================================
aryl halide 	categorical 3 			   aryl halide substrate (with Cl, Br or I) 
additive 		categorical 22 			   isoxazole additive 
base 			categorical 3 			   base used in deprotonation step 
ligand 			categorical 4 			   ligand of Pd catalyst 
=============== =========== ============== ========================================

================= ========== =========
Objective         Kind       Goal
================= ========== =========
yield 			  continuous maximize
================= ========== =========

.. rubric:: Reference

.. [#f1] D. T. Ahneman, J. G. Estrada, S. Lin, S. D. Dreher, and A. G. Doyle. Predicting reaction performance in C–N cross-coupling using machine learning. Science. 360 (2018), 6385.