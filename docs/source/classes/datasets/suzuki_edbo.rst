.. _dataset_suzuki_edbo:

Suzuki EDBO
=============

The ``suzuki_edbo`` dataset reports yields for a Suzuki-Miyaura coupling reaction preformed on the nanomole scale
using a automated flow-based synthesis platform. The reaction scheme, along with the search space is presented
in Figure S5. This dataset consists of 5 categorical parmeters: the boronic acid derivative electrophile, aryl halide
nucleophile, base used in the deprotonation step, Pd catalyst ligand, and solvent. Collectively, there are 3696 unique
reactions. [#f1]_ [#f2]_

=============== =========== ============== ========================================
Feature         Kind        Settings       Description
=============== =========== ============== ========================================
electrophile    categorical 4 			   boronic acid derivative 
nucleophile     categorical 3 			   aryl halide 
base 			categorical 7 			   base used in deprotonation step 
ligand 			categorical 11 			   Pd catalyst ligand 
solvent 		categorical 4 			   solvent 
=============== =========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
yield             continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] A.-C. Bédard, A. Adamo, K. C. Aroh, M. G. Russell, A. A. Bedermann, J. Torosian, B. Yue, K. F. Jensen, and T. F. Jamison. Reconfigurable system for automated optimization of diverse chemical reactions. Science.361 (2018), 1220–1225.

.. [#f2] B. J. Shields, J. Stevens, J. Li, M. Parasram, F. Damani, J. I. M. Alvarado, J. M. Janey, R. P. Adams, and A. G. Doyle. Bayesian reaction optimization as a tool for chemical synthesis. Nature. 590 (2021), 89–96.