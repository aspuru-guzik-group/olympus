.. _dataset_suzuki_i:

Suzuki I
=========

The datasets ``suzuki_i−suzuki_iv`` report yeild and catalyst turnover number for flow-based Suzuki-Miyaura
cross-coupling reactions with varying substrates. There are three continous parameters (temperature, residence
time, and catalyst loading) and one categorical parameter (Pd catalyst ligand). The objective is to simultaneously
maximize both the yield and catalyst turnover number. [#f1]_

================ =========== =============== ========================================
Feature          Kind        Settings        Description
================ =========== =============== ========================================
ligand           categorical 8 				 Pd catalyst ligand 
res time         continuous  [60.0 − 600.0]  reaction residence time [s] 
temperature      continuous  [30.0 − 110.0]  reaction temperature [◦C]
catalyst loading continuous  [0.498 − 2.515] catalyst loading fraction [a.u.] 
================ =========== =============== =======================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
yield 			  continuous maximize
turnover		  continuous maximize
================= ========== ========

.. rubric:: Reference

.. [#f1] B. J. Reizman, Y.-M. Wang, S. L. Buchwald, and K. F. Jensen,. Suzuki–miyaura cross-coupling optimization enabled by automated feedback. Reaction chemistry & engineering. 1 (2016), 658–666.