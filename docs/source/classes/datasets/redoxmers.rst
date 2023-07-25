.. _dataset_redoxmers:

Redoxmers
==========

The ``redoxmers`` dataset reports maximum absorption wavelengths, reduction potentials against a Li/Li+ referenceelectrode, and solvation free energies computed using DFT for a dataset of 1408 benzothiadiazole derivatives. The molecules in this dataset are screened as candidates for self-reporting redox-active materials for non-aqeuous redox flow batteries. We provide simple physicochemical descriptors for each of the substituents [#f1]_

=============== =========== ============== ========================================
Feature         Kind        Settings       Description
=============== =========== ============== ========================================
R1 substituent  categorical 2              substituted group at R1
R2 substituent  categorical 8              substituted group at R2
R3 substituent  categorical 8              substituted group at R3
R4 substituent  categorical 11             substituted group at R4
=============== =========== ============== ========================================

========================= ========== ========
Objective           	  Kind       Goal
========================= ========== ========
max absorption difference continuous minimize
reduction potential       continuous minimize
solvation free energy     continuous minimize
========================= ========== ========

.. rubric:: Reference

.. [#f1] G. Agarwal, H. A. Doan, L. A. Robertson, L. Zhang, and R. S. Assary. Discovery of Energy Storage Molecular Materials Using Quantum Chemistry-Guided Multiobjective Bayesian Optimization. Chemistry of Materials. 33 (2021) 8133–8144.