.. _dataset_snar:

SnAr reaction
=============

This dataset reports the e-factor for a nucleophilic aromatic substitution
following the SnAr mechanism. Individual data points encode four process
parameters for a flow reactor to run the reaction, along with the measured
e-factor (defined as the ratio of the mass waste to the mass of product). [#f1]_

The dataset includes 67 samples with four parameters and one objective.

=================== ========== ================ ====================================
Feature             Kind       Settings         Description
=================== ========== ================ ====================================
residence_time      continuous [ 0.5,   2.0]    residence time for flow apparatus [min]
morpholine_equiv    continuous [ 1.0,   5.0]    morpholine equivalence
concentration       continuous [ 0.1,   0.5]    concentration of reagents [M]
temperature         continuous [60.0, 140.0]    temperature of the reactor [Celsius]
=================== ========== ================ ====================================


================= ========== ========
Objective         Kind       Goal
================= ========== ========
E-Factor          continuous minimize
================= ========== ========

The dataset can be extracted from the supporting information of the publication:
https://www.sciencedirect.com/science/article/pii/S1385894718312634

.. rubric:: Reference

.. [#f1] A.M. Schweidtmann, A.D. Clayton, N. Holmes, E. Bradford, R.A. Bourne, A.A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chem. Eng. J. 352 (2018) 277-282.
