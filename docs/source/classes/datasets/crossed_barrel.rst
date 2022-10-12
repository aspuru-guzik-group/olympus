.. _dataset_crossed_barrel:

Crossed Barrel
==============

This dataset reports the yield of undesired product (impurity) in an N-benzylation reaction. Four conditions of this reaction
performed in a flow reactor can be controlled to minimize the yield of impurity. [#f1]_

The dataset includes 73 samples with four parameters and one objective.

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
flow rate       continuous [ 0.2, 0.4]    flow rate [mL/min]
ratio           continuous [   1,   5]    benzyl bromide equivalents
solvent         continuous [ 0.5, 1.0]    solvent equivalents
temperature     continuous [ 110, 150]    reaction temperature [Celsius]
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
yield of impurity continuous minimize
================= ========== ========

.. rubric:: Reference

.. [#f1] A.M. Schweidtmann, A.D. Clayton, N. Holmes, E. Bradford, R.A. Bourne, A.A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chem. Eng. J. 352 (2018) 277-282.