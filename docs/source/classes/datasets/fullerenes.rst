.. _dataset_fullerenes:

Buckminsterfullerene adducts
============================

This dataset reports the production of o-xylenyl adducts of
Buckminsterfullerenes. Three process conditions (temperature, reaction time and
ratio of sultine to C60) are varied to maximize the mole fraction of the desired
product. Experiments are executed on a three factor fully factorial grid with
six levels per factor. [#f1]_

The dataset includes 246 samples with three parameters and one objective.

=============== ========== ============== ========================================
Feature         Kind       Settings       Description
=============== ========== ============== ========================================
reaction_time   continuous [  3,  31]     reaction time in flow reactor [min]
sultine_conc    continuous [  1.5, 6.0]   relative concentration of sultine to C60
temperature     continuous [100, 150]     temperature of the reaction [Celsius]
=============== ========== ============== ========================================

================= ========== ========
Objective         Kind       Goal
================= ========== ========
mole fraction     continuous maximize
================= ========== ========

The dataset can be downloaded from:
https://imperialcollegelondon.app.box.com/v/tuning-reaction-products

.. rubric:: Reference

.. [#f1] B.E. Walker, J.H. Bannock, A.M. Nightingale, J.C. deMello. Tuning reaction products by constrained optimisation. React. Chem. Eng., (2017), 2, 785-798.
